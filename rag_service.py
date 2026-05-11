# -*- coding: utf-8 -*-
"""
RAG 核心服务：加载 FAISS + 元数据、向量检索、拼装 Prompt、调用对话模型生成回答。

模块职责划分：
- 全局缓存 `_index` / `_metadata`：避免每次请求重复从磁盘读 FAISS（可通过 clear_index_cache 失效）。
- get_text_embedding：与 build_index 使用同一嵌入模型，保证「建索引」与「查索引」在同一向量空间。
- search / search_with_paths：前者服务线上问答；后者按给定索引路径检索，供版本回归不加污染全局缓存。
- answer_question：完整 RAG 流水线（检索 → 构造上下文 Prompt → LLM → 整理 sources 返回前端）。
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import dashscope
import faiss
import numpy as np
from http import HTTPStatus
from openai import OpenAI

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import (  # noqa: E402
    CHAT_MODEL,
    INDEX_FILE,
    METADATA_FILE,
    MULTIMODAL_EMBEDDING_MODEL,
    RAG_TOP_K,
)
from env_loader import require_dashscope_key  # noqa: E402


# 进程内单例缓存：首次检索或问答时加载；重建索引文件后应调用 clear_index_cache()
_index = None
_metadata = None


def clear_index_cache():
    """
    清空内存中的 FAISS 索引与元数据缓存。

    典型场景：在服务器不重启的情况下运行了 build_index.py 覆盖磁盘文件后，
    调用本函数使下一次检索重新 read_index / load json。
    """
    global _index, _metadata
    _index = None
    _metadata = None


def _ensure_loaded():
    """懒加载：仅在首次需要时读入 INDEX_FILE 与 METADATA_FILE。"""
    global _index, _metadata
    if _index is not None and _metadata is not None:
        return
    if not os.path.isfile(INDEX_FILE) or not os.path.isfile(METADATA_FILE):
        raise FileNotFoundError(
            f"未找到索引文件。请先在本目录运行: python build_index.py\n"
            f"期望路径: {INDEX_FILE}"
        )
    _index = faiss.read_index(str(INDEX_FILE))
    with open(str(METADATA_FILE), "r", encoding="utf-8") as f:
        _metadata = json.load(f)


def get_text_embedding(text: str):
    """
    将查询文本编码为与索引向量同维度的向量（DashScope 多模态嵌入·文本）。

    Args:
        text: 用户问题或任意待检索文本。

    Returns:
        list[float]: 查询向量。
    """
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{"text": text}],
    )
    if resp.status_code != HTTPStatus.OK:
        raise RuntimeError(f"Embedding 失败: {getattr(resp, 'message', resp)}")
    return resp.output["embeddings"][0]["embedding"]


def distance_to_similarity(distance: float) -> float:
    """
    将 FAISS L2 距离映射为便于展示的「相似度」标量（非概率，仅相对比较）。

    公式：1/(1+d)，距离越小相似度越大。
    """
    return float(1.0 / (1.0 + distance))


def search_with_paths(
    query: str,
    index_path: str | Path,
    metadata_path: str | Path,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    使用任意路径下的 FAISS 与元数据 JSON 做检索，不读写模块全局缓存。

    用途：kb_versions 对比两套索引时，避免替换进程内 `_index` 影响线上问答。

    Args:
        query: 查询文本。
        index_path: .faiss 文件路径。
        metadata_path: 与索引行号对齐的 metadata JSON 路径。
        top_k: 返回条数上限；默认取 config.RAG_TOP_K。

    Returns:
        每项含 metadata（原始条目 dict）、distance（L2）、similarity（映射后）。
    """
    api_key = require_dashscope_key()
    dashscope.api_key = api_key

    index_path = Path(index_path)
    metadata_path = Path(metadata_path)
    if not index_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"索引或元数据不存在: {index_path} / {metadata_path}")

    index = faiss.read_index(str(index_path))
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_local = json.load(f)

    k = top_k if top_k is not None else RAG_TOP_K
    k = min(k, index.ntotal)
    if k <= 0:
        return []

    qvec = np.array([get_text_embedding(query)]).astype("float32")
    distances, indices = index.search(qvec, k)

    out = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        meta_row = meta_local[idx]
        out.append(
            {
                "metadata": meta_row,
                "distance": float(dist),
                "similarity": distance_to_similarity(float(dist)),
            }
        )
    return out


def search(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """
    使用当前线上默认索引（INDEX_FILE + METADATA_FILE）执行向量检索。

    Args:
        query: 用户问题。
        top_k: 检索条数；None 时用 RAG_TOP_K，且不超过索引总向量数。

    Returns:
        检索命中列表，结构同 search_with_paths。
    """
    _ensure_loaded()
    k = top_k if top_k is not None else RAG_TOP_K
    k = min(k, _index.ntotal)

    qvec = np.array([get_text_embedding(query)]).astype("float32")
    distances, indices = _index.search(qvec, k)

    out = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        meta = _metadata[idx]
        out.append(
            {
                "metadata": meta,
                "distance": float(dist),
                "similarity": distance_to_similarity(float(dist)),
            }
        )
    return out


def build_llm_prompt(query: str, hits: list[dict[str, Any]]) -> str:
    """
    将 Top-K 检索结果格式化为单条 user Prompt，约束模型严格依据引用作答。

    Args:
        query: 用户原始问题。
        hits: search() 返回的列表。

    Returns:
        发往对话模型的用户消息全文。
    """
    parts = []
    for i, h in enumerate(hits):
        m = h["metadata"]
        sim = h["similarity"]
        src = m.get("source", "")
        body = m.get("content", "")
        parts.append(
            f"参考资料 {i + 1}（来源: {src}，相关度: {sim:.4f}）:\n{body}\n"
        )
    ctx = "\n".join(parts)
    return f"""你是上海迪士尼度假区知识助手。请严格依据下列参考资料回答用户问题；若资料中没有相关信息，请诚实说明，不要编造。

{ctx}

用户问题：{query}

请用条理清晰的中文作答。"""


def answer_question(query: str, top_k: int | None = None) -> dict[str, Any]:
    """
    RAG 问答入口：检索 → 组装 Prompt → 通义对话模型生成 → 返回答案与引用摘要。

    Args:
        query: 用户问题。
        top_k: 检索条数。

    Returns:
        dict 包含 answer（字符串）、sources（列表，每项含 type/source/similarity/preview/可选 rel_image）。
    """
    api_key = require_dashscope_key()
    dashscope.api_key = api_key

    hits = search(query, top_k=top_k)
    if not hits:
        return {
            "answer": "知识库中暂无匹配内容，请先运行 build_index.py 构建索引。",
            "sources": [],
        }

    prompt = build_llm_prompt(query, hits)
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是专业的上海迪士尼度假区知识助手，回答要准确、友好。",
            },
            {"role": "user", "content": prompt},
        ],
    )
    answer = completion.choices[0].message.content or ""

    # 前端卡片展示：截断 content 预览；图片类型附带 rel_image 拼静态路由
    sources = []
    for h in hits:
        m = h["metadata"]
        item = {
            "type": m.get("type"),
            "source": m.get("source"),
            "similarity": round(h["similarity"], 4),
            "preview": (m.get("content") or "")[:200],
        }
        if m.get("rel_image"):
            item["rel_image"] = m["rel_image"]
        sources.append(item)

    return {"answer": answer.strip(), "sources": sources}
