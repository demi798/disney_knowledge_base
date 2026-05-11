# -*- coding: utf-8 -*-
"""
知识片段 → 多样化问题生成 → 落盘问题库 → 供 build_index 检索增强。

要点：
- 输入来源为已构建的 `disney_metadata.json`（每条 row 含 id、type、content）。
- chunk_id 与 metadata 中 id 一致，便于 augment_content_for_index 按 id 拼接衍生问题。
- 知识库文档或切块变化后应重新「生成问题库」并重建索引，否则 id 与内容可能错位。
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (  # noqa: E402
    CHAT_MODEL,
    QUESTION_BANK_FILE,
    QUESTIONS_PER_CHUNK_MAX,
)
from env_loader import require_dashscope_key  # noqa: E402

from openai import OpenAI  # noqa: E402

from .json_llm_utils import parse_json_from_llm  # noqa: E402


# 提示词模板：要求多样性问法 + 固定 JSON schema（键名为英文便于解析）
_QUESTION_GEN_PROMPT = """你是一个专业的问答系统专家。给定的知识内容能回答哪些多样化的问题，这些问题可以：
1）问题类型多样化：直接问、间接问、对比问、条件问、假设问、推理问等
2）表达方式多样化：使用不同的句式、词汇、语气
3）难度层次多样化：简单、中等、困难的问题都要有
4）角度多样化：从不同角度和维度提问
5）确保问题不超出知识内容范围

给定知识内容：
---
{knowledge}
---

请返回 JSON 格式（键名必须为英文）：
{{
    "questions": [{{
        "question": "问题内容",
        "question_type": "问题类型（直接问/间接问/对比问/条件问等）",
        "difficulty": "难度等级（简单/中等/困难）",
        "is_answerable": true,
        "answer": "基于该知识的回答"
    }}]
}}

要求：至少生成 6 条、至多 {max_q} 条；is_answerable 仅在知识足以支撑时为 true；answer 简明准确。"""


def _client() -> OpenAI:
    """创建指向 DashScope 兼容模式的 OpenAI SDK 客户端（用于对话补全）。"""
    key = require_dashscope_key()
    return OpenAI(
        api_key=key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def generate_questions_for_text(knowledge: str, max_q: int | None = None) -> list[dict[str, Any]]:
    """
    对单段知识文本调用 CHAT_MODEL，解析返回的 questions 数组并规范化字段类型。

    Args:
        knowledge: 某条 metadata 的 content（或合并文本）。
        max_q: 最多保留的问题条数；默认 QUESTIONS_PER_CHUNK_MAX。

    Returns:
        字典列表，每项含 question、question_type、difficulty、is_answerable、answer。
    """
    mq = max_q if max_q is not None else QUESTIONS_PER_CHUNK_MAX
    prompt = _QUESTION_GEN_PROMPT.format(knowledge=knowledge.strip(), max_q=mq)
    client = _client()
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "你只输出合法 JSON，不要输出其它解释文字。",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
    )
    raw = completion.choices[0].message.content or ""
    data = parse_json_from_llm(raw)
    questions = data.get("questions")
    if not isinstance(questions, list):
        raise ValueError("模型返回缺少 questions 数组")
    out = []
    for item in questions[:mq]:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "question": str(item.get("question", "")).strip(),
                "question_type": str(item.get("question_type", "")).strip(),
                "difficulty": str(item.get("difficulty", "")).strip(),
                "is_answerable": bool(item.get("is_answerable", True)),
                "answer": str(item.get("answer", "")).strip(),
            }
        )
    return out


def load_metadata_list() -> list[dict[str, Any]]:
    """读取 METADATA_FILE，供批量生成问题使用。"""
    from config import METADATA_FILE  # noqa: E402

    if not METADATA_FILE.is_file():
        raise FileNotFoundError(f"请先构建索引生成元数据: {METADATA_FILE}")
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_question_bank(
    limit: int | None = None,
    only_types: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """
    遍历元数据条目，对每条调用 generate_questions_for_text，写入 QUESTION_BANK_FILE。

    Args:
        limit: 若指定，只处理元数据列表前 N 条（用于快速试跑）。
        only_types: 仅处理指定 type，默认 ("text","image") 即全部。

    Returns:
        写入文件的 payload 字典（含 generated_at、count_items、items）。
    """
    types = only_types or ("text", "image")
    rows = load_metadata_list()
    if limit is not None:
        rows = rows[: int(limit)]

    items_out: list[dict[str, Any]] = []
    for row in rows:
        t = row.get("type")
        if t not in types:
            continue
        kid = row.get("id")
        content = row.get("content") or ""
        if not str(content).strip():
            continue
        qs = generate_questions_for_text(content)
        items_out.append(
            {
                "chunk_id": kid,
                "source": row.get("source"),
                "type": t,
                "questions": qs,
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count_items": len(items_out),
        "items": items_out,
    }
    os.makedirs(QUESTION_BANK_FILE.parent, exist_ok=True)
    with open(QUESTION_BANK_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def load_question_bank() -> dict[str, Any] | None:
    """读取问题库 JSON；文件不存在则返回 None（build_index 将不做检索增强）。"""
    if not QUESTION_BANK_FILE.is_file():
        return None
    with open(QUESTION_BANK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def question_bank_id_map(bank: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """
    将 items 列表转为 chunk_id → 条目的映射，便于 O(1) 查找。

    Args:
        bank: 问题库完整 dict（含 items 数组）。

    Returns:
        键为 int(chunk_id) 的字典。
    """
    m: dict[int, dict[str, Any]] = {}
    for it in bank.get("items") or []:
        cid = it.get("chunk_id")
        if cid is None:
            continue
        m[int(cid)] = it
    return m


def augment_content_for_index(chunk_id: int, base_content: str, bank: dict[str, Any] | None) -> str:
    """
    构建「用于向量嵌入」的文本：在原文后追加衍生问题行，提升问句风格查询的召回。

    注意：metadata 中保存的 content 仍应为 base_content（未追加），由 build_index 区分。

    Args:
        chunk_id: 与 metadata.id 一致。
        base_content: 原始待嵌入文本（文本块或图片拼接串）。
        bank: 问题库 dict；None 则原样返回 base_content。
    """
    if not bank:
        return base_content
    entry = question_bank_id_map(bank).get(int(chunk_id))
    if not entry:
        return base_content
    qs = entry.get("questions") or []
    lines = [q.get("question", "").strip() for q in qs if q.get("question")]
    if not lines:
        return base_content
    joined = "\n".join(lines)
    return f"{base_content}\n\n【语义检索增强·衍生问题】\n{joined}"


def export_regression_questions_from_bank(bank: dict[str, Any], out_path: Path) -> int:
    """
    汇总所有生成问题为去重列表，写入 regression_questions.json，供 kb_versions 回归使用。

    Args:
        bank: 问题库 dict。
        out_path: 一般为 config.REGRESSION_SUITE_FILE。

    Returns:
        导出的问句条数。
    """
    qs_set: list[str] = []
    seen = set()
    for it in bank.get("items") or []:
        for q in it.get("questions") or []:
            text = (q.get("question") or "").strip()
            if text and text not in seen:
                seen.add(text)
                qs_set.append(text)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "questions": qs_set,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return len(qs_set)
