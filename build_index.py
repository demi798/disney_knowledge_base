# -*- coding: utf-8 -*-
"""
构建 FAISS 向量索引（离线索引流程）。

整体流程：
1. 文本：扫描知识库目录下 .docx → 解析正文与表格 → 按 CHUNK_SIZE/CHUNK_OVERLAP 滑动切片
   → 调用 DashScope 多模态嵌入 API 的文本分支得到向量 → 与元数据一并暂存。
2. 图片：对 images/ 下图片用 VLM 生成中文描述 → 将「文件名 + 描述」拼成一段文本
   → 同样走文本嵌入 → 元数据中 type=image 并记录 rel_image 供前端展示。
3. 可选：若已存在「问题库」且 config.USE_QUESTION_BANK_FOR_INDEX 为 True，向量化时使用
   「原文 + 衍生问题」作为嵌入输入，但 metadata 中 content 仍保存原文/原描述文本。

输出：
- data/disney_index.faiss：FAISS IndexFlatL2，与向量一一对应 metadata 下标。
- data/disney_metadata.json：与 FAISS 行号一致的 JSON 数组，每条含 id、source、type、content 等。

注意：本脚本会大量调用嵌入与 VLM，需网络与有效 API 额度。
"""

import base64
import json
import os
import sys

import dashscope
import faiss
import numpy as np
from docx import Document as DocxDocument
from http import HTTPStatus
from openai import OpenAI

# 将本脚本所在目录加入 sys.path，支持「在本目录下直接 python build_index.py」运行
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    IMAGES_DIR,
    INDEX_FILE,
    KNOWLEDGE_BASE_DIR,
    METADATA_FILE,
    MULTIMODAL_EMBEDDING_MODEL,
    USE_QUESTION_BANK_FOR_INDEX,
    VL_MODEL,
)
from env_loader import require_dashscope_key  # noqa: E402


def parse_docx(file_path):
    """
    解析 Word 文档，提取段落与表格为纯文本/简易 Markdown 表。

    通过解析 XML 子元素顺序保留「表格在段落之间」的相对位置；表格转为管道符表格字符串，
    便于后续切片与阅读。逻辑与示例保持一致。

    Args:
        file_path: .docx 文件绝对或相对路径。

    Returns:
        拼合后的整篇文本字符串。
    """
    doc = DocxDocument(file_path)
    all_text = []

    for element in doc.element.body:
        # 段落：收集 w:t 文本 run，避免丢字
        if element.tag.endswith("p"):
            paragraph_text = ""
            for run in element.findall(
                ".//w:t",
                {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"},
            ):
                paragraph_text += run.text if run.text else ""
            if paragraph_text.strip():
                all_text.append(paragraph_text.strip())

        # 表格：取与当前 body 元素绑定的 table 对象，转 Markdown 风格行
        elif element.tag.endswith("tbl"):
            table = [t for t in doc.tables if t._element is element][0]
            if table.rows:
                md_table = []
                header = [cell.text.strip() for cell in table.rows[0].cells]
                md_table.append("| " + " | ".join(header) + " |")
                md_table.append("|" + "---|" * len(header))
                for row in table.rows[1:]:
                    row_data = [cell.text.strip() for cell in row.cells]
                    md_table.append("| " + " | ".join(row_data) + " |")
                all_text.append("\n".join(md_table))

    return "\n".join(all_text)


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    固定窗口滑动切片：窗口长度 chunk_size，步长为 chunk_size - overlap。

    重叠 overlap 可减少上下文在切片边界处被切断导致的语义丢失。

    Args:
        text: 全文。
        chunk_size: 单块最大字符数。
        overlap: 相邻两块共享后缀长度。

    Returns:
        非空字符串块列表。
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def get_text_embedding(text: str):
    """
    调用 DashScope 多模态嵌入接口，仅使用 input 中的 text 字段。

    Returns:
        list[float]: 嵌入向量（维度由模型决定）。

    Raises:
        RuntimeError: HTTP 非成功状态时抛出。
    """
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{"text": text}],
    )
    if resp.status_code != HTTPStatus.OK:
        raise RuntimeError(f"文本 Embedding 失败: {getattr(resp, 'message', resp)}")
    return resp.output["embeddings"][0]["embedding"]


def describe_image_with_vlm(client: OpenAI, image_path: str) -> str:
    """
    使用兼容 OpenAI 协议的 DashScope 视觉模型，根据本地图片生成中文描述。

    图片以 data URL（base64）形式传入，便于走 chat.completions 多模态消息格式。
    描述用于后续「文本向量」入库，实现「先视觉理解、再语义检索」的多模态 RAG。

    Args:
        client: base_url 指向 DashScope 兼容模式的 OpenAI 客户端。
        image_path: 本地图片路径。

    Returns:
        模型输出的描述正文（strip 后）。
    """
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    mime = f"image/{ext}"
    data_url = f"data:{mime};base64,{b64}"

    prompt = (
        "请用中文详细描述这张图片：画面主体、色彩风格、是否含文字或标语、主题（如节日活动、迪士尼角色等）。"
        "描述将用于语义检索，请包含可能与用户提问相关的关键词。"
    )
    completion = client.chat.completions.create(
        model=VL_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return (completion.choices[0].message.content or "").strip()


def build_and_save():
    """
    扫描 KNOWLEDGE_BASE_DIR 与 IMAGES_DIR，构建 FAISS 索引并写入磁盘。

    全局自增 doc_id 与 FAISS 行号、metadata 数组下标一致；问题库检索增强按 chunk_id 对齐。
    """
    api_key = require_dashscope_key()
    dashscope.api_key = api_key
    # VLM 走 OpenAI 兼容接口；嵌入仍用 dashscope SDK 的 MultiModalEmbedding
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    if not KNOWLEDGE_BASE_DIR.is_dir():
        raise FileNotFoundError(f"知识库目录不存在: {KNOWLEDGE_BASE_DIR}")

    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)

    # metadata_store[i] 对应 FAISS 第 i 行向量；id 字段与 doc_id 同步便于运营模块关联
    metadata_store = []
    all_vectors = []
    doc_id = 0

    # 若已生成问题库且配置开启，则嵌入「原文 + 衍生问题」；metadata.content 仍保存未拼接前的正文
    question_bank = None
    if USE_QUESTION_BANK_FOR_INDEX:
        from services.kb_questions import load_question_bank  # noqa: E402

        question_bank = load_question_bank()
        if question_bank and (question_bank.get("items") or []):
            print("--- 已加载问题库，将向量化文本附带衍生问题（检索增强）---")
        else:
            question_bank = None

    print("--- 处理 Word 文档 ---")
    for filename in sorted(os.listdir(KNOWLEDGE_BASE_DIR)):
        if filename.startswith(".") or os.path.isdir(
            os.path.join(KNOWLEDGE_BASE_DIR, filename)
        ):
            continue
        file_path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        if not filename.endswith(".docx"):
            continue

        print(f"  文档: {filename}")
        full_text = parse_docx(file_path)
        chunks = split_text(full_text)
        print(f"    字符数 {len(full_text)}，切片数 {len(chunks)}")

        for chunk in chunks:
            embed_text = chunk
            if question_bank:
                from services.kb_questions import augment_content_for_index  # noqa: E402

                embed_text = augment_content_for_index(doc_id, chunk, question_bank)
            vec = get_text_embedding(embed_text)
            all_vectors.append(vec)
            metadata_store.append(
                {
                    "id": doc_id,
                    "source": filename,
                    "type": "text",
                    "content": chunk,
                    "rel_image": None,
                }
            )
            doc_id += 1

    print("--- 处理图片（先描述再 Embedding）---")
    if IMAGES_DIR.is_dir():
        for img_name in sorted(os.listdir(IMAGES_DIR)):
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                continue
            img_path = os.path.join(IMAGES_DIR, img_name)
            print(f"  图片: {img_name}")

            caption = describe_image_with_vlm(client, img_path)
            # 标题行带上文件名，便于用户检索「某张图文件名」时命中
            indexed_text = f"【图片文件】{img_name}\n{caption}"
            embed_text = indexed_text
            if question_bank:
                from services.kb_questions import augment_content_for_index  # noqa: E402

                embed_text = augment_content_for_index(doc_id, indexed_text, question_bank)

            vec = get_text_embedding(embed_text)
            all_vectors.append(vec)
            rel_path = os.path.join("images", img_name).replace("\\", "/")
            metadata_store.append(
                {
                    "id": doc_id,
                    "source": f"图片: {img_name}",
                    "type": "image",
                    "content": indexed_text,
                    "caption": caption,
                    "rel_image": rel_path,
                }
            )
            doc_id += 1

    if not all_vectors:
        raise RuntimeError("未生成任何向量，请检查知识库目录是否包含 docx 或图片。")

    dim = len(all_vectors[0])
    arr = np.array(all_vectors).astype("float32")
    # IndexFlatL2：精确 L2 距离；向量未归一化时距离与「相似度」仅作相对参考
    index = faiss.IndexFlatL2(dim)
    index.add(arr)

    faiss.write_index(index, str(INDEX_FILE))
    with open(str(METADATA_FILE), "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, ensure_ascii=False, indent=2)

    text_n = sum(1 for m in metadata_store if m["type"] == "text")
    img_n = sum(1 for m in metadata_store if m["type"] == "image")
    print(f"\n完成。文本块: {text_n}，图片条目: {img_n}，向量维度: {dim}")
    print(f"索引文件: {INDEX_FILE}")
    print(f"元数据: {METADATA_FILE}")


if __name__ == "__main__":
    build_and_save()
