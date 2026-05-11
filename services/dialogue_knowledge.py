# -*- coding: utf-8 -*-
"""
从用户与助手的对话中抽取结构化知识点，并与历史记录合并后写入 JSON。

数据流：
1. extract_from_dialogue：单轮 LLM 抽取 → facts / user_preferences / faq / procedures / cautions。
2. merge_into_store：若磁盘已有 knowledge，则再走一轮 LLM 合并新旧 JSON；否则直接规范化落盘。
3. ingest_dialogue：入口，支持纯文本或 OpenAI 风格的 messages 列表。

落盘路径：config.DIALOGUE_KNOWLEDGE_FILE；顶层含 updated_at 与 knowledge 对象。
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

from config import CHAT_MODEL, DIALOGUE_KNOWLEDGE_FILE  # noqa: E402
from env_loader import require_dashscope_key  # noqa: E402
from openai import OpenAI  # noqa: E402

from .json_llm_utils import parse_json_from_llm  # noqa: E402


# 第一轮：从自然语言对话抽取为固定 schema 的 JSON
_EXTRACT_PROMPT = """你是知识整理专家。请从下列用户与助手之间的对话中，提取有价值的知识点，并归入下列类别（某类无内容则给空数组）：
- facts：事实性信息（地点、时间、价格、规则等），每项含 summary（一句话）、detail（可选补充）
- user_preferences：用户需求与偏好
- faq：常见问答，每项含 question、answer
- procedures：操作流程与步骤，每项含 title、steps（字符串数组）
- cautions：注意事项与提醒（字符串数组）

对话内容：
---
{dialogue}
---

请只输出 JSON：
{{
  "facts": [{{"summary":"", "detail":""}}],
  "user_preferences": [""],
  "faq": [{{"question":"","answer":""}}],
  "procedures": [{{"title":"","steps":[]}}],
  "cautions": [""]
}}"""


# 第二轮：与已有 JSON 合并去重，冲突写入 notes 简述
_MERGE_PROMPT = """你是知识库管理员。下方「已有结构化知识」与「本轮新抽取」需合并去重、归类一致。
原则：语义重复则合并为一条更完整的表述；冲突时标注在 notes 中简要说明。

已有 JSON：
{existing}

本轮新抽取 JSON：
{new_items}

请输出合并后的完整 JSON，schema 与已有相同，并可额外包含顶层字段 notes（字符串，简述合并处理）：
{{
  "facts": [],
  "user_preferences": [],
  "faq": [],
  "procedures": [],
  "cautions": [],
  "notes": ""
}}"""


def _client() -> OpenAI:
    """DashScope 兼容模式的对话客户端。"""
    key = require_dashscope_key()
    return OpenAI(
        api_key=key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def _normalize_schema(data: dict[str, Any]) -> dict[str, Any]:
    """
    保证各字段为列表类型，notes 为字符串，避免 None 导致前端或后续合并异常。
    """
    return {
        "facts": list(data.get("facts") or []),
        "user_preferences": list(data.get("user_preferences") or []),
        "faq": list(data.get("faq") or []),
        "procedures": list(data.get("procedures") or []),
        "cautions": list(data.get("cautions") or []),
        "notes": str(data.get("notes") or ""),
    }


def extract_from_dialogue(dialogue_text: str) -> dict[str, Any]:
    """
    仅执行「抽取」步骤，不与磁盘已有内容合并。

    Args:
        dialogue_text: 完整对话字符串。

    Returns:
        规范化后的结构化 dict。
    """
    prompt = _EXTRACT_PROMPT.format(dialogue=dialogue_text.strip())
    client = _client()
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "只输出合法 JSON。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    raw = completion.choices[0].message.content or ""
    data = parse_json_from_llm(raw)
    return _normalize_schema(data)


def load_stored_knowledge() -> dict[str, Any] | None:
    """读取完整存储文件（含 updated_at 与 knowledge）；不存在返回 None。"""
    if not DIALOGUE_KNOWLEDGE_FILE.is_file():
        return None
    with open(DIALOGUE_KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_into_store(new_piece: dict[str, Any]) -> dict[str, Any]:
    """
    将本轮抽取结果合并入本地文件：无历史则直接写；有历史则 LLM 合并后再写。

    Args:
        new_piece: extract_from_dialogue 的输出。

    Returns:
        写入磁盘的完整 payload（updated_at + knowledge）。
    """
    prev = load_stored_knowledge()
    existing_obj = (prev or {}).get("knowledge") if prev else None
    if not existing_obj:
        merged = new_piece
        merged = _normalize_schema(merged)
    else:
        prompt = _MERGE_PROMPT.format(
            existing=json.dumps(existing_obj, ensure_ascii=False),
            new_items=json.dumps(new_piece, ensure_ascii=False),
        )
        client = _client()
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "只输出合法 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        raw = completion.choices[0].message.content or ""
        data = parse_json_from_llm(raw)
        merged = _normalize_schema(data)

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "knowledge": merged,
    }
    os.makedirs(DIALOGUE_KNOWLEDGE_FILE.parent, exist_ok=True)
    with open(DIALOGUE_KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def ingest_dialogue(
    dialogue_text: str | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    对外统一入口：可选 messages 转写在先，再抽取并合并落盘。

    Args:
        dialogue_text: 直接传入整段对话文本。
        messages: 形如 [{"role":"user","content":"..."}] 的列表，将拼接为多行 role: content。

    Returns:
        merge_into_store 的返回值。

    Raises:
        ValueError: 未提供有效文本时。
    """
    if messages:
        lines = []
        for m in messages:
            role = m.get("role") or "user"
            content = (m.get("content") or "").strip()
            if content:
                lines.append(f"{role}: {content}")
        dialogue_text = "\n".join(lines)
    if not dialogue_text or not dialogue_text.strip():
        raise ValueError("对话内容为空")

    extracted = extract_from_dialogue(dialogue_text)
    return merge_into_store(extracted)
