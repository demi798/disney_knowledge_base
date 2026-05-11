# -*- coding: utf-8 -*-
"""
知识库健康度检查：分批将 metadata 条目送入 LLM，按时间/价格/政策等维度输出 issues，
多批结果再合并为带 summary 的总报告。

说明：
- 结论依赖模型对文本的理解，属辅助巡检，不能替代人工核对官方渠道。
- 单批正文长度受 _format_batch 限制，超长 content 会截断并标注。
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

from config import CHAT_MODEL, KB_HEALTH_REPORT_FILE  # noqa: E402
from env_loader import require_dashscope_key  # noqa: E402
from openai import OpenAI  # noqa: E402

from .json_llm_utils import parse_json_from_llm  # noqa: E402


# 单批质检：输出 issues / conflicts / gaps 三块结构化意见
_BATCH_PROMPT = """你是上海迪士尼度假区知识库质检专家。请根据下列「知识条目片段」检查质量。
检查维度（逐条记录发现的问题；若某维度无问题则不在该维度下列出）：
1）时间相关信息是否可能过期（年份、日期、时间范围）
2）价格信息是否可能过时（票价、费用、金额）
3）政策规则是否可能需更新
4）活动信息是否可能已失效
5）联系方式是否可能不准确（电话、地址、网址）
6）技术或版本描述是否可能过时

知识条目（id / 来源 / 类型 / 内容摘要）：
---
{batch_text}
---

请只输出 JSON：
{{
  "issues": [
    {{
      "chunk_id": 0,
      "source": "",
      "category": "时间|价格|政策|活动|联系方式|技术|其它",
      "severity": "低|中|高",
      "description": "问题说明",
      "suggestion": "改进建议"
    }}
  ],
  "conflicts": [
    {{
      "topic": "冲突主题",
      "chunk_ids": [],
      "description": "冲突说明"
    }}
  ],
  "gaps": [
    {{
      "topic": "可能缺失的主题",
      "reason": "为何认为缺失"
    }}
  ]
}}"""


# 多批合并：去重归纳并生成 summary 便于人读
_MERGE_PROMPT = """下面是同一知识库分多批次的质检结果 JSON 数组，请合并为一份最终报告：
- 合并 issues / conflicts / gaps，去除重复项，冲突与缺口描述更清晰处保留
- 增加 summary（字符串）：整体健康度概述与优先处理建议

批次结果：
{batches_json}

请输出最终 JSON：
{{
  "summary": "",
  "issues": [],
  "conflicts": [],
  "gaps": []
}}"""


def _client() -> OpenAI:
    """对话客户端，用于批次质检与最终合并。"""
    key = require_dashscope_key()
    return OpenAI(
        api_key=key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def _format_batch(rows: list[dict[str, Any]], max_chars: int = 12000) -> str:
    """
    将多条 metadata 格式化为带 id/source/type 的纯文本块，控制总字符避免超长 prompt。

    单条 content 超过 4000 字符时截断，防止单文档撑爆上下文。
    """
    parts = []
    n = 0
    for row in rows:
        cid = row.get("id")
        src = row.get("source")
        typ = row.get("type")
        body = row.get("content") or ""
        if len(body) > 4000:
            body = body[:4000] + "\n…(截断)"
        line = f"id={cid} | {src} | {typ}\n{body}\n---\n"
        if n + len(line) > max_chars:
            break
        parts.append(line)
        n += len(line)
    return "\n".join(parts)


def run_health_check(batch_size: int = 6) -> dict[str, Any]:
    """
    读取 METADATA_FILE，按 batch_size 切片逐批调用模型，再合并或补全 summary。

    Args:
        batch_size: 每批包含的 metadata 条数上限（实际字符还受 _format_batch 限制）。

    Returns:
        含 checked_at、metadata_count、report（合并后的 dict）的 payload，并写入 KB_HEALTH_REPORT_FILE。
    """
    from config import METADATA_FILE  # noqa: E402

    if not METADATA_FILE.is_file():
        raise FileNotFoundError(f"缺少元数据: {METADATA_FILE}")
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    client = _client()
    partial_reports: list[dict[str, Any]] = []

    for i in range(0, len(meta), batch_size):
        batch = meta[i : i + batch_size]
        batch_text = _format_batch(batch)
        prompt = _BATCH_PROMPT.format(batch_text=batch_text)
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "只输出合法 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        raw = completion.choices[0].message.content or ""
        partial_reports.append(parse_json_from_llm(raw))

    if len(partial_reports) == 1:
        final = partial_reports[0]
        # 单批时模型可能未给 summary 字段，补一句占位
        if "summary" not in final:
            final["summary"] = "已完成单批次质检。"
    else:
        merge_prompt = _MERGE_PROMPT.format(
            batches_json=json.dumps(partial_reports, ensure_ascii=False)
        )
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "只输出合法 JSON。"},
                {"role": "user", "content": merge_prompt},
            ],
            temperature=0.15,
        )
        raw = completion.choices[0].message.content or ""
        final = parse_json_from_llm(raw)

    payload = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "metadata_count": len(meta),
        "report": final,
    }
    os.makedirs(KB_HEALTH_REPORT_FILE.parent, exist_ok=True)
    with open(KB_HEALTH_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def load_last_report() -> dict[str, Any] | None:
    """读取最近一次写入的 kb_health_report.json；不存在则 None。"""
    if not KB_HEALTH_REPORT_FILE.is_file():
        return None
    with open(KB_HEALTH_REPORT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
