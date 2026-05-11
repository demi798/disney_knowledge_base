# -*- coding: utf-8 -*-
"""
从大模型返回的纯文本中解析 JSON 对象。

背景：
- 模型常包裹 ```json ... ``` 或在 JSON 前后附加说明文字；
- 中文语境下偶发全角逗号、弯引号、尾逗号等导致 json.loads 失败。

策略：
1. 优先匹配 markdown 围栏内内容；
2. 否则截取第一个 `{` 与最后一个 `}` 之间的子串；
3. 首次 loads 失败则做一轮常见字符替换与尾逗号清理后再 loads。
"""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_from_llm(text: str) -> dict[str, Any]:
    """
    将模型输出解析为 Python dict（要求根类型为 JSON object）。

    Args:
        text: 模型返回的完整字符串。

    Returns:
        解析后的字典。

    Raises:
        ValueError: 输入为空。
        json.JSONDecodeError: 经修正后仍无法解析时抛出。
    """
    if not text or not text.strip():
        raise ValueError("模型返回为空")

    s = text.strip()
    # 去掉 ``` 或 ```json 围栏，取中间正文
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()

    # 防止前后有「好的以下是 JSON」等前缀后缀：取最外层花括号片段
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        s = s[start : end + 1]

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # 常见修正：中文逗号、弯引号、对象/数组尾逗号
        s2 = s.replace("，", ",")
        s2 = s2.replace("\u201c", '"').replace("\u201d", '"')
        s2 = re.sub(r",\s*}", "}", s2)
        s2 = re.sub(r",\s*]", "]", s2)
        return json.loads(s2)
