# -*- coding: utf-8 -*-
"""
环境变量加载与 API Key 校验。

设计目的：
- 允许从「本包目录」「项目根目录」多级加载 `.env`，这样无论从哪一层
  目录启动 `python app.py` 或 `python build_index.py`，都能读到 `DASHSCOPE_API_KEY`。
- `load_dotenv(..., override=False)`：已存在于操作系统环境中的变量不会被 .env 覆盖，
  便于在 CI/脚本中临时 export 覆盖本地文件。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from config import PACKAGE_DIR, CASE_DIR, PROJECT_ROOT


def load_env():
    """
    按优先级顺序尝试加载环境变量文件（找到即加载，可多文件依次合并）。
    """
    candidates = [
        PACKAGE_DIR / ".env",
        CASE_DIR / ".env",
        PROJECT_ROOT / ".env",
    ]
    for p in candidates:
        if p.is_file():
            load_dotenv(p, override=False)


def require_dashscope_key():
    """
    确保已配置阿里云 DashScope API Key。

    Returns:
        str: 非空的 DASHSCOPE_API_KEY。

    Raises:
        RuntimeError: 未设置环境变量时抛出，提示用户配置方式。
    """
    load_env()
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError(
            "未设置环境变量 DASHSCOPE_API_KEY。请在 .env 或系统中配置阿里云 DashScope API Key。"
        )
    return key
