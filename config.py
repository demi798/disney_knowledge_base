# -*- coding: utf-8 -*-
"""
路径与模型配置模块。

说明：
- 本目录为disney_knowledge_base包根目录；原始 Word 与图片知识库位于上级
  `disney_knowledge_base/`，通过 KNOWLEDGE_BASE_DIR 引用。
- 向量索引与运营产生的 JSON 默认落在本包下的 `data/`。
- 修改模型名、切块参数后，通常需要重新运行 `build_index.py`（索引）或仅重启服务（仅改 CHAT_MODEL 等）。
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 目录层级（便于 env_loader 向上查找 .env）
# ---------------------------------------------------------------------------
# 本模块所在目录（disney_knowledge_base项目根）
PACKAGE_DIR = Path(__file__).resolve().parent
CASE_DIR = PACKAGE_DIR.parent
# 再上一级，用于加载该层 .env
PROJECT_ROOT = CASE_DIR.parent

# ---------------------------------------------------------------------------
# 知识库原始文件路径（构建索引时读取）
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE_DIR = CASE_DIR / "disney_knowledge_base"
IMAGES_DIR = KNOWLEDGE_BASE_DIR / "images"

# ---------------------------------------------------------------------------
# 向量索引与元数据输出路径（build_index.py 写入；rag_service.py 读取）
# ---------------------------------------------------------------------------
DATA_DIR = PACKAGE_DIR / "data"
INDEX_FILE = DATA_DIR / "disney_index.faiss"
METADATA_FILE = DATA_DIR / "disney_metadata.json"

# ---------------------------------------------------------------------------
# 阿里云 DashScope 模型名称（控制台需开通对应模型）
# ---------------------------------------------------------------------------
# 多模态嵌入：此处仅用「文本」分支，与图片 caption 拼接后同一模型嵌入
MULTIMODAL_EMBEDDING_MODEL = "tongyi-embedding-vision-plus"
# 视觉语言模型：对图片生成中文描述，再作为文本参与检索管线
VL_MODEL = "qwen-vl-plus"
# 对话模型：RAG 最终生成回答；亦用于知识运营中的问题生成、质检、抽取等
CHAT_MODEL = "qwen-flash"

# ---------------------------------------------------------------------------
# 文本切片（影响每条向量长度与召回粒度）
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ---------------------------------------------------------------------------
# 检索默认 Top-K（可被 API / 调用方覆盖）
# ---------------------------------------------------------------------------
RAG_TOP_K = 5

# ---------------------------------------------------------------------------
# 知识运营产物路径（各 services 模块读写）
# ---------------------------------------------------------------------------
QUESTION_BANK_FILE = DATA_DIR / "kb_generated_questions.json"
DIALOGUE_KNOWLEDGE_FILE = DATA_DIR / "dialogue_knowledge.json"
KB_HEALTH_REPORT_FILE = DATA_DIR / "kb_health_report.json"
VERSIONS_DIR = DATA_DIR / "versions"
REGRESSION_SUITE_FILE = DATA_DIR / "regression_questions.json"

# True：build_index 时若存在 QUESTION_BANK_FILE，将把衍生问题拼入「嵌入文本」以增强检索；
# 展示用 metadata.content 仍为原文，详见 kb_questions.augment_content_for_index
USE_QUESTION_BANK_FOR_INDEX = True

# 单段知识调用 LLM 生成问题的条数上限（控制成本）
QUESTIONS_PER_CHUNK_MAX = 12
