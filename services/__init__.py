# -*- coding: utf-8 -*-
"""
知识库运营子包。

包含：
- json_llm_utils：解析大模型输出的 JSON（容错 markdown 代码块、截断、常见符号修正）
- kb_questions：基于元数据批量生成多样化问题、检索增强拼接、导出回归问句
- dialogue_knowledge：从对话文本抽取结构化知识点并与历史记录合并
- kb_health：分批调用模型做知识库「体检」并合并报告
- kb_versions：索引快照、列出版本、基于向量检索的回归对比

这些模块依赖 config / env_loader 及 DashScope；部分功能另依赖 rag_service（延迟导入避免循环）。
"""
