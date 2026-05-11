# 迪士尼知识库问答系统

基于阿里云 **DashScope**（通义嵌入、通义千问与视觉模型）与 **FAISS** 向量库的迪士尼主题 **RAG（检索增强生成）** 问答 Web 应用。用户在前端提问后，系统从本地知识库检索相关内容，再由大模型整合并生成中文回答。

---

## 功能概览

| 环节 | 说明 |
|------|------|
| **文本知识** | 读取 `disney_knowledge_base/` 下 `.docx`，按固定长度滑动切片后调用 `tongyi-embedding-vision-plus` 生成向量，写入 FAISS。 |
| **图片知识** | 对 `images/` 中图片先用 **`qwen-vl-plus`** 生成中文描述，再将「文件名 + 描述」作为一整段文本做嵌入并入库（满足「先描述、再向量检索」）。 |
| **问答** | 用户问题向量化 → FAISS Top-K 检索 → 将检索片段作为上下文，由 **`qwen-flash`** 生成回答；前端展示答案与参考资料，命中图片类条目时可预览图片。 |
| **知识库运营** | 独立页面 **「知识库运营」**（`/kb_ops`）：基于元数据自动生成多样化问题、从对话抽取知识点、健康度巡检、索引版本快照与检索回归对比等（均依赖 DashScope，会产生费用）。 |

---

## 目录结构

```
disney_knowledge_base/
├── README.md              # 本说明
├── requirements.txt       # Python 依赖
├── config.py              # 路径、模型名、切块与 Top-K 等参数
├── env_loader.py          # 多级目录加载 .env
├── build_index.py         # 构建向量索引（需联网调用 API）
├── rag_service.py         # 加载索引、检索、LLM 作答（含按路径检索，供回归）
├── app.py                 # Flask Web 服务
├── services/              # 知识运营后端逻辑
│   ├── json_llm_utils.py      # 从大模型返回中解析 JSON
│   ├── kb_questions.py        # 多样化问题生成与问题库
│   ├── dialogue_knowledge.py  # 对话知识点抽取与合并
│   ├── kb_health.py           # 知识库健康度检查
│   └── kb_versions.py         # 版本快照与检索回归对比
├── templates/
│   ├── index.html         # 问答主页（仅问答）
│   └── kb_ops.html        # 知识库运营页
├── static/style.css       # 页面样式
└── data/                  # 索引与运营数据（构建或运营后生成）
    ├── disney_index.faiss
    ├── disney_metadata.json
    ├── kb_generated_questions.json  # 自动生成的问题库（可选）
    ├── dialogue_knowledge.json      # 从对话抽取并合并的知识点（可选）
    ├── kb_health_report.json        # 最近一次健康度报告（可选）
    ├── regression_questions.json    # 回归测试用问句集（可选，可由问题库导出）
    └── versions/                    # 索引快照：versions/<标签>/
        ├── manifest.json
        ├── disney_index.faiss
        └── disney_metadata.json
```

知识库原文档与图片位于 **上一级目录** 的 `disney_knowledge_base/`（与 `config.py` 中 `KNOWLEDGE_BASE_DIR` 一致），请勿随意改名路径，或在 `config.py` 中自行修改。

---

## 环境要求

- Python 3.10+（推荐 3.11 / 3.13）
- 已开通的阿里云 **DashScope** API Key，且账号具备所用模型的调用权限（嵌入、对话、多模态视觉等以控制台为准）。

---

## 安装依赖

在虚拟环境中执行（请将路径换成你本机的项目根目录）：

```bash
cd /path/to/disney_knowledge_base
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 配置环境变量

设置 **`DASHSCOPE_API_KEY`**。可选方式：

1. 在 **`.env`** 中写入（支持 `export KEY=...` 形式），程序会通过 `env_loader.py` 自动加载。
2. 或在 shell 中：`export DASHSCOPE_API_KEY="你的Key"`。

请勿将真实 Key 提交到公开仓库。

---

## 构建向量索引

首次使用或知识库文件变更后需要执行（会消耗嵌入与视觉模型的调用额度）：

```bash
cd disney_knowledge_base
python build_index.py
```

成功后会在 `data/` 下生成 `disney_index.faiss` 与 `disney_metadata.json`。

---

## 启动 Web 服务

```bash
cd disney_knowledge_base
python app.py
```

默认在浏览器打开：**http://127.0.0.1:5050**（以终端输出为准）。

- **问答首页**：`http://127.0.0.1:5050/`  
- **知识库运营页**：`http://127.0.0.1:5050/kb_ops`（与问答分离，避免误触消耗 API）

---

## 知识库运营（`/kb_ops`）

运营能力在浏览器中操作即可；底层写入 `data/` 下 JSON 或 `data/versions/` 快照。以下为能力摘要与推荐流程。

| 能力 | 说明 |
|------|------|
| **多样化问题生成** | 根据当前 `disney_metadata.json` 中每条知识调用对话模型生成多类型、多难度问法，结果写入 `kb_generated_questions.json`；可同时导出 `regression_questions.json` 供回归。 |
| **检索增强** | `config.py` 中 **`USE_QUESTION_BANK_FOR_INDEX`** 为 `True` 且已存在问题库时，`build_index.py` 在向量化时会把「衍生问题」拼入嵌入文本，**展示用的 `content` 仍为原文**；知识变更或切块变化后请 **重新生成问题库并重建索引**，否则 `chunk_id` 可能与条目错位。 |
| **对话知识点抽取** | 粘贴用户与助手对话文本，抽取事实、偏好、FAQ、流程、注意事项等，与已有 `dialogue_knowledge.json` **合并去重**后保存。 |
| **健康度检查** | 分批检视元数据，从时效、价格、政策、活动、联系方式、技术等维度输出 issues / conflicts / gaps 及 **summary**（LLM 研判，仅供参考）。 |
| **版本与回归** | **快照**：将当前 `disney_index.faiss` 与 `disney_metadata.json` 复制到 `data/versions/<标签>/`。**回归**：对 `regression_questions.json`（若无则用内置少量问句）分别在两套索引上做向量检索，对比 **Top-1 相似度均值**，不调用生成模型；用于上线前对比「检索匹配」强弱。 |
| **重载索引缓存** | 在终端执行 `build_index.py` 覆盖索引后，若服务未重启，可在运营页调用 **重载索引缓存**（`POST /api/kb/reload_cache`），使问答线程重新从磁盘加载 FAISS。 |

**推荐操作顺序（启用检索增强时）**

1. `python build_index.py`（得到最新 `disney_metadata.json`）  
2. 打开 `/kb_ops`，执行 **生成问题库**  
3. 再次 `python build_index.py`（嵌入阶段会附带衍生问题）  
4. 在 `/kb_ops` 点击 **重载索引缓存**（或重启 `app.py`）

---

## HTTP 接口说明

### 页面

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 问答主页 |
| GET | `/kb_ops` | 知识库运营页 |

### 问答与静态资源

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/ask` | JSON：`{"question": "用户问题", "top_k": 5}`（`top_k` 可选） |
| GET | `/kb_images/...` | 知识库内图片预览（仅限 `images/` 下资源） |

### 知识运营 API（与 `/kb_ops` 按钮一致）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/kb/generate_questions` | 生成问题库；Body 可选 `{"limit": N}` 仅处理前 N 条（调试） |
| GET | `/api/kb/question_bank` | 问题库摘要；Query `full=1` 返回完整 JSON |
| POST | `/api/kb/export_regression` | 从问题库导出 `regression_questions.json` |
| POST | `/api/kb/extract_dialogue` | Body：`{"text":"..."}` 或 `{"messages":[...]}` |
| GET | `/api/kb/dialogue_knowledge` | 读取已保存的对话知识点 |
| POST | `/api/kb/health_check` | 执行健康检查；Body 可选 `{"batch_size": 6}` |
| GET | `/api/kb/health_report` | 读取最近一次健康报告 |
| GET | `/api/kb/versions` | 列出本地版本快照 |
| POST | `/api/kb/version/snapshot` | Body：`{"label":"v1","note":""}` |
| POST | `/api/kb/version/regression` | 当前索引 vs 某版本；Body：`{"version_label":"...","top_k":5}` |
| POST | `/api/kb/version/regression_pair` | 两版本互比；Body：`{"label_a":"...","label_b":"...","top_k":5}` |
| POST | `/api/kb/reload_cache` | 清空内存中的 FAISS 缓存，下次问答重新读盘 |

---

## 可调参数（`config.py`）

- **`CHUNK_SIZE` / `CHUNK_OVERLAP`**：文档切片长度与重叠。
- **`RAG_TOP_K`**：默认检索条数。
- **`MULTIMODAL_EMBEDDING_MODEL`**：嵌入模型名称。
- **`VL_MODEL`**：图片描述模型（若账号不可用，可改为控制台已开通的其他通义视觉模型）。
- **`CHAT_MODEL`**：对话生成模型（问答与多数运营能力共用）。
- **`USE_QUESTION_BANK_FOR_INDEX`**：构建索引时是否合并已生成的问题库以增强检索（需先有 `kb_generated_questions.json`）。
- **`QUESTIONS_PER_CHUNK_MAX`**：单条知识调用模型生成问题的条数上限。
- **`QUESTION_BANK_FILE` / `DIALOGUE_KNOWLEDGE_FILE` / `KB_HEALTH_REPORT_FILE` / `VERSIONS_DIR` / `REGRESSION_SUITE_FILE`**：运营产物路径（一般无需改文件名，必要时改目录请同步此处）。

---

## 注意事项（修改与使用）

1. **费用**：运营页会多次调用对话 / 嵌入等 API，请在阿里云控制台关注额度与账单；「回归对比」仅向量检索，相对省钱。
2. **健康度与冲突检测**：依赖大模型对文本的理解，**不能替代**人工核对官网；适用于巡检与排优先级。
3. **索引与缓存**：修改 `build_index.py` 产出文件后，若长时间运行的 Flask 未重启，务必 **重载索引缓存** 或重启进程，否则问答仍用旧向量。
4. **版本标签**：快照目录名仅允许字母、数字、点、下划线、短横线；同标签已存在时会报错，请换新标签或手动删除 `data/versions/<标签>/`（谨慎操作）。
5. **隐私**：`dialogue_knowledge.json` 可能含用户对话提炼内容，勿提交到公开仓库；`.env` 与 `data/` 下敏感文件建议加入 `.gitignore`。
6. **功能隔离**：问答首页不包含运营入口，避免普通使用者误触；需要运营时直接访问 **`/kb_ops`**。

---

## 常见问题

1. **提示未找到索引文件**  
   请先在本目录运行 `python build_index.py` 完成构建。

2. **图片描述或嵌入报错**  
   检查 API Key、余额与模型权限；必要时在 `config.py` 中更换 `VL_MODEL`。

3. **知识库路径**  
   默认读取 `disney_knowledge_base/`。若你移动了文件夹，只需保证该相对关系不变，或修改 `config.py` 中的 `KNOWLEDGE_BASE_DIR`。

4. **生成问题库后检索未变好**  
   需第二次执行 `python build_index.py` 且 **`USE_QUESTION_BANK_FOR_INDEX=True`**，然后在运营页 **重载索引缓存** 或重启服务。

5. **健康检查很慢或超时**  
   条目多时会多次调用模型；可适当减小 `POST /api/kb/health_check` 的 `batch_size`，或分批更新知识库后再检。

6. **回归结果如何理解**  
   对比的是「同一批问句」在两个索引上的 **向量相似度**，越高表示检索到的片段与问句越近；不直接等价于回答正确率，需结合业务抽样验证。

---

## 许可证与声明

本项目为练习代码示例。迪士尼相关商标与素材版权归各自权利人所有；演示请勿用于商业用途。
