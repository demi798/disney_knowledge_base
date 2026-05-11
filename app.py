# -*- coding: utf-8 -*-
"""
迪士尼知识库问答 · Flask Web 应用入口。

路由分组：
1. 页面：首页问答 `/`、知识库运营 `/kb_ops`（运营能力与问答分离，降低误触 API 成本）。
2. 核心 API：`POST /api/ask` 调用 rag_service.answer_question。
3. 静态资源：`/kb_images/<path>` 映射本地知识库 images 目录（带路径穿越防护）。
4. 知识运营 API：`/api/kb/*`，委托 services 包内模块，详见 README。

返回约定：业务接口统一 JSON，尽量包含 ok 字段；错误时 HTTP 4xx/5xx 与 error 说明。
"""

import json
import os
import sys

from flask import Flask, jsonify, render_template, request, send_from_directory

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import (  # noqa: E402
    KNOWLEDGE_BASE_DIR,
    QUESTION_BANK_FILE,
    REGRESSION_SUITE_FILE,
)
from rag_service import answer_question, clear_index_cache  # noqa: E402

app = Flask(
    __name__,
    template_folder=os.path.join(_ROOT, "templates"),
    static_folder=os.path.join(_ROOT, "static"),
)


# ---------------------------------------------------------------------------
# 页面路由
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """问答主页模板（仅提问与展示结果，不含运营入口）。"""
    return render_template("index.html")


@app.route("/kb_ops")
def kb_ops():
    """知识库运营独立页面：问题生成、对话抽取、质检、版本快照与回归。"""
    return render_template("kb_ops.html")


# ---------------------------------------------------------------------------
# 核心问答 API
# ---------------------------------------------------------------------------


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """
    JSON Body：question（必填）；top_k（可选）覆盖默认检索条数。

    Returns:
        200：{"ok": true, "answer": str, "sources": list}
        400：问题为空
        500：底层嵌入/检索/对话异常信息
    """
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"ok": False, "error": "问题不能为空"}), 400
    try:
        top_k = data.get("top_k")
        if top_k is not None:
            top_k = int(top_k)
        result = answer_question(question, top_k=top_k)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/kb_images/<path:rel>")
def serve_kb_image(rel):
    """
    按相对路径提供知识库下的图片文件，供前端 `<img src="/kb_images/images/xxx">` 使用。

    安全：禁止 `..` 路径穿越；统一归一化为 KNOWLEDGE_BASE_DIR 下的子路径。
    """
    rel = rel.replace("\\", "/")
    if rel.startswith("..") or "/.." in rel:
        return "禁止访问", 403
    if not rel.startswith("images/"):
        rel = "images/" + rel.lstrip("/")
    folder = os.path.dirname(rel)
    fname = os.path.basename(rel)
    base = os.path.join(str(KNOWLEDGE_BASE_DIR), folder)
    return send_from_directory(base, fname)


# ---------------------------------------------------------------------------
# 知识运营 API（与 kb_ops.html 按钮及 README 表格一致）
# ---------------------------------------------------------------------------


@app.route("/api/kb/generate_questions", methods=["POST"])
def api_kb_generate_questions():
    """
    遍历当前 metadata 生成问题库文件，并导出回归问句集。

    Body 可选：{"limit": N} 仅处理前 N 条（调试减负）。
    """
    data = request.get_json(silent=True) or {}
    limit = data.get("limit")
    if limit is not None:
        limit = int(limit)
    try:
        from services.kb_questions import build_question_bank, export_regression_questions_from_bank

        payload = build_question_bank(limit=limit)
        n_export = 0
        if QUESTION_BANK_FILE.is_file():
            with open(QUESTION_BANK_FILE, "r", encoding="utf-8") as f:
                bank = json.load(f)
            n_export = export_regression_questions_from_bank(bank, REGRESSION_SUITE_FILE)
        return jsonify(
            {
                "ok": True,
                "message": "问题库生成完成；count_items 为知识条目数，regression_exported 为衍生问句去重后的数量。",
                "path": str(QUESTION_BANK_FILE),
                "count_items": payload.get("count_items"),
                "regression_exported": n_export,
                "hint": "若需检索增强，请在本目录执行 python3 build_index.py（或 python）后，在本页点击「重载索引缓存」。",
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/kb/question_bank", methods=["GET"])
def api_kb_question_bank():
    """Query：full=1 返回完整问题库 JSON；否则返回摘要与少量示例问句。"""
    try:
        from services.kb_questions import load_question_bank

        full = request.args.get("full") == "1"
        bank = load_question_bank()
        if not bank:
            return jsonify({"ok": True, "exists": False})
        if full:
            return jsonify({"ok": True, "exists": True, "bank": bank})
        items = bank.get("items") or []
        return jsonify(
            {
                "ok": True,
                "exists": True,
                "generated_at": bank.get("generated_at"),
                "count_items": len(items),
                "sample_questions": [
                    (it.get("questions") or [{}])[0].get("question", "")
                    for it in items[:5]
                ],
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/kb/export_regression", methods=["POST"])
def api_kb_export_regression():
    """将问题库中所有 question 字段去重写入 regression_questions.json。"""
    try:
        from services.kb_questions import export_regression_questions_from_bank, load_question_bank

        bank = load_question_bank()
        if not bank:
            return jsonify({"ok": False, "error": "尚未生成问题库"}), 400
        n = export_regression_questions_from_bank(bank, REGRESSION_SUITE_FILE)
        return jsonify({"ok": True, "count": n, "path": str(REGRESSION_SUITE_FILE)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/kb/extract_dialogue", methods=["POST"])
def api_kb_extract_dialogue():
    """Body：{"text":"..."} 或 {"messages":[{"role","content"},...]}，抽取后与本地 JSON 合并。"""
    data = request.get_json(silent=True) or {}
    try:
        from services.dialogue_knowledge import ingest_dialogue

        payload = ingest_dialogue(
            dialogue_text=data.get("text"),
            messages=data.get("messages"),
        )
        return jsonify({"ok": True, **payload})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/kb/dialogue_knowledge", methods=["GET"])
def api_kb_dialogue_knowledge():
    """读取 dialogue_knowledge.json 包装为 JSON 返回。"""
    try:
        from services.dialogue_knowledge import load_stored_knowledge

        data = load_stored_knowledge()
        return jsonify({"ok": True, "data": data})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/kb/health_check", methods=["POST"])
def api_kb_health_check():
    """Body 可选 batch_size；跑完整质检并写入 kb_health_report.json。"""
    data = request.get_json(silent=True) or {}
    batch_size = int(data.get("batch_size") or 6)
    try:
        from services.kb_health import run_health_check

        payload = run_health_check(batch_size=batch_size)
        return jsonify({"ok": True, **payload})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/kb/health_report", methods=["GET"])
def api_kb_health_report():
    """返回最近一次落盘的健康报告（若无文件则 data 为 null）。"""
    try:
        from services.kb_health import load_last_report

        r = load_last_report()
        return jsonify({"ok": True, "report": r})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/kb/versions", methods=["GET"])
def api_kb_versions():
    """列出 data/versions 下各快照的 manifest（若仅有目录无 manifest 则返回占位信息）。"""
    try:
        from services.kb_versions import list_versions

        return jsonify({"ok": True, "versions": list_versions()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/kb/version/snapshot", methods=["POST"])
def api_kb_version_snapshot():
    """Body：label（必填）、note（可选）；复制当前索引与元数据到 versions/<label>/。"""
    data = request.get_json(silent=True) or {}
    label = (data.get("label") or "").strip()
    note = (data.get("note") or "").strip()
    try:
        from services.kb_versions import snapshot_version

        man = snapshot_version(label, note=note)
        return jsonify({"ok": True, "manifest": man})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/kb/version/regression", methods=["POST"])
def api_kb_version_regression():
    """对比「当前 data 索引」与「versions/<version_label>」的检索 Top-1 相似度均值。"""
    data = request.get_json(silent=True) or {}
    version_label = (data.get("version_label") or "").strip()
    top_k = data.get("top_k")
    if top_k is not None:
        top_k = int(top_k)
    else:
        top_k = 5
    try:
        from services.kb_versions import regression_current_vs_label

        out = regression_current_vs_label(version_label, top_k=top_k)
        return jsonify({"ok": True, **out})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/kb/version/regression_pair", methods=["POST"])
def api_kb_version_regression_pair():
    """对比两个历史版本目录下的索引（均位于 versions/）。"""
    data = request.get_json(silent=True) or {}
    a = (data.get("label_a") or "").strip()
    b = (data.get("label_b") or "").strip()
    top_k = int(data.get("top_k") or 5)
    try:
        from services.kb_versions import regression_two_versions

        out = regression_two_versions(a, b, top_k=top_k)
        return jsonify({"ok": True, **out})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/kb/reload_cache", methods=["POST"])
def api_kb_reload_cache():
    """调用 rag_service.clear_index_cache，使后续问答重新加载磁盘索引。"""
    try:
        clear_index_cache()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("请在浏览器打开 http://127.0.0.1:5050")
    app.run(host="0.0.0.0", port=5050, debug=True)
