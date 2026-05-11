# -*- coding: utf-8 -*-
"""
知识库索引版本管理：快照落盘、列出标签、基于同一批问句的向量检索回归对比。

回归指标：
- 对每条问句分别在索引 A、B 上做 search_with_paths，取 Top-1 的 similarity；
- 对所有问句求平均，较高者视为检索匹配略优（不等价于回答正确率）。

依赖：
- compare_indexes_by_queries 内部延迟导入 rag_service.search_with_paths，需可用嵌入 API。
- snapshot_version 仅文件拷贝，不要求调用模型。
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (  # noqa: E402
    INDEX_FILE,
    METADATA_FILE,
    REGRESSION_SUITE_FILE,
    VERSIONS_DIR,
)
from env_loader import require_dashscope_key  # noqa: E402


def _safe_label(label: str) -> str:
    """
    校验版本目录名，防止路径注入或非法字符。

    Raises:
        ValueError: 空串或含非法字符。
    """
    s = label.strip()
    if not s:
        raise ValueError("版本标签不能为空")
    if not re.match(r"^[\w.\-]+$", s):
        raise ValueError("版本标签仅允许字母数字、点、下划线、短横线")
    return s


def snapshot_version(label: str, note: str = "") -> dict[str, Any]:
    """
    将当前线上 INDEX_FILE、METADATA_FILE 复制到 VERSIONS_DIR/<label>/，并写 manifest.json。

    Args:
        label: 目录名标签。
        note: 可选备注，写入 manifest。

    Returns:
        manifest 字典。

    Raises:
        FileExistsError: 同名版本目录已存在。
        FileNotFoundError: 当前索引或元数据缺失。
    """
    lab = _safe_label(label)
    target = VERSIONS_DIR / lab
    if target.exists():
        raise FileExistsError(f"版本已存在: {lab}")
    target.mkdir(parents=True)
    files = []
    for src in (INDEX_FILE, METADATA_FILE):
        if not src.is_file():
            raise FileNotFoundError(f"缺少文件无法快照: {src}")
        dst = target / src.name
        shutil.copy2(src, dst)
        files.append(dst.name)

    manifest = {
        "label": lab,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": note,
        "files": files,
    }
    with open(target / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def list_versions() -> list[dict[str, Any]]:
    """
    扫描 VERSIONS_DIR 子目录，读取 manifest.json；若无 manifest 则返回占位 dict。
    """
    if not VERSIONS_DIR.is_dir():
        return []
    out = []
    for p in sorted(VERSIONS_DIR.iterdir()):
        if not p.is_dir():
            continue
        man = p / "manifest.json"
        if man.is_file():
            with open(man, "r", encoding="utf-8") as f:
                out.append(json.load(f))
        else:
            out.append({"label": p.name, "created_at": "", "note": "", "files": []})
    return out


def default_regression_questions() -> list[str]:
    """未生成 regression_questions.json 时使用的内置少量问句（上海迪士尼主题）。"""
    return [
        "上海迪士尼一日票平日成人票价大约多少？",
        "老人票年龄要求是什么？",
        "万圣节海报主题是什么？",
    ]


def load_regression_questions() -> list[str]:
    """
    优先加载 REGRESSION_SUITE_FILE 中的 questions 数组；
    文件缺失或为空则回退 default_regression_questions。
    """
    if REGRESSION_SUITE_FILE.is_file():
        with open(REGRESSION_SUITE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        qs = data.get("questions")
        if isinstance(qs, list) and qs:
            return [str(q).strip() for q in qs if str(q).strip()]
    return default_regression_questions()


def _mean(vals: list[float]) -> float:
    """算术均值；空列表返回 0.0。"""
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def compare_indexes_by_queries(
    index_path_a: Path,
    metadata_path_a: Path,
    index_path_b: Path,
    metadata_path_b: Path,
    questions: list[str],
    top_k: int = 5,
) -> dict[str, Any]:
    """
    对两套「索引+元数据」用同一批问句做检索，比较每条问句的 Top-1 相似度并汇总。

    不调用生成模型，仅嵌入 + FAISS；每次检索前 search_with_paths 会设置 dashscope.api_key。

    Args:
        index_path_* / metadata_path_*：两套文件的 pathlib.Path。
        questions：回归问句列表。
        top_k：传给 FAISS 的检索条数（比较时常取 Top-1，保留参数便于扩展）。

    Returns:
        含 avg_top1_similarity_a/b、逐题 details、recommendation、better_retrieval 等字段。
    """
    require_dashscope_key()

    # 延迟导入避免 kb_versions ↔ rag_service 顶层循环依赖
    from rag_service import search_with_paths  # noqa: E402

    sims_a: list[float] = []
    sims_b: list[float] = []
    details = []

    for q in questions:
        ha = search_with_paths(q, index_path_a, metadata_path_a, top_k=top_k)
        hb = search_with_paths(q, index_path_b, metadata_path_b, top_k=top_k)
        sa = ha[0]["similarity"] if ha else 0.0
        sb = hb[0]["similarity"] if hb else 0.0
        sims_a.append(float(sa))
        sims_b.append(float(sb))
        winner = "tie"
        if sa > sb + 1e-9:
            winner = "a"
        elif sb > sa + 1e-9:
            winner = "b"
        details.append(
            {
                "question": q,
                "top1_similarity_a": round(sa, 6),
                "top1_similarity_b": round(sb, 6),
                "better": winner,
            }
        )

    ma = _mean(sims_a)
    mb = _mean(sims_b)
    if ma > mb + 1e-9:
        rec = "版本 A 平均 Top-1 相似度更高，检索匹配略优。"
        better = "a"
    elif mb > ma + 1e-9:
        rec = "版本 B 平均 Top-1 相似度更高，检索匹配略优。"
        better = "b"
    else:
        rec = "两版本检索相似度整体接近，可结合健康度报告与业务抽样再定夺。"
        better = "tie"

    return {
        "question_count": len(questions),
        "avg_top1_similarity_a": round(ma, 6),
        "avg_top1_similarity_b": round(mb, 6),
        "details": details,
        "recommendation": rec,
        "better_retrieval": better,
    }


def regression_current_vs_label(version_label: str, top_k: int = 5) -> dict[str, Any]:
    """
    左侧为当前 data 目录下的 INDEX_FILE/METADATA_FILE，右侧为 VERSIONS_DIR/<label>/ 中的副本。
    """
    lab = _safe_label(version_label)
    vdir = VERSIONS_DIR / lab
    if not vdir.is_dir():
        raise FileNotFoundError(f"未找到版本: {lab}")

    idx_a = INDEX_FILE
    meta_a = METADATA_FILE
    idx_b = vdir / INDEX_FILE.name
    meta_b = vdir / METADATA_FILE.name
    if not idx_b.is_file() or not meta_b.is_file():
        raise FileNotFoundError(f"版本 {lab} 缺少索引或元数据文件")

    qs = load_regression_questions()
    cmp_ret = compare_indexes_by_queries(
        idx_a, meta_a, idx_b, meta_b, qs, top_k=top_k
    )
    return {
        "mode": "current_vs_version",
        "version_b": lab,
        **cmp_ret,
    }


def regression_two_versions(label_a: str, label_b: str, top_k: int = 5) -> dict[str, Any]:
    """
    两套快照均在 VERSIONS_DIR 下；A/B 分别作为 compare_indexes_by_queries 中的第一套与第二套索引。
    """
    a = _safe_label(label_a)
    b = _safe_label(label_b)
    da = VERSIONS_DIR / a
    db = VERSIONS_DIR / b
    for name, p in (("A", da), ("B", db)):
        if not p.is_dir():
            raise FileNotFoundError(f"未找到版本 {name}: {p}")
    idx_a = da / INDEX_FILE.name
    meta_a = da / METADATA_FILE.name
    idx_b = db / INDEX_FILE.name
    meta_b = db / METADATA_FILE.name

    qs = load_regression_questions()
    cmp_ret = compare_indexes_by_queries(
        idx_a, meta_a, idx_b, meta_b, qs, top_k=top_k
    )
    return {
        "mode": "version_vs_version",
        "version_a": a,
        "version_b": b,
        **cmp_ret,
    }
