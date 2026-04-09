"""生成第6章与附录的 LaTeX 表格文件。"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_LABELS = {
    "cora": "Cora",
    "citeseer": "Citeseer",
    "pubmed": "PubMed",
    "ogbn_arxiv": "ogbn-arxiv",
    "chameleon": "Chameleon",
    "squirrel": "Squirrel",
}
MAIN_HOMOPHILY_ROWS = [
    ("gcn", "GCN", False),
    ("sgc", "SGC", False),
    ("pprgo", "PPRGo", False),
    ("glnn", "GLNN", False),
    ("b0", "SDGNN", False),
    ("b5", "MSAS-GNN", True),
]
MAIN_HETEROPHILY_ROWS = [
    ("gcn", "GCN", False),
    ("sgc", "SGC", False),
    ("pprgo", "PPRGo", False),
    ("geom_gcn", "\\textit{Geom-GCN}", False),
    ("h2gcn", "\\textit{H2GCN}", False),
    ("glnn", "GLNN", False),
    ("b0", "SDGNN", False),
    ("b5", "MSAS-GNN", True),
]
ABLATION_ROWS = [
    ("b0", "B0", "SDGNN基线", False),
    ("b1", "B1", "+谱能量$E_{\\text{spectral}}(i)$（第4章节点级正则系数设计的核心项）", False),
    ("b2", "B2", "+度中心性$C_{\\text{deg}}(i)$", False),
    ("b3", "B3", "+$k$-core指数", False),
    ("b4", "B4", "+局部图熵$H(i)$", False),
    ("b5", "B5", "+分层跳距预算$k_i^{(l)}$（第4章跳距预算分配结论）", True),
    ("b2_rnd", "B2-RND", "B2 + 随机扰动$\\tau(i)$（对照）", False),
]
HOP_ROWS = [
    ("uniform", "均匀分配（各层相等）", False),
    ("xi10", "$\\xi=1.0$（工程近似）", False),
    ("xi05", "$\\xi=0.5$（第4章第4.3节分配参考形式）", True),
    ("reverse", "反向分配（深层更多）", False),
]
EFFICIENCY_ROWS = [
    ("gcn", "GCN", False),
    ("sgc", "SGC", False),
    ("pprgo", "PPRGo", False),
    ("glnn", "GLNN", False),
    ("sdgnn", "SDGNN", False),
    ("msas_gnn", "MSAS-GNN", True),
]
EFFICIENCY_PARAM_FALLBACK = {
    "gcn": 0.47,
    "sgc": 0.23,
    "pprgo": 0.35,
    "glnn": 0.48,
    "sdgnn": 0.68,
    "msas_gnn": 0.72,
}
def _normalize_method_id(method_id: Any) -> str | None:
    if method_id is None:
        return None
    method_id = str(method_id).lower()
    aliases = {
        "sdgnn": "b0",
        "msas_gnn": "b5",
        "msas_gnn_b5": "b5",
    }
    return aliases.get(method_id, method_id)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def _latest_file(results_dir: Path, pattern: str) -> Path | None:
    candidates = list(results_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _latest_payloads(results_dir: Path, pattern: str, key_builder) -> dict[Any, dict[str, Any]]:
    latest: dict[Any, tuple[float, dict[str, Any]]] = {}
    for path in results_dir.glob(pattern):
        try:
            payload = _load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("跳过损坏结果文件 %s: %s", path, exc)
            continue
        key = key_builder(payload)
        if key is None:
            continue
        stamp = path.stat().st_mtime
        prev = latest.get(key)
        if prev is None or stamp >= prev[0]:
            latest[key] = (stamp, payload)
    return {key: payload for key, (_, payload) in latest.items()}


def _protocol_version(payload: dict[str, Any]) -> int:
    protocols = payload.get("protocols")
    if isinstance(protocols, dict):
        return int(protocols.get("table_protocol_version", 0) or 0)
    return 0


def _ablation_payload_view(payload: dict[str, Any]) -> dict[str, Any]:
    """兼容历史 ablation 汇总格式与新版单次实测格式。"""

    support_total = payload.get("mean_support_total", payload.get("support_total"))
    candidate_total = payload.get("mean_candidate_total", payload.get("candidate_total"))
    derived_sparsity = None
    support_value = _safe_float(support_total)
    candidate_value = _safe_float(candidate_total)
    if support_value is not None and candidate_value is not None and candidate_value > 0:
        derived_sparsity = 1.0 - support_value / candidate_value

    return {
        "mean_acc": payload.get("mean_acc", payload.get("clean_test_acc")),
        "std_acc": payload.get("std_acc"),
        "noise_mean_acc": payload.get("noise_mean_acc", payload.get("noise_test_acc")),
        "mean_sparsity": payload.get(
            "mean_sparsity",
            payload.get("pruning_sparsity", payload.get("legacy_sparsity", derived_sparsity)),
        ),
        "mean_inference_ms": payload.get("mean_inference_ms", payload.get("inference_ms")),
        "mean_candidate_total": candidate_total,
        "mean_support_total": support_total,
    }


def _ablation_payload_quality(payload: dict[str, Any], stamp: float) -> tuple[int, int, int, int, float]:
    view = _ablation_payload_view(payload)
    has_core_metrics = int(_safe_float(view.get("mean_acc")) is not None)
    has_noise = int(_safe_float(view.get("noise_mean_acc")) is not None)
    has_new_sparsity = int(
        payload.get("pruning_sparsity") is not None
        or payload.get("mean_candidate_total") is not None
        or payload.get("candidate_total") is not None
        or (
            isinstance(payload.get("protocols"), dict)
            and payload["protocols"].get("sparsity_protocol") == "candidate_pruning_rate"
        )
    )
    is_summary = int("mean_acc" in payload)
    return (
        has_core_metrics,
        has_new_sparsity,
        is_summary,
        _protocol_version(payload),
        stamp,
    )


def _ensure_results(payloads: dict[Any, dict[str, Any]], group_name: str) -> None:
    if not payloads:
        raise FileNotFoundError(f"未找到 {group_name} 所需结果文件")


def _maybe_bold(cell: str, highlight: bool) -> str:
    return f"\\textbf{{{cell}}}" if highlight and cell != "--" else cell


def _format_acc_pm(mean: Any, std: Any, digits: int = 1) -> str:
    mean_value = _safe_float(mean)
    if mean_value is None:
        return "--"
    std_value = _safe_float(std)
    cell = f"{mean_value * 100:.{digits}f}"
    if std_value is not None:
        cell += f"$\\pm${std_value * 100:.{digits}f}"
    return cell


def _format_percent(mean: Any, digits: int = 1) -> str:
    value = _safe_float(mean)
    return "--" if value is None else f"{value * 100:.{digits}f}"


def _format_float(value: Any, digits: int = 3) -> str:
    parsed = _safe_float(value)
    return "--" if parsed is None else f"{parsed:.{digits}f}"


def _format_ms(value: Any, digits: int = 1) -> str:
    parsed = _safe_float(value)
    return "--" if parsed is None else f"{parsed:.{digits}f}"


def _format_memory_mb(value: Any) -> str:
    parsed = _safe_float(value)
    return "--" if parsed is None else f"{parsed:,.0f}"


def _format_speedup(value: Any) -> str:
    parsed = _safe_float(value)
    return "--" if parsed is None else f"${parsed:.1f}\\times$"


def _format_param_m(value: Any) -> str:
    parsed = _safe_float(value)
    return "--" if parsed is None else f"{parsed:.2f}"


def _format_relative_overhead(value: Any) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "--"
    pct = parsed * 100.0
    if abs(pct) < 1e-9:
        return "基准"
    sign = "+" if pct > 0 else ""
    return f"基准{sign}{pct:.0f}\\%"


def _format_approx_seconds(value: Any) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "--"
    if abs(parsed) >= 100:
        return f"约{parsed:,.0f}"
    if abs(parsed) >= 1:
        return f"约{parsed:.2f}"
    return f"约{parsed:.3f}"


def _main_results(results_dir: Path) -> dict[Any, dict[str, Any]]:
    payloads = _latest_payloads(
        results_dir,
        "main_*.json",
        lambda payload: (
            _normalize_method_id(payload.get("method_id") or payload.get("ablation_id")),
            payload.get("dataset"),
        )
        if payload.get("dataset") is not None
        else None,
    )
    _ensure_results(payloads, "主实验")
    return payloads


def _ablation_results(results_dir: Path) -> dict[Any, dict[str, Any]]:
    latest: dict[Any, tuple[tuple[int, int, int, int, float], dict[str, Any]]] = {}
    for path in results_dir.glob("ablation_*.json"):
        try:
            payload = _load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("跳过损坏结果文件 %s: %s", path, exc)
            continue
        key = (
            payload.get("ablation_id"),
            payload.get("dataset"),
        )
        if key[0] is None or key[1] is None:
            continue
        stamp = path.stat().st_mtime
        quality = _ablation_payload_quality(payload, stamp)
        prev = latest.get(key)
        if prev is None or quality >= prev[0]:
            latest[key] = (quality, payload)
    payloads = {key: payload for key, (_, payload) in latest.items()}
    _ensure_results(payloads, "逐模块消融")
    return payloads


def _efficiency_payload(results_dir: Path) -> dict[str, Any] | None:
    latest = _latest_file(results_dir, "efficiency_*.json")
    return _load_json(latest) if latest is not None else None


def _hop_payload(results_dir: Path) -> dict[str, Any]:
    latest = _latest_file(results_dir, "hop_strategy_*.json")
    if latest is None:
        raise FileNotFoundError("未找到 hop_strategy_*.json")
    return _load_json(latest)


def _breakeven_payload(results_dir: Path) -> dict[str, Any]:
    fixed_path = results_dir / "breakeven_analysis.json"
    if fixed_path.exists():
        return _load_json(fixed_path)
    latest = _latest_file(results_dir, "*breakeven*.json")
    if latest is None:
        raise FileNotFoundError("未找到 break-even 结果")
    return _load_json(latest)


def _appendix_sigma_payload(results_dir: Path) -> dict[str, Any]:
    merged_path = results_dir / "appendix_sigma_proxy.json"
    if merged_path.exists():
        payload = _load_json(merged_path)
        if isinstance(payload, dict):
            return payload
    data: dict[str, Any] = {}
    for dataset in ("citeseer", "ogbn_arxiv"):
        path = results_dir / f"appendix_sigma_proxy_{dataset}.json"
        if path.exists():
            data[dataset] = _load_json(path)
    if not data:
        raise FileNotFoundError("未找到 appendix_sigma_proxy 结果")
    return data


def _time_cell(efficiency_payload: dict[str, Any] | None, dataset: str, method_id: str) -> str:
    if not efficiency_payload:
        return "--"
    method_key = {"b0": "sdgnn", "b5": "msas_gnn"}.get(method_id, method_id)
    payload = efficiency_payload.get("per_dataset", {}).get(dataset, {}).get(method_key, {})
    return _format_ms(payload.get("median_ms_per_batch") or payload.get("median_ms"))


def _write_table(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("已生成 %s", path)


def build_main_tables(results_dir: Path, output_root: Path) -> list[Path]:
    main_results = _main_results(results_dir)
    efficiency_payload = _efficiency_payload(results_dir)
    outputs: list[Path] = []

    homophily_lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{引文网络与大规模图节点分类准确率（\%，均值$\pm$标准差，10次重复）}",
        r"  \label{tab:ch6-homophily-large}",
        r"  \begin{threeparttable}",
        r"    \begin{tabular}{>{\centering\arraybackslash}m{2.34cm} >{\centering\arraybackslash}m{1.63cm} >{\centering\arraybackslash}m{1.93cm} >{\centering\arraybackslash}m{1.73cm} >{\centering\arraybackslash}m{2.56cm} >{\centering\arraybackslash}m{2.44cm}}",
        r"\toprule",
        r"\textbf{方法} & \textbf{Cora} & \textbf{Citeseer} & \textbf{PubMed} & \textbf{ogbn-arxiv（测试）} & \textbf{ogbn-arxiv 推理时间（ms/batch）} \\",
        r"\midrule",
    ]
    for method_id, label, highlight in MAIN_HOMOPHILY_ROWS:
        row = [_maybe_bold(label, highlight)]
        for dataset in ("cora", "citeseer", "pubmed", "ogbn_arxiv"):
            payload = main_results.get((method_id, dataset), {})
            row.append(_maybe_bold(_format_acc_pm(payload.get("mean_acc"), payload.get("std_acc")), highlight))
        row.append(_maybe_bold(_time_cell(efficiency_payload, "ogbn_arxiv", method_id), highlight))
        homophily_lines.append(" & ".join(row) + r" \\")
    homophily_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"    \begin{tablenotes}",
            r"\item 注：若当前效率日志尚未生成，则 ogbn-arxiv 推理时间列以“--”占位；重新运行效率实验后可自动补齐该列。",
            r"\end{tablenotes}",
            r"  \end{threeparttable}",
            r"\end{table}",
        ]
    )
    homophily_path = output_root / "table_6_2_homophily_large.tex"
    _write_table(homophily_path, homophily_lines)
    outputs.append(homophily_path)

    heterophily_lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{网页图节点分类准确率（\%，均值$\pm$标准差，10次重复）}",
        r"  \label{tab:ch6-heterophily}",
        r"  \begin{threeparttable}",
        r"    \begin{tabular}{>{\centering\arraybackslash}m{2.54cm} >{\centering\arraybackslash}m{2.09cm} >{\centering\arraybackslash}m{2.09cm}}",
        r"\toprule",
        r"\textbf{方法} & \textbf{Chameleon} & \textbf{Squirrel} \\",
        r"\midrule",
    ]
    for method_id, label, highlight in MAIN_HETEROPHILY_ROWS:
        row = [_maybe_bold(label, highlight)]
        for dataset in ("chameleon", "squirrel"):
            payload = main_results.get((method_id, dataset), {})
            row.append(_maybe_bold(_format_acc_pm(payload.get("mean_acc"), payload.get("std_acc")), highlight))
        heterophily_lines.append(" & ".join(row) + r" \\")
    heterophily_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"    \begin{tablenotes}",
            r"\item 注：本表仅依赖主实验结果文件；缺失方法会以“--”占位，便于后续增量补跑后直接重生成。",
            r"\end{tablenotes}",
            r"  \end{threeparttable}",
            r"\end{table}",
        ]
    )
    heterophily_path = output_root / "table_6_3_heterophily.tex"
    _write_table(heterophily_path, heterophily_lines)
    outputs.append(heterophily_path)
    return outputs


def build_ablation_tables(results_dir: Path, output_root: Path) -> list[Path]:
    outputs: list[Path] = []
    ablation_results = _ablation_results(results_dir)
    ablation_lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"\caption{综合消融实验（Cora，准确率\%，主体结果为10次重复；个别日志缺失项以“--”表示）}",
        r"  \label{tab:ch6-ablation}",
        r"  \begin{threeparttable}",
        r"    \begin{tabular}{>{\centering\arraybackslash}m{1.3cm} >{\centering\arraybackslash}m{3.6cm} >{\centering\arraybackslash}m{1.9cm} >{\centering\arraybackslash}m{1.9cm} >{\centering\arraybackslash}m{1.5cm} >{\centering\arraybackslash}m{1.5cm}}",
        r"\toprule",
        r"\textbf{配置} & \textbf{模块组成} & \textbf{干净图准确率} & \textbf{30\%噪声准确率} & \textbf{稀疏度（\%）} & \textbf{推理时间（ms）} \\",
        r"\midrule",
    ]
    for ablation_id, label, desc, highlight in ABLATION_ROWS:
        if ablation_id == "b2_rnd":
            ablation_lines.append(r"\midrule")
        payload = _ablation_payload_view(ablation_results.get((ablation_id, "cora"), {}))
        row = [
            _maybe_bold(label, highlight),
            _maybe_bold(desc, highlight),
            _maybe_bold(_format_acc_pm(payload.get("mean_acc"), payload.get("std_acc")), highlight),
            _maybe_bold(_format_percent(payload.get("noise_mean_acc")), highlight),
            _maybe_bold(_format_percent(payload.get("mean_sparsity")), highlight),
            _maybe_bold(_format_ms(payload.get("mean_inference_ms")), highlight),
        ]
        ablation_lines.append(" & ".join(row) + r" \\")
    ablation_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"    \begin{tablenotes}",
            r"\item 注：本表默认读取最新的 `ablation_*.json` 结果；30\%噪声列仅报告均值，便于与正文口径保持一致。",
            r"\end{tablenotes}",
            r"  \end{threeparttable}",
            r"\end{table}",
        ]
    )
    ablation_path = output_root / "table_6_4_ablation.tex"
    _write_table(ablation_path, ablation_lines)
    outputs.append(ablation_path)

    hop_payload = _hop_payload(results_dir)
    hop_rows = hop_payload.get("table_rows", {})
    hop_lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{跳距预算分配策略消融（3层GNN，准确率\%）}",
        r"  \label{tab:ch6-hopbudget}",
        r"  \begin{threeparttable}",
        r"    \begin{tabular}{>{\centering\arraybackslash}m{3.04cm} >{\centering\arraybackslash}m{2.19cm} >{\centering\arraybackslash}m{2.66cm} >{\centering\arraybackslash}m{2.38cm} >{\centering\arraybackslash}m{1.9cm}}",
        r"\toprule",
        r"\textbf{分配策略} & \textbf{Cora准确率} & \textbf{Chameleon准确率} & \textbf{平均$\varepsilon_{\text{approx}}$} & \textbf{相对计算开销} \\",
        r"\midrule",
    ]
    for strategy, label, highlight in HOP_ROWS:
        row_payload = hop_rows.get(strategy, {})
        row = [
            _maybe_bold(label, highlight),
            _maybe_bold(_format_acc_pm(row_payload.get("cora_mean_acc"), row_payload.get("cora_std_acc")), highlight),
            _maybe_bold(_format_acc_pm(row_payload.get("chameleon_mean_acc"), row_payload.get("chameleon_std_acc")), highlight),
            _maybe_bold(_format_float(row_payload.get("mean_epsilon_approx")), highlight),
            _maybe_bold(_format_relative_overhead(row_payload.get("relative_compute_overhead")), highlight),
        ]
        hop_lines.append(" & ".join(row) + r" \\")
    hop_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"    \begin{tablenotes}",
            r"\item 注：相对计算开销基于 `hop_strategy_*.json` 中各策略的 `mean_alternating_opt_seconds` 自动换算；缺失项以“--”表示。",
            r"\end{tablenotes}",
            r"  \end{threeparttable}",
            r"\end{table}",
        ]
    )
    hop_path = output_root / "table_6_5_hopbudget.tex"
    _write_table(hop_path, hop_lines)
    outputs.append(hop_path)
    return outputs


def build_efficiency_tables(results_dir: Path, output_root: Path) -> list[Path]:
    outputs: list[Path] = []
    efficiency_payload = _efficiency_payload(results_dir)
    if efficiency_payload is None:
        raise FileNotFoundError("未找到 efficiency_*.json")
    per_dataset = efficiency_payload.get("per_dataset", {})
    summary = efficiency_payload.get("summary", {})
    gcn_mem = _safe_float(per_dataset.get("ogbn_arxiv", {}).get("gcn", {}).get("peak_memory_mb"))
    efficiency_lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{推理效率与显存占用综合对比（GPU，$\text{batch\_size}=1024$）}",
        r"  \label{tab:ch6-infer}",
        r"  \begin{threeparttable}",
        r"    \begin{tabular}{>{\centering\arraybackslash}m{2.04cm} >{\centering\arraybackslash}m{1.33cm} >{\centering\arraybackslash}m{1.33cm} >{\centering\arraybackslash}m{1.68cm} >{\centering\arraybackslash}m{1.77cm} >{\centering\arraybackslash}m{1.50cm} >{\centering\arraybackslash}m{1.68cm}}",
        r"\toprule",
        r"\textbf{方法} & \textbf{Cora（ms）} & \textbf{ogbn-arxiv（ms）} & \textbf{平均加速比（vs.~GCN，四数据集）} & \textbf{总显存（MB）} & \textbf{显存节省} & \textbf{参数量（M）} \\",
        r"\midrule",
    ]
    for method_id, label, highlight in EFFICIENCY_ROWS:
        cora_payload = per_dataset.get("cora", {}).get(method_id, {})
        large_payload = per_dataset.get("ogbn_arxiv", {}).get(method_id, {})
        speedup_cell = _format_speedup(summary.get(method_id, {}).get("avg_speedup_vs_gcn"))
        memory_value = _safe_float(large_payload.get("peak_memory_mb"))
        if method_id == "gcn":
            saving_cell = "--"
        elif memory_value is None or gcn_mem is None:
            saving_cell = "--"
        else:
            saving_cell = f"{(1.0 - memory_value / gcn_mem) * 100:.1f}\\%"
        param_count = large_payload.get("parameter_count_millions", EFFICIENCY_PARAM_FALLBACK.get(method_id))
        row = [
            _maybe_bold(label, highlight),
            _maybe_bold(_format_ms(cora_payload.get("median_ms")), highlight),
            _maybe_bold(_format_ms(large_payload.get("median_ms_per_batch")), highlight),
            _maybe_bold(speedup_cell, highlight),
            _maybe_bold(_format_memory_mb(memory_value), highlight),
            _maybe_bold(saving_cell, highlight),
            _maybe_bold(_format_param_m(param_count), highlight),
        ]
        efficiency_lines.append(" & ".join(row) + r" \\")
    efficiency_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"    \begin{tablenotes}",
            r"\item 注：参数量列优先读取 `efficiency_*.json` 中的真实统计字段；仅对旧版日志保留回退值兼容。",
            r"\end{tablenotes}",
            r"  \end{threeparttable}",
            r"\end{table}",
        ]
    )
    efficiency_path = output_root / "table_6_6_efficiency.tex"
    _write_table(efficiency_path, efficiency_lines)
    outputs.append(efficiency_path)

    breakeven_payload = _breakeven_payload(results_dir)
    breakeven_lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{预处理摊销与break-even估计（时间单位统一为秒）}",
        r"  \label{tab:ch6-breakeven}",
        r"  \begin{threeparttable}",
        r"    \begin{tabular}{>{\centering\arraybackslash}m{1.84cm} >{\centering\arraybackslash}m{1.84cm} >{\centering\arraybackslash}m{1.84cm} >{\centering\arraybackslash}m{1.84cm} >{\centering\arraybackslash}m{2.39cm} >{\centering\arraybackslash}m{2.01cm}}",
        r"\toprule",
        r"\textbf{数据集} & \textbf{$t_{\text{pre}}$（s）} & \textbf{$t_{\text{dense}}$（s）} & \textbf{$t_{\text{sparse}}$（s）} & \textbf{$t_{\text{dense}}-t_{\text{sparse}}$（s）} & \textbf{$Q_{\text{be}}$（调用次数）} \\",
        r"\midrule",
    ]
    for dataset in ("cora", "pubmed", "ogbn_arxiv"):
        payload = breakeven_payload.get(dataset, {})
        t_dense = _safe_float(payload.get("t_dense"))
        t_sparse = _safe_float(payload.get("t_sparse"))
        delta = None if t_dense is None or t_sparse is None else t_dense - t_sparse
        row = [
            DATASET_LABELS.get(dataset, dataset),
            _format_approx_seconds(payload.get("t_pre")),
            _format_approx_seconds(t_dense),
            _format_approx_seconds(t_sparse),
            _format_approx_seconds(delta),
            _format_approx_seconds(payload.get("Q_be")),
        ]
        breakeven_lines.append(" & ".join(row) + r" \\")
    breakeven_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"    \begin{tablenotes}",
            r"\item 注：本表直接读取 `breakeven_analysis.json`；若只跑了部分数据集，未生成的数据集会以“--”占位。",
            r"\end{tablenotes}",
            r"  \end{threeparttable}",
            r"\end{table}",
        ]
    )
    breakeven_path = output_root / "table_6_7_breakeven.tex"
    _write_table(breakeven_path, breakeven_lines)
    outputs.append(breakeven_path)
    return outputs


def build_appendix_tables(results_dir: Path, output_root: Path) -> list[Path]:
    outputs: list[Path] = []
    appendix_root = output_root / "appendix"
    spectral_payload = _appendix_sigma_payload(results_dir)
    spectral_lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{谱相似性代理量$\widetilde{\sigma}_{\mathrm{proxy}}$验证补充（Citeseer与ogbn-arxiv，$K_{\text{poly}}=3$）}",
        r"  \label{tab:appC-spectral}",
        r"  \begin{threeparttable}",
        r"    \begin{tabular}{>{\centering\arraybackslash}m{1.5cm}",
        r"                    >{\centering\arraybackslash}m{1.7cm}",
        r"                    >{\centering\arraybackslash}m{2.0cm}",
        r"                    >{\centering\arraybackslash}m{2.7cm}",
        r"                    >{\centering\arraybackslash}m{1.9cm}",
        r"                    >{\centering\arraybackslash}m{2.7cm}}",
        r"      \toprule",
        r"      \textbf{数据集} & \textbf{预算$k$} & \textbf{谱相似性代理量$\widetilde{\sigma}_{\mathrm{proxy}}$} &",
        r"      \textbf{按简化系数计算的工程参考值} & \textbf{实测$\varepsilon_{\text{approx}}$} &",
        r"      \textbf{节点分类准确率（\%）} \\",
        r"      \midrule",
    ]
    for dataset_index, dataset in enumerate(("citeseer", "ogbn_arxiv")):
        rows = sorted(spectral_payload.get(dataset, []), key=lambda row: row.get("k", 0))
        if not rows:
            rows = [{} for _ in range(5)]
        for row_index, row in enumerate(rows):
            prefix = (
                f"      \\multirow{{{len(rows)}}}{{*}}{{{DATASET_LABELS.get(dataset, dataset)}}}"
                if row_index == 0
                else "      "
            )
            spectral_lines.append(
                prefix
                + f" & {row.get('k', '--')}"
                + " & "
                + _format_float(row.get("sigma_proxy_mean", row.get("sigma_proxy")), digits=2)
                + " & "
                + _format_float(row.get("engineering_ref_mean", row.get("engineering_ref")), digits=3)
                + " & "
                + _format_float(row.get("epsilon_approx_mean", row.get("epsilon_approx")), digits=3)
                + " & "
                + _format_acc_pm(row.get("acc_mean", row.get("acc")), row.get("acc_std"))
                + r" \\"
            )
        if dataset_index != 1:
            spectral_lines.append(r"      \midrule")
    spectral_lines.extend(
        [
            r"      \bottomrule",
            r"    \end{tabular}",
            r"    \begin{tablenotes}",
            r"      \item 注：若 `appendix_sigma_proxy.json` 已按多种子聚合生成，则本表读取其均值列；否则回退为现有单次结果字段。",
            r"    \end{tablenotes}",
            r"  \end{threeparttable}",
            r"\end{table}",
        ]
    )
    spectral_path = appendix_root / "table_appC_spectral.tex"
    _write_table(spectral_path, spectral_lines)
    outputs.append(spectral_path)
    return outputs


def build_selected_tables(results_dir: Path, output_root: Path, groups: set[str]) -> list[Path]:
    outputs: list[Path] = []
    if "main" in groups:
        outputs.extend(build_main_tables(results_dir, output_root))
    if "ablation" in groups:
        outputs.extend(build_ablation_tables(results_dir, output_root))
    if "efficiency" in groups:
        outputs.extend(build_efficiency_tables(results_dir, output_root))
    if "appendix" in groups:
        outputs.extend(build_appendix_tables(results_dir, output_root))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--efficiency", action="store_true")
    parser.add_argument("--appendix", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--results_dir", default="outputs/results")
    parser.add_argument("--output_dir", default="outputs/tables")
    args = parser.parse_args()

    groups = set()
    if args.main or args.all:
        groups.add("main")
    if args.ablation or args.all:
        groups.add("ablation")
    if args.efficiency or args.all:
        groups.add("efficiency")
    if args.appendix or args.all:
        groups.add("appendix")
    if not groups:
        parser.print_help()
        return

    outputs = build_selected_tables(Path(args.results_dir), Path(args.output_dir), groups)
    logger.info("完成，共生成 %s 个 LaTeX 表格。", len(outputs))


if __name__ == "__main__":
    main()
