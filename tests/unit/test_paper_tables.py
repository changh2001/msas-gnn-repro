"""论文表格生成回归测试。"""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    path = ROOT / "scripts/visualization/build_paper_tables.py"
    spec = spec_from_file_location("build_paper_tables", path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_build_selected_tables_generates_main_ch6_and_supplemental_outputs(tmp_path):
    module = _load_module()
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "tables"

    _write_json(
        results_dir / "main_b5_cora_1.json",
        {"dataset": "cora", "method_id": "b5", "mean_acc": 0.883, "std_acc": 0.007},
    )
    _write_json(
        results_dir / "main_b5_chameleon_1.json",
        {"dataset": "chameleon", "method_id": "b5", "mean_acc": 0.672, "std_acc": 0.009},
    )
    _write_json(
        results_dir / "main_b5_squirrel_1.json",
        {"dataset": "squirrel", "method_id": "b5", "mean_acc": 0.569, "std_acc": 0.012},
    )
    _write_json(
        results_dir / "main_gcn_cora_1.json",
        {"dataset": "cora", "method_id": "gcn", "mean_acc": 0.861, "std_acc": 0.010},
    )
    _write_json(
        results_dir / "main_gcn_chameleon_1.json",
        {"dataset": "chameleon", "method_id": "gcn", "mean_acc": 0.621, "std_acc": 0.011},
    )
    _write_json(
        results_dir / "main_gcn_squirrel_1.json",
        {"dataset": "squirrel", "method_id": "gcn", "mean_acc": 0.538, "std_acc": 0.013},
    )

    _write_json(
        results_dir / "ablation_b0_cora_1.json",
        {
            "ablation_id": "b0",
            "dataset": "cora",
            "mean_acc": 0.866,
            "std_acc": 0.009,
            "noise_mean_acc": 0.783,
            "mean_inference_ms": 1.8,
        },
    )
    _write_json(
        results_dir / "ablation_b5_cora_1.json",
        {
            "ablation_id": "b5",
            "dataset": "cora",
            "mean_acc": 0.883,
            "std_acc": 0.007,
            "noise_mean_acc": 0.810,
            "mean_sparsity": 0.737,
            "mean_inference_ms": 2.0,
        },
    )
    _write_json(
        results_dir / "ablation_b5_cora_seed42_real_2.json",
        {
            "ablation_id": "b5",
            "dataset": "cora",
            "clean_test_acc": 0.875,
            "noise_test_acc": 0.830,
            "pruning_sparsity": 0.9735,
            "candidate_total": 344138,
            "support_total": 9125,
            "inference_ms": 1.8,
            "protocols": {
                "table_protocol_version": 2,
                "sparsity_protocol": "candidate_pruning_rate",
            },
        },
    )
    _write_json(
        results_dir / "ablation_b2_rnd_cora_1.json",
        {
            "ablation_id": "b2_rnd",
            "dataset": "cora",
            "mean_acc": 0.873,
            "std_acc": 0.009,
            "noise_mean_acc": 0.789,
            "mean_sparsity": 0.754,
            "mean_inference_ms": 1.9,
        },
    )

    _write_json(
        results_dir / "hop_strategy_1.json",
        {
            "table_rows": {
                "uniform": {
                    "cora_mean_acc": 0.872,
                    "cora_std_acc": 0.008,
                    "chameleon_mean_acc": 0.661,
                    "chameleon_std_acc": 0.011,
                    "mean_epsilon_approx": 0.182,
                    "relative_compute_overhead": 0.0,
                },
                "xi05": {
                    "cora_mean_acc": 0.880,
                    "cora_std_acc": 0.007,
                    "chameleon_mean_acc": 0.673,
                    "chameleon_std_acc": 0.009,
                    "mean_epsilon_approx": 0.151,
                    "relative_compute_overhead": 0.05,
                },
            }
        },
    )

    _write_json(
        results_dir / "efficiency_1.json",
        {
            "per_dataset": {
                "cora": {
                    "gcn": {"median_ms": 12.5, "parameter_count_millions": 0.02},
                    "msas_gnn": {"median_ms": 2.1, "parameter_count_millions": 0.70},
                },
                "ogbn_arxiv": {
                    "gcn": {"median_ms_per_batch": 125.3, "peak_memory_mb": 2500, "parameter_count_millions": 0.02},
                    "msas_gnn": {"median_ms_per_batch": 9.2, "peak_memory_mb": 630, "parameter_count_millions": 0.70},
                    "sdgnn": {"median_ms_per_batch": 8.5, "peak_memory_mb": 589, "parameter_count_millions": 0.68},
                },
            },
            "summary": {
                "gcn": {"avg_speedup_vs_gcn": 1.0},
                "msas_gnn": {"avg_speedup_vs_gcn": 8.0},
                "sdgnn": {"avg_speedup_vs_gcn": 9.0},
            },
        },
    )
    _write_json(
        results_dir / "breakeven_analysis.json",
        {
            "cora": {
                "t_pre": 1080.0,
                "t_dense": 0.038,
                "t_sparse": 0.006,
                "Q_be": 33750,
            }
        },
    )

    _write_json(
        results_dir / "supplemental_sigma_proxy.json",
        {
            "citeseer": [
                {
                    "k": 20,
                    "sigma_proxy_mean": 1.21,
                    "engineering_ref_mean": 0.268,
                    "epsilon_approx_mean": 0.213,
                    "acc_mean": 0.718,
                    "acc_std": 0.012,
                }
            ],
            "ogbn_arxiv": [
                {
                    "k": 20,
                    "sigma_proxy_mean": 1.31,
                    "engineering_ref_mean": 0.318,
                    "epsilon_approx_mean": 0.247,
                    "acc_mean": 0.713,
                    "acc_std": 0.006,
                }
            ],
        },
    )

    outputs = module.build_selected_tables(
        results_dir,
        output_dir,
        {"main", "ablation", "efficiency", "supplemental"},
    )
    assert len(outputs) == 7

    main_table = (output_dir / "table_6_2_homophily_large.tex").read_text(encoding="utf-8")
    assert r"\label{tab:ch6-homophily-large}" in main_table
    assert r"\textbf{88.3$\pm$0.7}" in main_table
    assert r"\textbf{9.2}" in main_table

    ablation_table = (output_dir / "table_6_4_ablation.tex").read_text(encoding="utf-8")
    assert r"\label{tab:ch6-ablation}" in ablation_table
    assert r"\textbf{87.5}" in ablation_table
    assert r"\textbf{83.0}" in ablation_table
    assert r"\textbf{97.4}" in ablation_table

    efficiency_table = (output_dir / "table_6_6_efficiency.tex").read_text(encoding="utf-8")
    assert r"\label{tab:ch6-infer}" in efficiency_table
    assert r"\textbf{0.70}" in efficiency_table

    supplemental_sigma = (output_dir / "supplemental/table_supp_spectral.tex").read_text(encoding="utf-8")
    assert r"\label{tab:supp-spectral}" in supplemental_sigma
    assert "1.21" in supplemental_sigma
    assert "71.8$\\pm$1.2" in supplemental_sigma
