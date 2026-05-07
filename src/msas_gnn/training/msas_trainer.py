"""MSAS-GNN完整训练入口（调度三阶段流水线）。全流程第3→4→5→6章。"""
from copy import deepcopy
import logging
import time
import torch
logger = logging.getLogger(__name__)


def _spectral_cache_path(cfg, dataset, seed):
    noise_cfg = cfg.get("noise", {}) if isinstance(cfg.get("noise", {}), dict) else {}
    if not noise_cfg.get("enabled", False):
        return f"data/cache/spectral/{dataset}_metrics.pt"
    mode = str(noise_cfg.get("mode", "flip"))
    ratio = float(noise_cfg.get("ratio", 0.0))
    seed_offset = int(noise_cfg.get("seed_offset", 10000))
    ratio_token = f"{ratio:.3f}".replace(".", "p")
    return (
        f"data/cache/spectral/{dataset}"
        f"_noise-{mode}-{ratio_token}-so{seed_offset}_seed{seed}_metrics.pt"
    )

class MSASTrainer:
    def __init__(self, cfg, device="auto"):
        from msas_gnn.config import normalize_config_aliases, normalize_teacher_config
        self.cfg = normalize_config_aliases(normalize_teacher_config(deepcopy(cfg)))
        self.device = "cuda" if (device=="auto" and torch.cuda.is_available()) else ("cpu" if device=="auto" else device)
        logger.info(f"MSASTrainer: dataset={self.cfg.get('dataset')} device={self.device}")

    def run_single_seed(
        self,
        seed,
        return_embeddings: bool = False,
        return_artifacts: bool = False,
    ) -> dict:
        t0 = time.time(); torch.manual_seed(seed)
        self.cfg["seed"] = seed
        ds = self.cfg.get("dataset","cora")
        ablation_id = self.cfg.get("ablation_id", "b5")
        logger.info(f"[{ds}][seed={seed}] 开始训练...")
        stage_times = {}
        # 数据
        stage_t0 = time.time()
        from msas_gnn.training.data_utils import prepare_supervised_data
        data, nc, nf = prepare_supervised_data(self.cfg, seed, device=self.device)
        stage_times["data_prep"] = time.time() - stage_t0
        # 图指标
        bundle = None
        if ablation_id == "sdgnn_pure":
            stage_times["spectral_metrics"] = 0.0
        else:
            stage_t0 = time.time()
            from msas_gnn.spectral.metric_bundle import compute_metric_bundle
            cache = _spectral_cache_path(self.cfg, ds, seed)
            spectral_cfg = self.cfg.get("spectral", {})
            bundle = compute_metric_bundle(
                data.cpu(),
                K_eig=int(spectral_cfg.get("K_eig", 50)),
                cache_path=cache,
                force_recompute=bool(self.cfg.get("no_cache", False)),
            )
            if bundle.eigenvalues.numel() > 1:
                self.cfg["lambda_gap"] = float(bundle.eigenvalues[1].item())
            stage_times["spectral_metrics"] = time.time() - stage_t0
        # 三维参数
        stage_t0 = time.time()
        from msas_gnn.adaptive.joint_budget import build_adaptive_params
        self.cfg["num_classes"] = nc
        params = build_adaptive_params(bundle, self.cfg, data=data.cpu())
        stage_times["adaptive_params"] = time.time() - stage_t0
        logger.info("[%s][seed=%s] τ(i) range: [%.4e, %.4e]", ds, seed, params.tau.min().item(), params.tau.max().item())
        # 教师训练
        stage_t0 = time.time()
        from msas_gnn.training.teacher_trainer import TeacherTrainer
        h_star = TeacherTrainer(self.cfg, device=self.device).train_and_cache(data, seed)
        stage_times["teacher_cache"] = time.time() - stage_t0
        # 交替优化
        stage_t0 = time.time()
        from msas_gnn.training.alternating_opt import AlternatingOptimizer
        theta_fixed, phi_tilde = AlternatingOptimizer(self.cfg, self.device).run(h_star, data, params)
        stage_times["alternating_opt"] = time.time() - stage_t0
        # 评估
        stage_t0 = time.time()
        from msas_gnn.decomposition.inference import infer_h_hat
        from msas_gnn.evaluation.metrics import compute_accuracy, compute_epsilon_approx
        import torch.nn.functional as F
        h_hat = infer_h_hat(theta_fixed, phi_tilde.to(self.device))
        clf = torch.nn.Linear(h_star.shape[1], nc).to(self.device)
        opt_c = torch.optim.Adam(clf.parameters(), lr=0.01)
        h_det = h_hat.detach()
        for _ in range(100):
            clf.train(); opt_c.zero_grad()
            loss = F.cross_entropy(clf(h_det)[data.train_mask], data.y[data.train_mask])
            loss.backward(); opt_c.step()
        clf.eval()
        with torch.no_grad(): logits = clf(h_det)
        test_acc = compute_accuracy(logits, data.y, data.test_mask)
        val_acc = compute_accuracy(logits, data.y, data.val_mask)
        eps = compute_epsilon_approx(h_star.to(self.device), h_hat)
        stage_times["classifier_eval"] = time.time() - stage_t0
        from msas_gnn.evaluation.protocols import build_protocol_metadata
        lars_cfg = self.cfg.get("lars", {})
        solver_mode = str(lars_cfg.get("theta_solver_mode", lars_cfg.get("scheme", "residual_cascade")))
        hop_strategy = str(self.cfg.get("hop_dim", {}).get("strategy", "spectral_gap_reference"))
        result = {"test_acc":test_acc,"val_acc":val_acc,"epsilon_approx":eps,
                  "e_approx":eps,
                  "k_bar":theta_fixed.k_bar,"sparsity":theta_fixed.sparsity,"tau_mean":params.tau.mean().item(),
                  "tau_min":params.tau.min().item(),"tau_max":params.tau.max().item(),
                  "support_total": theta_fixed.support_total, "candidate_total": theta_fixed.candidate_total,
                  "pruning_rate": theta_fixed.sparsity, "candidate_pruning_rate": theta_fixed.sparsity,
                  "inference_time_ms": None,
                  "solver_mode": solver_mode, "theta_solver_mode": solver_mode,
                  "hop_budget_strategy": hop_strategy,
                  "elapsed":time.time()-t0,"seed":seed,"dataset":ds,"ablation_id":ablation_id,
                  "stage_times":stage_times, "protocols": build_protocol_metadata(self.cfg)}
        if return_embeddings:
            result["embeddings"] = h_hat.detach().cpu()
        if return_artifacts:
            result["theta_fixed"] = theta_fixed
            result["phi_tilde"] = phi_tilde.detach().cpu()
            result["data"] = data.cpu()
        logger.info(f"[{ds}][seed={seed}] acc={test_acc:.4f} ε={eps:.4f} k̄={theta_fixed.k_bar:.1f} 耗时={result['elapsed']:.1f}s")
        return result
