"""Phase-Θ/Phase-W交替优化主循环。对应论文§5.1.2算法5.4 与第6章大图训练口径。

非凸问题声明：交替优化为非凸问题，不主张严格全局收敛保证。
观测到总体损失呈下降趋势（经验稳定性），但非严格单调递减。
协议：b5_full：热启动+Phase-Θ+Phase-W+Φ̃重算；b5_frozen：跳过Phase-W；
      sdgnn_orig：原始 SDGNN 风格的平坦候选池 + 单次 LARS/Lasso Phase-Θ。

第6章的大图工程口径在本实现中以“显式 W_phi + 按需重算 Φ̃”落地：
- 小图保持 full-batch；
- 大图按节点 mini-batch 交替执行 Phase-Θ / Phase-W；
- 训练结束后固定最终 Φ̃，再对全图补做一次 Phase-Θ 重对齐。
"""
import logging
import torch, torch.nn.functional as F
from msas_gnn.decomposition.warm_start import warm_start_phi_tilde
from msas_gnn.decomposition.inference import infer_h_hat
from msas_gnn.training.feature_transform import (
    compute_phi_tilde,
    initialize_linear_feature_transform,
)
logger = logging.getLogger(__name__)

class AlternatingOptimizer:
    def __init__(self, cfg, device="cpu"):
        self.cfg = cfg; self.device = device
        ac = cfg.get("alternating_opt",{})
        self.max_iter = ac.get("max_outer_iter",20)
        self.patience = ac.get("patience",5)
        self.protocol = ac.get("protocol","b5_full")
        self.phase_w_steps = ac.get("t_w", 5)
        self.phase_w_lr = ac.get("eta_w", 5e-3)
        self.batch_mode = ac.get("batch_mode", "auto")
        self.minibatch_threshold = int(ac.get("node_threshold_for_minibatch", 50000))
        train_cfg = cfg.get("train", {})
        self.batch_size = int(train_cfg.get("batch_size", cfg.get("batch_size", 1024)))
        self.seed = int(cfg.get("seed", 42))

    def run(self, h_star, data, params):
        h_star = h_star.to(self.device)
        x = data.x.to(self.device)
        feature_cfg = self.cfg.get("feature_transform", {})
        feature_transform = initialize_linear_feature_transform(
            x=x,
            h_star=h_star,
            ridge=float(feature_cfg.get("ridge", 1e-4)),
            device=self.device,
        )
        phi_tilde = warm_start_phi_tilde(h_star)
        logger.info(f"热启动完成 协议={self.protocol}")
        data_cpu = data.cpu() if hasattr(data, "cpu") else data
        if self.protocol == "sdgnn_orig":
            from msas_gnn.decomposition.candidate_builder import build_sdgnn_candidate_set

            pure_cfg = self.cfg.get("sdgnn_pure", {})
            cands = build_sdgnn_candidate_set(
                data_cpu,
                base_hops=int(pure_cfg.get("base_hops", 2)),
                extra_hops=int(pure_cfg.get("extra_hops", 0)),
                fanouts=pure_cfg.get("fanouts", []),
                seed=self.seed,
            )
            theta_fixed, phi_tilde = self._run_sdgnn_orig(
                h_star,
                x,
                phi_tilde,
                feature_transform,
                params,
                cands,
            )
            if theta_fixed is None:
                raise RuntimeError("交替优化未执行")
            return theta_fixed, phi_tilde
        from msas_gnn.decomposition.candidate_builder import build_bfs_candidate_sets
        L = params.k_budget.shape[1]
        candidate_cfg = self.cfg.get("candidate_sampling", {})
        cands = build_bfs_candidate_sets(
            data_cpu,
            L=L,
            max_candidates=self.cfg.get("lars", {}).get("max_candidates"),
            seed=self.seed,
            keep_complete_hops=candidate_cfg.get("keep_complete_hops"),
            sampled_max_candidates=candidate_cfg.get("sampled_max_candidates"),
        )
        batch_mode = self._resolve_batch_mode(getattr(data, "num_nodes", h_star.shape[0]))
        logger.info("交替优化批模式=%s batch_size=%s", batch_mode, self.batch_size)
        if batch_mode == "mini_batch":
            theta_fixed, phi_tilde = self._run_minibatch(
                h_star,
                x,
                phi_tilde,
                feature_transform,
                params,
                cands,
            )
        else:
            theta_fixed, phi_tilde = self._run_fullbatch(
                h_star,
                x,
                phi_tilde,
                feature_transform,
                params,
                cands,
            )
        if theta_fixed is None: raise RuntimeError("交替优化未执行")
        return theta_fixed, phi_tilde

    def _run_sdgnn_orig(self, h_star, x, phi_tilde, feature_transform, params, candidate_set):
        best_loss = float("inf")
        pat = 0
        theta_fixed = None
        h_star_cpu = h_star.cpu()
        from msas_gnn.decomposition.theta_optimizer import run_phase_theta_sdgnn

        for it in range(self.max_iter):
            theta_fixed = run_phase_theta_sdgnn(h_star_cpu, phi_tilde.cpu(), params, candidate_set, self.cfg)
            h_hat = infer_h_hat(theta_fixed, phi_tilde.cpu())
            loss = F.mse_loss(h_hat, h_star_cpu).item()
            logger.info("外迭代%s[sdgnn_orig]: loss=%.6f k̄=%.1f", it + 1, loss, theta_fixed.k_bar)
            if loss < best_loss * 0.999:
                best_loss = loss
                pat = 0
            else:
                pat += 1
                if pat >= self.patience:
                    logger.info("sdgnn_orig 外循环早停 iter=%s", it + 1)
                    break
            feature_transform, phi_tilde = self._phase_w(
                theta_fixed,
                feature_transform,
                x,
                h_star,
                n_steps=self.phase_w_steps,
                lr=self.phase_w_lr,
            )
        return theta_fixed, phi_tilde

    def _resolve_batch_mode(self, num_nodes):
        if self.protocol != "b5_full":
            return "full_batch"
        if self.batch_mode == "mini_batch":
            return "mini_batch"
        if self.batch_mode == "full_batch":
            return "full_batch"
        if self.batch_mode != "auto":
            logger.warning("未知 batch_mode=%s，回退为 auto", self.batch_mode)
        return "mini_batch" if int(num_nodes) > self.minibatch_threshold else "full_batch"

    def _run_fullbatch(self, h_star, x, phi_tilde, feature_transform, params, candidate_sets):
        best_loss = float("inf")
        pat = 0
        theta_fixed = None
        h_star_cpu = h_star.cpu()
        from msas_gnn.decomposition.theta_optimizer import run_phase_theta

        for it in range(self.max_iter):
            theta_fixed = run_phase_theta(h_star_cpu, phi_tilde.cpu(), params, candidate_sets, self.cfg)
            h_hat = infer_h_hat(theta_fixed, phi_tilde.cpu())
            loss = F.mse_loss(h_hat, h_star_cpu).item()
            logger.info(f"外迭代{it+1}: loss={loss:.6f} k̄={theta_fixed.k_bar:.1f}")
            if loss < best_loss * 0.999:
                best_loss = loss
                pat = 0
            else:
                pat += 1
                if pat >= self.patience:
                    logger.info(f"外循环早停 iter={it+1}")
                    break
            if self.protocol == "b5_full":
                feature_transform, phi_tilde = self._phase_w(
                    theta_fixed,
                    feature_transform,
                    x,
                    h_star,
                    n_steps=self.phase_w_steps,
                    lr=self.phase_w_lr,
                )
            else:
                break
        return theta_fixed, phi_tilde

    def _run_minibatch(self, h_star, x, phi_tilde, feature_transform, params, candidate_sets):
        best_loss = float("inf")
        pat = 0
        theta_fixed = None
        h_star_cpu = h_star.cpu()
        from msas_gnn.decomposition.theta_optimizer import run_phase_theta

        for it in range(self.max_iter):
            epoch_losses = []
            epoch_kbars = []
            for batch_nodes in self._iter_node_batches(h_star_cpu.shape[0], epoch=it):
                theta_batch = run_phase_theta(
                    h_star_cpu,
                    phi_tilde.cpu(),
                    params,
                    candidate_sets,
                    self.cfg,
                    node_indices=batch_nodes,
                )
                feature_transform, phi_tilde = self._phase_w(
                    theta_batch,
                    feature_transform,
                    x,
                    h_star,
                    n_steps=self.phase_w_steps,
                    lr=self.phase_w_lr,
                    batch_indices=batch_nodes,
                )
                h_batch = infer_h_hat(theta_batch, phi_tilde.cpu())[batch_nodes]
                batch_loss = F.mse_loss(h_batch, h_star_cpu[batch_nodes]).item()
                epoch_losses.append(batch_loss)
                epoch_kbars.append(theta_batch.k_bar)

            if not epoch_losses:
                raise RuntimeError("mini-batch 交替优化未生成任何批次")
            loss = float(sum(epoch_losses) / len(epoch_losses))
            mean_kbar = float(sum(epoch_kbars) / len(epoch_kbars))
            logger.info(
                "外迭代%s[minibatch]: mean_loss=%.6f mean_k̄=%.1f batches=%s",
                it + 1,
                loss,
                mean_kbar,
                len(epoch_losses),
            )
            if loss < best_loss * 0.999:
                best_loss = loss
                pat = 0
            else:
                pat += 1
                if pat >= self.patience:
                    logger.info("mini-batch 外循环早停 iter=%s", it + 1)
                    break

        logger.info("第6章大图训练口径：固定最终Φ̃后补做一次全图 Phase-Θ 重对齐")
        theta_fixed = run_phase_theta(h_star_cpu, phi_tilde.cpu(), params, candidate_sets, self.cfg)
        return theta_fixed, phi_tilde

    def _iter_node_batches(self, num_nodes, epoch=0):
        generator = torch.Generator()
        generator.manual_seed(self.seed + int(epoch))
        order = torch.randperm(num_nodes, generator=generator).tolist()
        for start in range(0, num_nodes, self.batch_size):
            yield order[start:start + self.batch_size]

    def _phase_w(self, theta_fixed, feature_transform, x, h_star, n_steps=5, lr=5e-3, batch_indices=None):
        """Phase-W：固定Θ^fixed，优化 W_phi，并由 phi(X; W_phi) 重算 Φ̃。"""
        feature_transform = feature_transform.to(self.device)
        feature_transform.train()
        opt = torch.optim.Adam(feature_transform.parameters(), lr=lr)
        theta = theta_fixed.theta
        if theta.device != self.device:
            theta = theta.to(self.device)
        if x.device != self.device:
            x = x.to(self.device)
        if h_star.device != self.device:
            h_star = h_star.to(self.device)
        target_rows = None
        if batch_indices is not None:
            target_rows = torch.tensor(batch_indices, dtype=torch.long, device=self.device)
        for _ in range(n_steps):
            opt.zero_grad()
            phi = compute_phi_tilde(feature_transform, x)
            h = torch.sparse.mm(theta.t(), phi) if theta.is_sparse else theta.t() @ phi
            if target_rows is None:
                loss = F.mse_loss(h, h_star)
            else:
                loss = F.mse_loss(h.index_select(0, target_rows), h_star.index_select(0, target_rows))
            loss.backward()
            opt.step()
        feature_transform.eval()
        with torch.no_grad():
            phi_tilde = compute_phi_tilde(feature_transform, x).detach()
        return feature_transform, phi_tilde
