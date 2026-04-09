"""通用基线训练与评估。"""
from __future__ import annotations

from copy import deepcopy
import logging
import time

import torch
import torch.nn.functional as F

from msas_gnn.evaluation.metrics import compute_accuracy
from msas_gnn.training.data_utils import prepare_supervised_data

logger = logging.getLogger(__name__)


class BaselineTrainer:
    def __init__(self, cfg, method_name, baseline_cfg=None, device="auto"):
        self.cfg = deepcopy(cfg)
        self.method_name = str(method_name).lower()
        self.method_cfg = deepcopy(baseline_cfg or {})
        self.device = (
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else ("cpu" if device == "auto" else device)
        )
        self.minibatch_threshold = int(
            self.cfg.get("alternating_opt", {}).get("node_threshold_for_minibatch", 50000)
        )

    def _merged_train_cfg(self):
        train_cfg = deepcopy(self.cfg.get("train", {}))
        for key in ("lr", "weight_decay", "epochs", "dropout", "early_stopping_patience", "eta_min", "batch_size"):
            if key in self.method_cfg:
                train_cfg[key] = self.method_cfg[key]
        return train_cfg

    def _build_model(self, num_features, num_classes):
        from msas_gnn.baselines.registry import get_baseline

        train_cfg = self._merged_train_cfg()
        return get_baseline(
            self.method_name,
            in_channels=num_features,
            hidden_channels=int(self.method_cfg.get("hidden_dim", 64)),
            out_channels=num_classes,
            dropout=float(train_cfg.get("dropout", 0.5)),
            num_layers=int(self.method_cfg.get("layers", 2)),
            K=int(self.method_cfg.get("K", 2)),
            alpha=float(self.method_cfg.get("alpha", 0.1)),
            structural_neighbors=int(self.method_cfg.get("structural_neighbors", 2)),
        ).to(self.device)

    def _use_minibatch(self, data, train_cfg):
        if self.cfg.get("dataset") == "ogbn_arxiv":
            return True
        return bool(int(getattr(data, "num_nodes", 0)) > self.minibatch_threshold and int(train_cfg.get("batch_size", 0)) > 0)

    def _evaluate_loader(self, model, loader):
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = model(batch.x, batch.edge_index)[: batch.batch_size]
                labels = batch.y[: batch.batch_size]
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                total += int(labels.numel())
        return float(correct / max(total, 1))

    def run_single_seed(self, seed, return_embeddings=False):
        t0 = time.time()
        torch.manual_seed(seed)
        train_cfg = self._merged_train_cfg()
        prep_device = "cpu" if self.cfg.get("dataset") == "ogbn_arxiv" else self.device
        data, num_classes, num_features = prepare_supervised_data(self.cfg, seed, device=prep_device)
        model = self._build_model(num_features, num_classes)
        opt = torch.optim.Adam(
            model.parameters(),
            lr=float(train_cfg.get("lr", 0.01)),
            weight_decay=float(train_cfg.get("weight_decay", 5e-4)),
        )
        max_epochs = int(train_cfg.get("epochs", 200))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=max_epochs,
            eta_min=float(train_cfg.get("eta_min", 1e-5)),
        )
        best_val = float("-inf")
        best_state = None
        patience = int(train_cfg.get("early_stopping_patience", 20))
        wait = 0
        use_minibatch = self._use_minibatch(data, train_cfg)

        if use_minibatch:
            from msas_gnn.data.batchers import get_neighbor_loader

            logger.info(
                "[baseline:%s][seed=%s] mini-batch 训练 batch_size=%s",
                self.method_name,
                seed,
                train_cfg.get("batch_size", 1024),
            )
            train_nodes = data.train_mask.nonzero(as_tuple=False).view(-1)
            val_nodes = data.val_mask.nonzero(as_tuple=False).view(-1)
            train_loader = get_neighbor_loader(
                data,
                train_nodes,
                batch_size=int(train_cfg.get("batch_size", 1024)),
                num_workers=0,
                shuffle=True,
            )
            val_loader = get_neighbor_loader(
                data,
                val_nodes,
                batch_size=int(train_cfg.get("batch_size", 1024)),
                num_workers=0,
                shuffle=False,
            )
            for ep in range(max_epochs):
                model.train()
                for batch in train_loader:
                    batch = batch.to(self.device)
                    opt.zero_grad()
                    logits = model(batch.x, batch.edge_index)[: batch.batch_size]
                    labels = batch.y[: batch.batch_size]
                    loss = F.cross_entropy(logits, labels)
                    loss.backward()
                    opt.step()
                scheduler.step()
                val_acc = self._evaluate_loader(model, val_loader)
                if val_acc > best_val:
                    best_val = val_acc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        logger.info("[baseline:%s] 早停ep=%s", self.method_name, ep)
                        break
            test_loader = get_neighbor_loader(
                data,
                data.test_mask.nonzero(as_tuple=False).view(-1),
                batch_size=int(train_cfg.get("batch_size", 1024)),
                num_workers=0,
                shuffle=False,
            )
            if best_state:
                model.load_state_dict(best_state)
            val_acc = self._evaluate_loader(model, val_loader)
            test_acc = self._evaluate_loader(model, test_loader)
            embeddings = None
        else:
            data = data.to(self.device)
            for ep in range(max_epochs):
                model.train()
                opt.zero_grad()
                logits = model(data.x, data.edge_index)
                loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
                loss.backward()
                opt.step()
                scheduler.step()

                model.eval()
                with torch.no_grad():
                    val_logits = model(data.x, data.edge_index)
                    val_acc = compute_accuracy(val_logits, data.y, data.val_mask)
                if val_acc > best_val:
                    best_val = val_acc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        logger.info("[baseline:%s] 早停ep=%s", self.method_name, ep)
                        break
            if best_state:
                model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index)
            val_acc = compute_accuracy(logits, data.y, data.val_mask)
            test_acc = compute_accuracy(logits, data.y, data.test_mask)
            embeddings = None
            if return_embeddings:
                with torch.no_grad():
                    if hasattr(model, "get_embedding"):
                        embeddings = model.get_embedding(data.x, data.edge_index).detach().cpu()
                    else:
                        embeddings = logits.detach().cpu()

        result = {
            "dataset": self.cfg.get("dataset", "unknown"),
            "method_id": self.method_name,
            "seed": seed,
            "test_acc": float(test_acc),
            "val_acc": float(val_acc),
            "elapsed": time.time() - t0,
        }
        if embeddings is not None:
            result["embeddings"] = embeddings
        logger.info(
            "[baseline:%s][%s][seed=%s] acc=%.4f 耗时=%.1fs",
            self.method_name,
            result["dataset"],
            seed,
            result["test_acc"],
            result["elapsed"],
        )
        return result
