"""教师GNN训练与H*缓存。对应论文§5.1（教师表示H*获取）。"""
import logging, os
import torch, torch.nn.functional as F
logger = logging.getLogger(__name__)


def _noise_cache_suffix(cfg) -> str:
    noise_cfg = cfg.get("noise", {}) if isinstance(cfg.get("noise", {}), dict) else {}
    if not noise_cfg.get("enabled", False):
        return ""
    mode = str(noise_cfg.get("mode", "flip"))
    ratio = float(noise_cfg.get("ratio", 0.0))
    seed_offset = int(noise_cfg.get("seed_offset", 10000))
    ratio_token = f"{ratio:.3f}".replace(".", "p")
    return f"_noise-{mode}-{ratio_token}-so{seed_offset}"

class TeacherTrainer:
    def __init__(self, cfg, device="auto"):
        self.cfg = cfg
        self.device = "cuda" if (device=="auto" and torch.cuda.is_available()) else ("cpu" if device=="auto" else device)
        self.teacher_cfg = cfg.get("teacher",{})
        tc = self.teacher_cfg if isinstance(self.teacher_cfg, dict) else {}
        self.teacher_name = tc.get("name", cfg.get("teacher_name", "gcn"))
        self.hidden = tc.get("hidden_dim",64)
        self.layers = tc.get("layers",2)

    def train_and_cache(self, data, seed, cache_dir="data/cache/teachers"):
        ds = self.cfg.get("dataset","unknown")
        noise_suffix = _noise_cache_suffix(self.cfg)
        path = os.path.join(cache_dir, f"{ds}_{self.teacher_name}{noise_suffix}_seed{seed}_H_star.pt")
        if os.path.exists(path) and not self.cfg.get("no_cache",False):
            logger.info(f"加载H*缓存：{path}"); return torch.load(path, map_location="cpu")
        torch.manual_seed(seed)
        data = data.to(self.device)
        nc = self.cfg.get("num_classes", int(data.y.max().item())+1)
        nf = data.num_node_features
        from msas_gnn.baselines.registry import get_baseline
        if isinstance(self.teacher_cfg, dict):
            tr = dict(self.teacher_cfg)
        else:
            tr = {}
        tr.update(self.cfg.get("train",{}))
        max_epochs = tr.get("epochs", 200)
        model = get_baseline(
            self.teacher_name,
            in_channels=nf,
            hidden_channels=self.hidden,
            out_channels=nc,
            dropout=tr.get("dropout",0.5),
            num_layers=self.layers,
            K=tr.get("K", 2),
            alpha=tr.get("alpha", 0.1),
            structural_neighbors=tr.get("structural_neighbors", 2),
        ).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=tr.get("lr",0.01), weight_decay=tr.get("weight_decay",5e-4))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=max_epochs,
            eta_min=tr.get("eta_min", 1e-5),
        )
        best_val, best_state, pat = 0.0, None, 0
        for ep in range(max_epochs):
            model.train(); opt.zero_grad()
            loss = F.cross_entropy(model(data.x,data.edge_index)[data.train_mask], data.y[data.train_mask])
            loss.backward(); opt.step(); scheduler.step()
            model.eval()
            with torch.no_grad():
                pred = model(data.x,data.edge_index)[data.val_mask].argmax(1)
                val_acc = (pred==data.y[data.val_mask]).float().mean().item()
            if val_acc > best_val:
                best_val=val_acc; best_state={k:v.clone() for k,v in model.state_dict().items()}; pat=0
            else:
                pat+=1
                if pat >= tr.get("early_stopping_patience",20): logger.info(f"早停ep={ep}"); break
        if best_state: model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            if hasattr(model, "get_embedding"):
                h = model.get_embedding(data.x, data.edge_index)
            else:
                logger.warning("教师模型未实现get_embedding()，回退为logits作为H*")
                h = model(data.x, data.edge_index)
        h_cpu = h.cpu(); os.makedirs(cache_dir, exist_ok=True); torch.save(h_cpu, path)
        logger.info(f"H*已缓存 {path} shape={h_cpu.shape}"); return h_cpu
