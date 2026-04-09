"""模型权重保存与加载（含配置哈希防止混用）。"""
import hashlib, json, logging, os
import torch; logger = logging.getLogger(__name__)

def config_hash(cfg): return hashlib.sha256(json.dumps(cfg,sort_keys=True,default=str).encode()).hexdigest()[:8]

def save_checkpoint(model, cfg, path, extra=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict":model.state_dict() if hasattr(model,"state_dict") else model,
                "config_hash":config_hash(cfg),"extra":extra or {}}, path)
    logger.info(f"已保存：{path}")

def load_checkpoint(path, cfg, strict=True):
    p = torch.load(path, map_location="cpu")
    if strict and p.get("config_hash","") != config_hash(cfg):
        logger.warning("配置哈希不匹配！")
    return p
