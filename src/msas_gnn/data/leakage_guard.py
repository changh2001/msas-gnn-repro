"""数据划分泄漏检测。

附录 C.2 关注的核心不是“特征值本身是否可见”，而是训练/验证/测试节点
是否发生集合重叠，导致标签监督或评估口径被污染。此处统一做 split leakage
检查，并保留旧函数名以兼容既有脚本入口。
"""
import logging

logger = logging.getLogger(__name__)


def check_feature_leakage(x, train_mask, test_mask, name="", val_mask=None):
    """检查 train/val/test 掩码是否互斥，并验证长度与节点数一致。"""
    num_nodes = int(getattr(x, "shape", [len(train_mask)])[0])
    masks = {
        "train": train_mask,
        "test": test_mask,
    }
    if val_mask is not None:
        masks["val"] = val_mask

    for split_name, mask in masks.items():
        if mask.numel() != num_nodes:
            logger.error(
                "[leakage_guard] %s: %s_mask长度=%s 与节点数=%s 不一致",
                name,
                split_name,
                mask.numel(),
                num_nodes,
            )
            return False

    overlaps = [
        ("train", "test", int((train_mask & test_mask).sum().item())),
    ]
    if val_mask is not None:
        overlaps.extend(
            [
                ("train", "val", int((train_mask & val_mask).sum().item())),
                ("val", "test", int((val_mask & test_mask).sum().item())),
            ]
        )

    bad_pairs = [(a, b, c) for a, b, c in overlaps if c > 0]
    if bad_pairs:
        detail = ", ".join(f"{a}/{b}重叠{count}" for a, b, count in bad_pairs)
        logger.error("[leakage_guard] %s: 检测到划分泄漏: %s", name, detail)
        return False

    logger.info("[leakage_guard] %s: [OK] split masks are mutually exclusive.", name)
    return True
