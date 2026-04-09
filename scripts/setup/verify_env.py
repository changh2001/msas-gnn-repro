#!/usr/bin/env python3
"""验证Python/PyTorch/PyG/OGB版本及GPU可用性。用法：python scripts/setup/verify_env.py"""
import sys

def ok(label, cond, msg=""):
    suffix = ": " + msg if msg else ""
    print("  {} {}{}".format("[OK]" if cond else "[FAIL]", label, suffix))
    return cond

def main():
    print("="*50 + "\nMSAS-GNN 环境验证\n" + "="*50)
    all_ok = True
    pv = sys.version_info
    all_ok &= ok("Python {}.{}.{}".format(pv.major, pv.minor, pv.micro), pv >= (3, 10))
    try:
        import torch
        ok("PyTorch {}".format(torch.__version__), True)
        cuda = torch.cuda.is_available()
        ok("CUDA 可用", cuda)
        if cuda:
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                ok("GPU {}: {} ({}GB)".format(i, p.name, p.total_memory // 1024**3), True)
    except ImportError:
        all_ok &= ok("PyTorch", False, "未安装")
    for pkg in ["torch_geometric", "ogb", "scipy", "sklearn", "optuna", "yaml", "networkx"]:
        try:
            __import__(pkg)
            ok(pkg, True)
        except ImportError:
            all_ok &= ok(pkg, False, "未安装")
    print("="*50)
    print("  所有检查通过！" if all_ok else "  存在未通过项，请安装缺失依赖")
    print("="*50)

if __name__ == "__main__":
    main()
