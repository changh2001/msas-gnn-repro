"""基线模型注册表。"""
import inspect
import logging

logger = logging.getLogger(__name__)


def _resolve_baseline(name):
    name = name.lower()
    if name == "gcn":
        from msas_gnn.baselines.gcn import GCN

        return GCN
    if name == "sgc":
        from msas_gnn.baselines.sgc import SGC

        return SGC
    if name == "pprgo":
        from msas_gnn.baselines.pprgo import PPRGo

        return PPRGo
    if name == "glnn":
        from msas_gnn.baselines.glnn import GLNN

        return GLNN
    if name == "sage":
        from msas_gnn.baselines.sage import GraphSAGE

        return GraphSAGE
    if name == "geom_gcn":
        from msas_gnn.baselines.geom_gcn import GeomGCN

        return GeomGCN
    if name == "h2gcn":
        from msas_gnn.baselines.h2gcn import H2GCN

        return H2GCN
    if name == "graphsaint":
        from msas_gnn.baselines.graphsaint import GraphSAINT

        return GraphSAINT
    if name == "nodeformer":
        from msas_gnn.baselines.graph_transformers import NodeFormer

        return NodeFormer
    if name == "difformer":
        from msas_gnn.baselines.graph_transformers import DIFFormer

        return DIFFormer
    if name == "sgformer":
        from msas_gnn.baselines.graph_transformers import SGFormer

        return SGFormer
    if name == "nagphormer":
        from msas_gnn.baselines.graph_transformers import NAGphormer

        return NAGphormer
    if name in ("sdgnn", "sdgnn_compat", "sdgnn_pure", "msas_gnn_b5", "b5"):
        from msas_gnn.baselines.sdgnn import SDGNN

        return SDGNN
    raise ValueError(f"Unknown baseline: {name}")


def get_baseline(name, **kwargs):
    baseline_cls = _resolve_baseline(name)
    signature = inspect.signature(baseline_cls.__init__)
    accepts_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return baseline_cls(**kwargs)

    allowed = {
        param.name
        for param in signature.parameters.values()
        if param.name != "self"
        and param.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in allowed}
    ignored = sorted(set(kwargs) - set(filtered_kwargs))
    if ignored:
        logger.info(
            "[baseline_registry] 忽略 %s 不支持的参数: %s",
            baseline_cls.__name__,
            ", ".join(ignored),
        )
    return baseline_cls(**filtered_kwargs)
