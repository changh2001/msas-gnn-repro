"""Wilcoxon符号秩检验（双侧，p<0.05）。对应论文§6.1。
预期p值：Cora=0.002, Citeseer=0.001, PubMed=0.048, ogbn-arxiv=0.009, Chameleon=0.003, Squirrel=0.011
"""
import logging
from scipy.stats import wilcoxon
logger = logging.getLogger(__name__)

def run_wilcoxon(scores_msas, scores_sdgnn, dataset, alpha=0.05):
    """双侧Wilcoxon检验，10个种子配对样本。"""
    assert len(scores_msas)==len(scores_sdgnn)
    stat, p = wilcoxon(scores_msas, scores_sdgnn, alternative="two-sided")
    result = {"dataset":dataset,"stat":float(stat),"p_value":float(p),"significant":p<alpha,"n_samples":len(scores_msas)}
    logger.info(f"[significance] {dataset}: p={p:.4f} {'[OK]' if p<alpha else '[NOT SIG]'}")
    return result
