#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NewBin 分箱质量基准评测模块 (evaluate.py)
========================================
模块定位与功能概述：
  本模块专为宏基因组分箱模型（NewBin）设计，提供严谨、生产级的质量评估体系。
  全面对齐国际宏基因组权威评估工具 AMBER / CAMI 评测准则与 MIMAGS 质量分级标准。

核心评测维度：
  1. 真实物种标签 (Ground Truth) 黄金标准评测：
     - 基于两分图最大重合原则对齐预测分箱 (Bins) 与真实参考基因组 (Genomes)。
  2. 长度加权 (Basepair-weighted) 核心生物学指标：
     - Completeness (完整度 / 召回率 Recall): 重合碱基占该真实基因组总碱基的比例；
     - Contamination (污染度 / 1 - Precision): 分箱中来自非目标基因组的外源杂质碱基比例；
     - Purity (纯度 / 查准率 Precision): 分箱中真正属于目标基因组的碱基比例；
     - F1-Score: 完整度与纯度的调和平均得分。
  3. MIMAGS 国际标准 MAGs 质量等级分布：
     - 🌟 高质量 (High-Quality, HQ): Completeness >= 90% 且 Contamination < 5%
     - 🔷 中质量 (Medium-Quality, MQ): Completeness >= 50% 且 Contamination < 10%
     - ⚪ 低质量 (Low-Quality, LQ): Completeness < 50% 且 Contamination < 10%
     - ⚠️ 高度污染 (Over-contaminated): Contamination >= 10%
  4. 核心创新卖点评估 —— 短序列与质粒拯救能力 (Short Contig Rescue Capacity)：
     - 生物学背景：短序列 (< 2000 bp) 因 k-mer 统计波动大、质粒 GC 组成异质性高，在传统分箱工具中极易丢失或错分；
     - 算法验证：专门衡量 NewBin 的“长度自适应门控 g(L)”与“双端 Reads 物理连通架桥 (PE Bridging)”对短片段的打捞召回率；
     - 关键指标：
       * 条数拯救率 (Rescue Rate Count): 正确归入主宿主 MAG 的短序列条数比例；
       * 碱基拯救率 (Rescue Rate BP): 正确归入主宿主 MAG 的短序列碱基数比例；
       * 长度分层透视 (Length Strata): 对比 <2000 bp、2000-5000 bp 与 >=5000 bp 三档区间的召回表现。
  5. 全局宏观聚类指标：
     - ARI (调整兰德系数): 度量全库聚类标签与物种真值的一致性；
     - NMI (标准化互信息): 信息论视角的聚类纯净度；
     - Binned BP Rate (有效分箱率): 成功分箱的总碱基覆盖比例。
  6. 多维度报表导出：
     - evaluation_summary.tsv: 全局核心指标与 HQ/MQ 汇总；
     - evaluation_details.tsv: 各基因组最佳匹配与短序列打捞明细；
     - evaluation_strata.tsv: 长度分层区间拯救效果对照表；
     - evaluation_report.txt: 终端美化格式化字符看板。
"""

import os
import sys
import argparse
import logging
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# 确保 Windows 控制台输出不发生 GBK 编码错误
if sys.platform == "win32":
    try:
        import io
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

logger = logging.getLogger("NewBinEvaluate")


def load_fasta_lengths(fasta_path):
    """
    从 FASTA 文件中流式解析每条 contig 的序列长度（bp）。

    采用纯文本流式单遍扫描，内存开销恒定为 O(1)。

    Parameters
    ----------
    fasta_path : str
        输入的 Contigs FASTA 文件路径。

    Returns
    -------
    dict
        每条 contig 的真实碱基长度映射字典 {contig_name: int_length}。
    """
    if not fasta_path or not os.path.exists(fasta_path):
        return {}
    lengths = {}
    current_name = None
    current_len = 0
    with open(fasta_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_name is not None:
                    lengths[current_name] = current_len
                current_name = line[1:].split()[0]
                current_len = 0
            else:
                current_len += len(line)
        if current_name is not None:
            lengths[current_name] = current_len
    return lengths


def load_clustering_results(binned_path):
    """
    加载模型输出的预测分箱聚类结果。

    支持双重输入格式自适应：
      1. 聚类标签表格文件 (CSV/TSV): 包含两列 (contig_name, bin_id)；
      2. 分箱 FASTA 目录: 包含各个由流水线生成的 bin.X.fa 或 bin_X.fasta 文件。

    Parameters
    ----------
    binned_path : str
        分箱结果表格文件路径或包含 FASTA 分箱文件的目录路径。

    Returns
    -------
    dict
        contig 到预测分箱 ID 的映射字典 {contig_name: bin_id}。
    """
    contig2bin = {}
    if os.path.isdir(binned_path):
        for fname in os.listdir(binned_path):
            if fname.endswith(('.fa', '.fasta', '.fna')):
                bin_id = os.path.splitext(fname)[0]
                bin_file = os.path.join(binned_path, fname)
                with open(bin_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if line.startswith('>'):
                            ctg = line[1:].strip().split()[0]
                            contig2bin[ctg] = bin_id
    elif os.path.isfile(binned_path):
        sep = '\t' if binned_path.endswith('.tsv') else ','
        df = pd.read_csv(binned_path, sep=sep)
        # 若有包含关键字的列，优先识别
        cols = list(df.columns)
        ctg_col = cols[0]
        bin_col = cols[1]
        for c in cols:
            c_lower = str(c).lower()
            if c_lower in ['contig', 'contig_id', 'ctg', 'ctg_id', 'sequence']:
                ctg_col = c
            elif c_lower in ['bin', 'bin_id', 'cluster', 'label']:
                bin_col = c
        for _, row in df.iterrows():
            contig2bin[str(row[ctg_col])] = str(row[bin_col])
    else:
        raise FileNotFoundError(f"无法找到分箱结果路径: {binned_path}")
    return contig2bin


def load_ground_truth(gt_path, contig_col=None, genome_col=None):
    """
    加载模拟或真实测序数据集的真实物种/菌株参考标签 (Ground Truth)。

    支持灵活的表格格式：
      - 自动识别 contig 列（如 ctg, ctg_id, contig, sequence 等）；
      - 自动识别 genome 列（如 clr, genome, tax, taxonomy, label 等）；
      - 支持用户通过参数显式指定目标列名。

    Parameters
    ----------
    gt_path : str
        包含真实物种标注的 CSV 或 TSV 文件路径。
    contig_col : str, optional
        显式指定 contig 名称列名。若为 None 则自动推断。
    genome_col : str, optional
        显式指定真实基因组 ID 列名。若为 None 则自动推断。

    Returns
    -------
    dict
        contig 到真实参考物种 ID 的映射字典 {contig_name: genome_id}。
    """
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"无法找到真值标签文件: {gt_path}")

    sep = '\t' if gt_path.endswith('.tsv') else ','
    df = pd.read_csv(gt_path, sep=sep)
    cols = list(df.columns)

    if contig_col is None:
        for c in cols:
            if str(c).lower() in ['contig', 'contig_id', 'ctg', 'ctg_id', 'sequence']:
                contig_col = c
                break
        if contig_col is None:
            contig_col = cols[0]

    if genome_col is None:
        for c in cols:
            if str(c).lower() in ['genome', 'genome_id', 'clr', 'tax', 'taxonomy', 'label']:
                genome_col = c
                break
        if genome_col is None:
            genome_col = cols[1] if len(cols) > 1 else cols[0]

    contig2genome = {}
    for _, row in df.iterrows():
        c_val = str(row[contig_col]).strip()
        g_val = str(row[genome_col]).strip()
        if c_val and g_val and g_val != 'nan':
            contig2genome[c_val] = g_val
    return contig2genome


def compute_benchmark_metrics(contig2bin, contig2genome, contig_lengths=None, short_threshold=2000, **kwargs):
    """
    计算基于 Ground Truth 的分箱基准评测核心生物学与聚类指标。

    评测准则：
      1. 长度加权：所有完整度、污染度与拯救率均以序列碱基长度（bp）为权重；
      2. 基因组为中心匹配 (Genome-Centric Matching)：为每个参考基因组寻找使其完整度最大的最佳预测分箱；
      3. 短序列与质粒拯救评估：针对长度 < short_threshold (默认 2000 bp) 的短片段，
         统计其是否被准确分入该基因组的最佳分箱中，评估抗噪与架桥能力。

    Parameters
    ----------
    contig2bin : dict
        预测分箱结果字典 {contig_name: bin_id}。
    contig2genome : dict
        真实基因组参考标签字典 {contig_name: genome_id}。
    contig_lengths : dict, optional
        序列真实碱基长度字典 {contig_name: length_bp}。
        若未提供，每条序列等权按 1 长度计算。
    short_threshold : int, optional
        短序列与质粒长度判定阈值，默认 2000 bp。

    Returns
    -------
    dict
        评测结果字典，包含三个核心 DataFrame / 字典：
        - "summary": 全局宏观统计字典 (ARI, NMI, 均值 Comp/Contam/F1, HQ/MQ 数量及短序列综合拯救率)；
        - "genome_details": DataFrame，记录每个参考基因组的 Best_Bin、Comp、Contam、F1、短序列归入明细及 MIMAGS 评级；
        - "strata_details": DataFrame，记录按序列长度分层 (<2k, 2k-5k, >=5k) 的拯救率与分箱率对照表。
    """
    if contig_lengths is None:
        contig_lengths = {}

    # 获取所有相关的 contigs
    all_gt_contigs = set(contig2genome.keys())
    all_binned_contigs = set(contig2bin.keys())
    eval_contigs = all_gt_contigs.union(all_binned_contigs)

    # 构造默认长度 (若缺失则默认为 1 或 1000)
    lengths = {c: contig_lengths.get(c, 1) for c in eval_contigs}

    # 统计各真实基因组的总长度
    genome_total_len = defaultdict(int)
    for c, g in contig2genome.items():
        genome_total_len[g] += lengths[c]

    # 统计各预测分箱的总长度
    bin_total_len = defaultdict(int)
    for c, b in contig2bin.items():
        bin_total_len[b] += lengths[c]

    # 统计真实基因组与分箱之间的重合碱基长度: overlap[genome][bin]
    overlap_len = defaultdict(lambda: defaultdict(int))
    for c in eval_contigs:
        g = contig2genome.get(c, None)
        b = contig2bin.get(c, None)
        if g is not None and b is not None:
            overlap_len[g][b] += lengths[c]

    all_genomes = sorted(list(genome_total_len.keys()))
    all_bins = sorted(list(bin_total_len.keys()))

    # Genome-Centric 最佳匹配评估
    # 为每一个真实基因组寻找最佳匹配的分箱 (优先按完整度最大，相同时按 F1 最大)
    genome_matches = []
    hq_count = 0
    mq_count = 0
    lq_count = 0
    over_contam_count = 0

    for g in all_genomes:
        g_len = genome_total_len[g]
        best_b = None
        best_comp = 0.0
        best_contam = 0.0
        best_f1 = 0.0
        best_purity = 0.0

        if g in overlap_len and len(overlap_len[g]) > 0:
            for b, ov_len in overlap_len[g].items():
                comp = ov_len / g_len if g_len > 0 else 0.0
                b_len = bin_total_len[b]
                purity = ov_len / b_len if b_len > 0 else 0.0
                contam = 1.0 - purity
                f1 = (2 * comp * purity) / (comp + purity) if (comp + purity) > 0 else 0.0

                # 判定更佳匹配
                if comp > best_comp or (abs(comp - best_comp) < 1e-6 and f1 > best_f1):
                    best_comp = comp
                    best_contam = contam
                    best_purity = purity
                    best_f1 = f1
                    best_b = b

        # 划分 MIMAGS 质量评级
        if best_comp >= 0.90 and best_contam < 0.05:
            tier = "HQ"
            hq_count += 1
        elif best_comp >= 0.50 and best_contam < 0.10:
            tier = "MQ"
            mq_count += 1
        elif best_contam >= 0.10:
            tier = "Over-Contaminated"
            over_contam_count += 1
        else:
            tier = "LQ"
            lq_count += 1

        genome_matches.append({
            "Genome": g,
            "Total_Length": g_len,
            "Best_Bin": best_b if best_b is not None else "Unbinned",
            "Completeness": best_comp,
            "Contamination": best_contam,
            "Purity": best_purity,
            "F1_Score": best_f1,
            "Tier": tier
        })

    df_matches = pd.DataFrame(genome_matches)

    # 建立最佳匹配快速字典: best_bin_dict[genome] = best_bin_id
    best_bin_dict = {
        row["Genome"]: row["Best_Bin"] for row in genome_matches if row["Best_Bin"] != "Unbinned"
    }

    # =========================================================================
    #  短序列与质粒拯救能力评估 (< short_threshold，默认 2000 bp)
    # =========================================================================
    short_threshold = kwargs.get("short_threshold", 2000)

    # 分层统计区间: [<2k, 2k-5k, >=5k]
    strata_bins = [
        ("< 2000 bp (短序列/质粒)", lambda l: l < short_threshold),
        ("2000 - 5000 bp (中等序列)", lambda l: short_threshold <= l < 5000),
        (">= 5000 bp (长骨架序列)", lambda l: l >= 5000)
    ]

    strata_results = []
    genome_short_stats = defaultdict(lambda: {"total_cnt": 0, "rescued_cnt": 0, "total_bp": 0, "rescued_bp": 0})

    for label, cond in strata_bins:
        tot_cnt = 0
        tot_bp = 0
        rescued_cnt = 0
        rescued_bp = 0
        misclassified_cnt = 0
        unbinned_cnt = 0

        for c in all_gt_contigs:
            c_len = lengths[c]
            if not cond(c_len):
                continue

            g = contig2genome[c]
            target_bin = best_bin_dict.get(g, None)
            pred_b = contig2bin.get(c, None)

            tot_cnt += 1
            tot_bp += c_len

            if cond == strata_bins[0][1]:  # 是短序列 (< short_threshold)
                genome_short_stats[g]["total_cnt"] += 1
                genome_short_stats[g]["total_bp"] += c_len

            if pred_b is None or pred_b == "Unbinned":
                unbinned_cnt += 1
            elif target_bin is not None and pred_b == target_bin:
                rescued_cnt += 1
                rescued_bp += c_len
                if cond == strata_bins[0][1]:
                    genome_short_stats[g]["rescued_cnt"] += 1
                    genome_short_stats[g]["rescued_bp"] += c_len
            else:
                misclassified_cnt += 1

        rec_rate_cnt = (rescued_cnt / tot_cnt) if tot_cnt > 0 else 0.0
        rec_rate_bp = (rescued_bp / tot_bp) if tot_bp > 0 else 0.0
        binned_rate_cnt = ((tot_cnt - unbinned_cnt) / tot_cnt) if tot_cnt > 0 else 0.0

        strata_results.append({
            "Length_Interval": label,
            "Total_Contigs": tot_cnt,
            "Total_BP": tot_bp,
            "Rescued_Contigs": rescued_cnt,
            "Rescued_BP": rescued_bp,
            "Rescue_Rate (Count)": rec_rate_cnt,
            "Rescue_Rate (BP)": rec_rate_bp,
            "Misclassified_Contigs": misclassified_cnt,
            "Unbinned_Contigs": unbinned_cnt,
            "Binned_Rate (Count)": binned_rate_cnt
        })

    df_strata = pd.DataFrame(strata_results)

    # 将各基因组的短序列拯救情况附加到 df_matches
    short_col_vals = []
    for _, row in df_matches.iterrows():
        g = row["Genome"]
        s = genome_short_stats[g]
        if s["total_cnt"] > 0:
            rate = (s["rescued_cnt"] / s["total_cnt"]) * 100
            short_col_vals.append(f"{s['rescued_cnt']}/{s['total_cnt']} ({rate:.1f}%)")
        else:
            short_col_vals.append("0/0 (N/A)")
    df_matches["Short_Rescue (<2kb)"] = short_col_vals

    # 全局宏观聚类统计
    # 提取共有 contig 集合用于计算 ARI 和 NMI
    common_contigs = list(all_gt_contigs.intersection(all_binned_contigs))
    if len(common_contigs) > 1:
        gt_labels = [contig2genome[c] for c in common_contigs]
        pred_labels = [contig2bin[c] for c in common_contigs]
        ari = adjusted_rand_score(gt_labels, pred_labels)
        nmi = normalized_mutual_info_score(gt_labels, pred_labels)
    else:
        ari = 0.0
        nmi = 0.0

    total_gt_bp = sum(genome_total_len.values())
    total_binned_bp = sum(lengths[c] for c in all_binned_contigs if c in lengths)
    binned_rate = (total_binned_bp / total_gt_bp) if total_gt_bp > 0 else 0.0

    mean_completeness = float(df_matches["Completeness"].mean()) if len(df_matches) > 0 else 0.0
    mean_contamination = float(df_matches["Contamination"].mean()) if len(df_matches) > 0 else 0.0
    mean_f1 = float(df_matches["F1_Score"].mean()) if len(df_matches) > 0 else 0.0

    # 提取短序列专属核心指标
    short_row = strata_results[0]

    summary = {
        "Total_Genomes": len(all_genomes),
        "Total_Bins": len(all_bins),
        "High_Quality_MAGs (HQ)": hq_count,
        "Medium_Quality_MAGs (MQ)": mq_count,
        "Low_Quality_MAGs (LQ)": lq_count,
        "Over_Contaminated": over_contam_count,
        "HQ_Ratio": (hq_count / len(all_genomes)) if len(all_genomes) > 0 else 0.0,
        "MQ_Ratio": (mq_count / len(all_genomes)) if len(all_genomes) > 0 else 0.0,
        "Mean_Completeness": mean_completeness,
        "Mean_Contamination": mean_contamination,
        "Mean_F1_Score": mean_f1,
        "ARI": float(ari),
        "NMI": float(nmi),
        "Binned_BP_Rate": float(binned_rate),
        # 短序列专属指标
        "Short_Threshold_BP": short_threshold,
        "Total_Short_Contigs": short_row["Total_Contigs"],
        "Rescued_Short_Contigs": short_row["Rescued_Contigs"],
        "Short_Rescue_Rate_Count": short_row["Rescue_Rate (Count)"],
        "Short_Rescue_Rate_BP": short_row["Rescue_Rate (BP)"],
        "Short_Binned_Rate_Count": short_row["Binned_Rate (Count)"],
    }

    return {
        "summary": summary,
        "genome_details": df_matches,
        "strata_details": df_strata
    }


def format_benchmark_report(results):
    """
    格式化生成清晰易读的控制台评测看板。
    """
    summary = results["summary"]
    details = results["genome_details"]
    strata = results.get("strata_details", None)

    lines = []
    lines.append("=" * 86)
    lines.append("            >>> NewBin 分箱质量基准评测报告 (Benchmark Report) <<<")
    lines.append("=" * 86)
    lines.append("【一、全局聚类指标 (Global Metrics)】")
    lines.append(f"  - 真实基因组数 (Total Genomes):   {summary['Total_Genomes']}")
    lines.append(f"  - 预测分箱数量 (Total Bins):      {summary['Total_Bins']}")
    lines.append(f"  - 碱基分箱率 (Binned BP Rate):    {summary['Binned_BP_Rate'] * 100:.2f}%")
    lines.append(f"  - 调整兰德系数 (ARI):             {summary['ARI']:.4f}")
    lines.append(f"  - 标准化互信息 (NMI):             {summary['NMI']:.4f}")
    lines.append(f"  - 宏平均完整度 (Mean Comp):       {summary['Mean_Completeness'] * 100:.2f}%")
    lines.append(f"  - 宏平均污染度 (Mean Contam):     {summary['Mean_Contamination'] * 100:.2f}%")
    lines.append(f"  - 宏平均 F1-Score:                {summary['Mean_F1_Score']:.4f}")
    lines.append("")
    lines.append("【二、MIMAGS 国际标准质量等级分布】")
    lines.append("-" * 86)
    lines.append(f"  {'质量等级 (Tier)':<26} {'标准界定 (Criteria)':<35} {'MAGs 数量':<15}")
    lines.append("-" * 86)
    hq_cnt = summary['High_Quality_MAGs (HQ)']
    mq_cnt = summary['Medium_Quality_MAGs (MQ)']
    lq_cnt = summary['Low_Quality_MAGs (LQ)']
    ov_cnt = summary['Over_Contaminated']
    tot = summary['Total_Genomes']
    lines.append(f"  🌟 {'高质量 (High-Quality, HQ)':<22} {'Comp >= 90% & Contam < 5%':<35} {hq_cnt:<4} ({(hq_cnt/tot*100) if tot else 0:.1f}%)")
    lines.append(f"  🔷 {'中质量 (Medium-Quality, MQ)':<21} {'Comp >= 50% & Contam < 10%':<35} {mq_cnt:<4} ({(mq_cnt/tot*100) if tot else 0:.1f}%)")
    lines.append(f"  ⚪ {'低质量 (Low-Quality, LQ)':<23} {'Comp < 50% & Contam < 10%':<35} {lq_cnt:<4} ({(lq_cnt/tot*100) if tot else 0:.1f}%)")
    lines.append(f"  ⚠️ {'高度污染 (Over-Contaminated)':<20} {'Contamination >= 10%':<35} {ov_cnt:<4} ({(ov_cnt/tot*100) if tot else 0:.1f}%)")
    lines.append("-" * 86)
    lines.append("")
    lines.append("【三、各基因组最佳匹配详情 (Genome-Centric Matches)】")
    lines.append("-" * 86)
    lines.append(f"  {'真实基因组':<12} {'最佳分箱':<10} {'完整度 (Comp)':<14} {'污染度 (Contam)':<15} {'F1-Score':<10} {'短序列拯救 (<2kb)':<16} {'评级':<6}")
    lines.append("-" * 86)
    for _, r in details.iterrows():
        comp_str = f"{r['Completeness'] * 100:.2f}%"
        contam_str = f"{r['Contamination'] * 100:.2f}%"
        f1_str = f"{r['F1_Score']:.4f}"
        short_rec = r.get("Short_Rescue (<2kb)", "N/A")
        tier_tag = f"[{r['Tier']}]"
        lines.append(f"  {r['Genome']:<12} {r['Best_Bin']:<10} {comp_str:<14} {contam_str:<15} {f1_str:<10} {short_rec:<16} {tier_tag:<6}")
    lines.append("-" * 86)

    if strata is not None and len(strata) > 0:
        lines.append("")
        lines.append("【四、短序列与质粒拯救能力评估 (Short Contig Rescue Capacity)】")
        lines.append("  (长度门控 g(L) 与 PE Reads 物理连通对短小元件/质粒的归入召回效果)")
        lines.append("-" * 86)
        lines.append(f"  {'序列长度区间 (Length Strata)':<28} {'总条数':<8} {'成功归入':<10} {'拯救率(条数)':<13} {'拯救率(碱基BP)':<13} {'分箱率':<8}")
        lines.append("-" * 86)
        for _, sr in strata.iterrows():
            tot_c = sr["Total_Contigs"]
            res_c = sr["Rescued_Contigs"]
            rc_rate = f"{sr['Rescue_Rate (Count)'] * 100:.1f}%"
            rb_rate = f"{sr['Rescue_Rate (BP)'] * 100:.1f}%"
            bin_rate = f"{sr['Binned_Rate (Count)'] * 100:.1f}%"
            lines.append(f"  {sr['Length_Interval']:<26} {tot_c:<8} {res_c:<10} {rc_rate:<13} {rb_rate:<13} {bin_rate:<8}")
        lines.append("-" * 86)

    lines.append("=" * 86)
    return "\n".join(lines)


def run_benchmark(binned_path, gt_path, fasta_path=None, contig_col=None, genome_col=None,
                  short_threshold=2000, output_dir=None):
    """
    执行完整的 Ground Truth 基准评测端到端流程，并打印看板与持久化报表。

    流水线逻辑：
      1. 加载预测分箱 (支持 CSV/TSV 表格或 FASTA 文件夹)；
      2. 加载参考真值物种/菌株标签 (支持自定义列名或自动推断)；
      3. 流式读取 Contigs FASTA 计算每条序列真实碱基长度 (支持缺省降级)；
      4. 计算长度加权 Completeness, Contamination, F1, HQ/MQ 分级与短序列拯救率；
      5. 格式化输出控制台评测看板；
      6. (可选) 导出 evaluation_summary.tsv, evaluation_details.tsv, 
         evaluation_strata.tsv 及 evaluation_report.txt。

    Parameters
    ----------
    binned_path : str
        预测分箱结果文件路径或包含 *.fa 分箱的目录。
    gt_path : str
        真实参考标签 CSV/TSV 文件路径。
    fasta_path : str, optional
        原始 Contigs FASTA 路径，用于精确碱基长度加权。
    contig_col : str, optional
        真值表中 contig 列名。
    genome_col : str, optional
        真值表中 genome 列名。
    short_threshold : int, optional
        短序列判定阈值，默认 2000 bp。
    output_dir : str, optional
        评估报表导出目录路径。若为 None 则不落盘。

    Returns
    -------
    dict
        包含 summary、genome_details、strata_details 的综合结果字典。
    """
    logger.info("正在加载分箱预测结果: %s", binned_path)
    contig2bin = load_clustering_results(binned_path)

    logger.info("正在加载真值标签: %s", gt_path)
    contig2genome = load_ground_truth(gt_path, contig_col=contig_col, genome_col=genome_col)

    contig_lengths = {}
    if fasta_path and os.path.exists(fasta_path):
        logger.info("正在读取 FASTA 计算真实碱基长度: %s", fasta_path)
        contig_lengths = load_fasta_lengths(fasta_path)
    else:
        logger.info("未提供 FASTA 文件，将按 contig 序列条数等权计算指标。")

    results = compute_benchmark_metrics(
        contig2bin, contig2genome, contig_lengths=contig_lengths, short_threshold=short_threshold
    )
    report_text = format_benchmark_report(results)
    print(report_text)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        summary_tsv = os.path.join(output_dir, "evaluation_summary.tsv")
        details_tsv = os.path.join(output_dir, "evaluation_details.tsv")
        strata_tsv = os.path.join(output_dir, "evaluation_strata.tsv")
        report_txt = os.path.join(output_dir, "evaluation_report.txt")

        pd.DataFrame([results["summary"]]).to_csv(summary_tsv, sep='\t', index=False)
        results["genome_details"].to_csv(details_tsv, sep='\t', index=False)
        if "strata_details" in results:
            results["strata_details"].to_csv(strata_tsv, sep='\t', index=False)
        with open(report_txt, 'w', encoding='utf-8') as f:
            f.write(report_text)
        logger.info("评测报告已成功导出至目录: %s", output_dir)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="NewBin 分箱质量基准评测工具 (Benchmark Evaluation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-b", "--binned", required=True,
                        help="分箱结果文件 (CSV/TSV，包含 contig, bin) 或存放各个 *.fa 分箱的目录")
    parser.add_argument("-g", "--ground-truth", required=True,
                        help="真实物种/菌株标签文件 (CSV/TSV)")
    parser.add_argument("-f", "--fasta", default=None,
                        help="原始 Contigs FASTA 文件路径 (用于按碱基长度加权，可选)")
    parser.add_argument("--contig-col", default=None,
                        help="真值表中 contig 名称列名 (默认自动识别 ctg, contig_id 等)")
    parser.add_argument("--genome-col", default=None,
                        help="真值表中真实基因组名称列名 (默认自动识别 clr, genome, tax 等)")
    parser.add_argument("--short-threshold", type=int, default=2000,
                        help="短序列与质粒长度判定阈值 (默认 2000 bp)")
    parser.add_argument("-o", "--output", default="eval_output",
                        help="评估报告输出目录")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    run_benchmark(
        binned_path=args.binned,
        gt_path=args.ground_truth,
        fasta_path=args.fasta,
        contig_col=args.contig_col,
        genome_col=args.genome_col,
        short_threshold=args.short_threshold,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()

