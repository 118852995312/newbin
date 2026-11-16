#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据统计与概览分析脚本 (适用于 contig2bin 联调数据)
功能：
1. coverage 统计：Contig 数量、长度分布 (Min, Max, Mean, Median, N50)、深度分布 (Depth)
2. ctg2genome 统计：真实 Genome (Bin) 数量、基因组大小及 Contig 分布
3. links 统计：样本内 (Intra) 与样本间 (Inter) 连接数、参与节点数、连接强度分布
"""

import os
import glob
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def calc_n50(lengths: np.ndarray) -> int:
    """计算 N50 指标"""
    if len(lengths) == 0:
        return 0
    sorted_len = np.sort(lengths)[::-1]
    csum = np.cumsum(sorted_len)
    half_total = sorted_len.sum() / 2.0
    return int(sorted_len[csum >= half_total][0])


def analyze_coverage(coverage_dir: Path):
    print("\n" + "=" * 80)
    print("【1. Coverage 深度与 Contig 序列长度统计】")
    print("=" * 80)
    
    files = sorted(coverage_dir.glob("*.coverage"))
    if not files:
        print(f"未在 {coverage_dir} 找到 .coverage 文件")
        return None

    records = []
    total_contigs = 0
    total_bp = 0
    all_lengths = []

    for f in files:
        sample_name = f.stem
        # 仅读取需要的列以大幅节省内存和时间
        df = pd.read_csv(f, sep="\t", usecols=["contigName", "contigLen", "totalAvgDepth"])
        lens = df["contigLen"].values
        depths = df["totalAvgDepth"].values

        n50 = calc_n50(lens)
        total_contigs += len(lens)
        sample_bp = lens.sum()
        total_bp += sample_bp
        all_lengths.extend(lens)

        records.append({
            "Sample": sample_name,
            "Contigs": len(lens),
            "Total_Mb": round(sample_bp / 1e6, 2),
            "Min_Len": int(lens.min()),
            "Median_Len": int(np.median(lens)),
            "Mean_Len": round(float(lens.mean()), 1),
            "N50": n50,
            "Max_Len": int(lens.max()),
            "Mean_Depth": round(float(depths.mean()), 2),
            "Median_Depth": round(float(np.median(depths)), 2),
            "Max_Depth": round(float(depths.max()), 1),
        })

    summary_df = pd.DataFrame(records)
    print(summary_df.to_string(index=False))

    all_lens_arr = np.array(all_lengths)
    overall_n50 = calc_n50(all_lens_arr)
    print("-" * 80)
    print(f"总样本数: {len(files)}")
    print(f"Contig 总数: {total_contigs:,}")
    print(f"总碱基数: {total_bp / 1e6:,.2f} Mb (~{total_bp / 1e9:.2f} Gb)")
    print(f"全局 Contig 长度: Min={all_lens_arr.min()}, Median={int(np.median(all_lens_arr))}, "
          f"Mean={all_lens_arr.mean():.1f}, N50={overall_n50}, Max={all_lens_arr.max():,}")
    return summary_df


def analyze_ctg2genome(ctg2genome_dir: Path):
    print("\n" + "=" * 80)
    print("【2. ctg2genome 真实基因组 (Ground Truth Bins) 统计】")
    print("=" * 80)

    files = sorted(ctg2genome_dir.glob("*.tsv"))
    if not files:
        print(f"未在 {ctg2genome_dir} 找到 .tsv 文件")
        return None

    records = []
    all_unique_genomes = set()

    for f in files:
        sample_name = f.name.replace(".ctg2genome.tsv", "")
        # 跳过 ctg_sequence 节省内存
        df = pd.read_csv(f, sep="\t", usecols=["ctg_id", "genome_id"])
        genomes = df["genome_id"].unique()
        all_unique_genomes.update(genomes)
        ctg_per_genome = df["genome_id"].value_counts()

        records.append({
            "Sample": sample_name,
            "Contigs": len(df),
            "Unique_Genomes": len(genomes),
            "Ctg/Gen_Min": ctg_per_genome.min(),
            "Ctg/Gen_Median": int(ctg_per_genome.median()),
            "Ctg/Gen_Mean": round(float(ctg_per_genome.mean()), 1),
            "Ctg/Gen_Max": ctg_per_genome.max(),
        })

    summary_df = pd.DataFrame(records)
    print(summary_df.to_string(index=False))
    print("-" * 80)
    print(f"全数据集去重后的真实 Genome 种类数: {len(all_unique_genomes)}")
    return summary_df


def analyze_links(links_dir: Path):
    print("\n" + "=" * 80)
    print("【3. links 连接图 (Contig Links / Graph) 统计】")
    print("=" * 80)

    files = sorted(links_dir.glob("*.tsv"))
    if not files:
        print(f"未在 {links_dir} 找到 .tsv 文件")
        return None

    total_links = 0
    intra_links = 0
    inter_links = 0

    intra_files = 0
    inter_files = 0

    sample_link_counts = {}

    for f in files:
        parts = f.name.split(".")
        s1, s2 = parts[0], parts[1]
        
        # 统计有效行数
        with open(f, "r", encoding="utf-8") as fp:
            line_count = sum(1 for _ in fp) - 1

        total_links += line_count
        if s1 == s2:
            intra_links += line_count
            intra_files += 1
            sample_link_counts[s1] = line_count
        else:
            inter_links += line_count
            inter_files += 1

    print(f"Link 文件总数: {len(files)} (样本内自连: {intra_files} 个文件, 跨样本互连: {inter_files} 个文件)")
    print(f"连接 (Edge) 总数: {total_links:,}")
    print(f"  ├─ 样本内连接 (Intra-sample links): {intra_links:,} ({intra_links / total_links * 100:.2f}%)")
    print(f"  └─ 样本间连接 (Inter-sample links): {inter_links:,} ({inter_links / total_links * 100:.2f}%)")

    # 抽取 1 个文件详细展示特征分布
    if files:
        sample_file = files[0]
        sample_df = pd.read_csv(sample_file, sep="\t")
        print(f"\n以样例文件 [{sample_file.name}] 为例的特征分布:")
        cols_to_show = ["pairs_all", "pairs_bothMQ20", "orientation_consistency"]
        avail_cols = [c for c in cols_to_show if c in sample_df.columns]
        if avail_cols:
            print(sample_df[avail_cols].describe().round(3).to_string())


def main():
    parser = argparse.ArgumentParser(description="统计 contig2bin 数据集概况")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="数据/联调数据",
        help="数据根目录路径（默认: 数据/联调数据）"
    )
    args = parser.parse_args()
    data_path = Path(args.data_dir)

    print(f"正在分析数据目录: {data_path.resolve()}")
    if not data_path.exists():
        print(f"错误: 目录 {data_path} 不存在！")
        return

    cov_dir = data_path / "coverage"
    ctg_dir = data_path / "ctg2genome"
    links_dir = data_path / "links"

    if cov_dir.exists():
        analyze_coverage(cov_dir)
    if ctg_dir.exists():
        analyze_ctg2genome(ctg_dir)
    if links_dir.exists():
        analyze_links(links_dir)

    print("\n" + "=" * 80)
    print("数据初步统计分析完成！")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
