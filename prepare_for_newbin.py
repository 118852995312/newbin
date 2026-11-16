#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联调数据适配器：将【数据/联调数据】快速转换为 NewBin 标准输入格式
生成：
  1. contigs.fa         (标准 FASTA 组装序列文件)
  2. data.csv           (148 维标准特征矩阵：136 kmer + 11 coverage + 1 length)
  3. pe_links.csv       (NewBin 标准 3 列 PE 连接边文件: ctg1, ctg2, count)
  4. ground_truth.tsv   (标准答案评估标签)
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# 引入 newbin 自身的 k-mer 生成逻辑以保证 100% 算法一致性
import sys
sys.path.insert(0, os.path.abspath("newbin/newbin"))
from newbin.generate_kmer import generate_feature_mapping, _build_lut_4mer


def extract_4mer_matrix(sequences: list, kmer_dict: dict, kmer_len: int = 4, eps: float = 1e-5) -> np.ndarray:
    """批量高效计算 4-mer 归一化频率矩阵"""
    n_seq = len(sequences)
    n_feats = max(kmer_dict.values()) + 1
    matrix = np.full((n_seq, n_feats), eps, dtype=np.float32)

    byte_map, lut = _build_lut_4mer(kmer_dict)

    for i, seq in enumerate(sequences):
        seq_bytes = seq.encode('latin1')
        b_arr = byte_map[np.frombuffer(seq_bytes, dtype=np.uint8)]
        valid_mask = (b_arr != 255)

        if len(b_arr) >= kmer_len:
            # 4-mer 2-bit sliding window
            c0 = b_arr[:-3]
            c1 = b_arr[1:-2]
            c2 = b_arr[2:-1]
            c3 = b_arr[3:]
            v_all = valid_mask[:-3] & valid_mask[1:-2] & valid_mask[2:-1] & valid_mask[3:]
            idx = (c0 << 6) | (c1 << 4) | (c2 << 2) | c3
            valid_idx = idx[v_all]
            feats = lut[valid_idx]
            feats = feats[feats >= 0]
            if len(feats) > 0:
                counts = np.bincount(feats, minlength=n_feats)
                matrix[i] += counts[:n_feats]

    # 行归一化
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix /= np.maximum(row_sums, 1e-9)
    return matrix


def convert_sample(data_dir: Path, sample_id: str, output_dir: Path, min_len: int = 1000, top_n: int = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    if top_n:
        print(f"\n>> 正在为样本 [{sample_id}] 提取前 {top_n} 条数据并准备 NewBin 极简测试集...")
    else:
        print(f"\n>> 正在为样本 [{sample_id}] 准备 NewBin 标准输入数据...")

    ctg_file = data_dir / "ctg2genome" / f"{sample_id}.ctg2genome.tsv"
    cov_file = data_dir / "coverage" / f"{sample_id}.coverage"
    link_files = sorted(list((data_dir / "links").glob(f"{sample_id}.*.contig_links.highconf.tsv")))

    if not ctg_file.exists():
        raise FileNotFoundError(f"未找到序列文件: {ctg_file}")
    if not cov_file.exists():
        raise FileNotFoundError(f"未找到覆盖度文件: {cov_file}")

    # 1. 读取 ctg2genome 并输出 contigs.fa 与 ground_truth.tsv
    print("1. 解析序列与真值标签...")
    read_nrows = (top_n * 5) if (top_n is not None and top_n > 0) else None
    df_ctg = pd.read_csv(ctg_file, sep="\t", nrows=read_nrows)
    # 过滤 min_len
    df_ctg["len"] = df_ctg["ctg_sequence"].str.len()
    df_ctg = df_ctg[df_ctg["len"] >= min_len].copy()
    if top_n is not None and top_n > 0:
        df_ctg = df_ctg.head(top_n).copy()

    fa_path = output_dir / "contigs.fa"
    with open(fa_path, "w", encoding="utf-8") as f_fa:
        for _, row in df_ctg.iterrows():
            f_fa.write(f">{row['ctg_id']}\n{row['ctg_sequence']}\n")
    print(f"   已生成标准 FASTA 文件: {fa_path} ({len(df_ctg)} 条 contigs)")

    gt_path = output_dir / "ground_truth.tsv"
    df_ctg[["ctg_id", "genome_id"]].to_csv(gt_path, sep="\t", index=False)
    print(f"   已生成真值评估标签: {gt_path}")

    # 2. 计算 136 维 k-mer 特征
    print("2. 计算 136 维 4-mer 组成频率矩阵...")
    kmer_dict, n_feats = generate_feature_mapping(4)
    kmer_mat = extract_4mer_matrix(df_ctg["ctg_sequence"].tolist(), kmer_dict, 4)
    kmer_cols = [f"kmer_{i}" for i in range(n_feats)]
    df_kmer = pd.DataFrame(kmer_mat, index=df_ctg["ctg_id"], columns=kmer_cols)

    # 3. 计算 11 维 Coverage 共丰度特征 (Level, Shape, Log-ID)
    print("3. 从 .coverage 计算 11 维标准共丰度特征 (M=5)...")
    cov_nrows = (top_n * 5) if (top_n is not None and top_n > 0) else None
    df_cov = pd.read_csv(cov_file, sep="\t", nrows=cov_nrows)
    df_cov.set_index("contigName", inplace=True)
    df_cov = df_cov.loc[df_cov.index.intersection(df_ctg["ctg_id"])]

    # 提取所有样本的 (mean, var)
    mean_cols = [c for c in df_cov.columns if c.endswith(".sorted.bam")]
    var_cols = [c for c in df_cov.columns if c.endswith(".sorted.bam-var")]

    mu_raw = df_cov[mean_cols].values.astype(np.float64)
    var_raw = df_cov[var_cols].values.astype(np.float64)

    # 筛选平均覆盖度最高的 5 个样本
    max_samples = 5
    avg_depths = mu_raw.mean(axis=0)
    top_indices = np.argsort(-avg_depths)[:max_samples]
    mu = mu_raw[:, top_indices]
    var = var_raw[:, top_indices]

    eps = 1e-5
    log_mu = np.log(mu + eps)
    level = log_mu.mean(axis=1, keepdims=True)  # (N, 1)
    shape = log_mu - level                       # (N, 5)
    dispersion_index = var / (mu + eps)
    log_id = np.log(np.clip(dispersion_index, eps, None))  # (N, 5)

    cov_features = np.hstack([level, shape, log_id])
    cov_cols = ["cov_level"] + [f"cov_shape_{i}" for i in range(max_samples)] + [f"cov_log_id_{i}" for i in range(max_samples)]
    df_cov_feat = pd.DataFrame(cov_features, index=df_cov.index, columns=cov_cols)

    # 4. 合并为 148 维 data.csv
    print("4. 合并生成 NewBin 标准 data.csv (148 维)...")
    data_df = pd.merge(df_kmer, df_cov_feat, left_index=True, right_index=True)
    data_df["length"] = df_ctg.set_index("ctg_id")["len"].astype(float)

    data_csv_path = output_dir / "data.csv"
    data_df.to_csv(data_csv_path)
    print(f"   已生成 data.csv: {data_csv_path} (形状: {data_df.shape})")

    # 5. 格式化 pe_links.csv (多样本累加合并)
    if link_files:
        print(f"5. 聚合 {len(link_files)} 个 PE 高置信度物理连接文件...")
        dfs = []
        for lf in link_files:
            try:
                sub_df = pd.read_csv(lf, sep="\t")
                if "ctg1" in sub_df.columns and "ctg2" in sub_df.columns and "pairs_bothMQ20" in sub_df.columns:
                    dfs.append(sub_df[["ctg1", "ctg2", "pairs_bothMQ20"]])
            except Exception as e:
                print(f"   警告: 读取连接文件 {lf} 失败: {e}")

        if dfs:
            combined_pe = pd.concat(dfs, ignore_index=True)
            # 无向图去重规范化与权重累加
            valid_ctgs = set(df_ctg["ctg_id"])
            combined_pe = combined_pe[combined_pe["ctg1"].isin(valid_ctgs) & combined_pe["ctg2"].isin(valid_ctgs)].copy()
            if not combined_pe.empty:
                combined_pe["c1"] = combined_pe[["ctg1", "ctg2"]].min(axis=1)
                combined_pe["c2"] = combined_pe[["ctg1", "ctg2"]].max(axis=1)
                pe_df = combined_pe.groupby(["c1", "c2"], as_index=False)["pairs_bothMQ20"].sum()
                pe_df.rename(columns={"c1": "ctg1", "c2": "ctg2"}, inplace=True)
                pe_path = output_dir / "pe_links.csv"
                pe_df.to_csv(pe_path, index=False)
                print(f"   已生成多样本聚合 pe_links.csv: {pe_path} ({len(pe_df)} 条物理连通边)")
            else:
                print("   未检索到属于当前过滤 contig 的有效 PE 物理边。")

    print(f"\n[OK] 样本 [{sample_id}] 数据转换完成，已完全适配 NewBin！")
    print(f"你现在可以直接使用 NewBin 跑如下命令：")
    print(f"  1. 训练模型: python -m newbin.main train_self -d {data_csv_path} -o {output_dir}")
    print(f"  2. 聚类分箱: python -m newbin.main bin -i {fa_path} -d {data_csv_path} --model-path {output_dir}/model.pt --pe-links {output_dir}/pe_links.csv -o {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="将联调数据转换为 NewBin 标准输入")
    parser.add_argument("--data_dir", type=str, default="数据/联调数据", help="联调数据目录")
    parser.add_argument("--sample", type=str, default="Mar_0", help="要处理的样本编号 (例如 Mar_0)")
    parser.add_argument("--output_dir", type=str, default="newbin_run/Mar_0", help="输出目录")
    parser.add_argument("--top_n", type=int, default=None, help="仅提取前 N 条 contig 用于快速联调/冒烟测试 (例如 5)")
    args = parser.parse_args()

    convert_sample(Path(args.data_dir), args.sample, Path(args.output_dir), top_n=args.top_n)


if __name__ == "__main__":
    main()
