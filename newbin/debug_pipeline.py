#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NewBin 模型训练与全流程数据调试脚本 (debug_pipeline.py)
======================================================
用于在本地单机环境透视查看：
  1. FASTA 与 BAM 的原始特征矩阵提取形态 (k-mer + Coverage + Length)
  2. 先验约束与 BAM PE Reads 跨 Contig 物理连接提取结果
  3. 成对样本平衡采样 (1:2 比例) 与 DataLoader 批次张量
  4. 神经网络孪生前向传播与双轨损失 (L_main + L_co) 计算
  5. 完整训练循环过程与权重保存 (model.pt)
  6. 100 维 Embedding 导出与双轨图 Leiden 分箱聚类结果
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# 确保能正确引入 newbin 包
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from newbin.generate_kmer import generate_kmer_features_from_fasta
from newbin.cluster import (
    parse_bam_pe_reads_stream,
    extract_pe_bridging_links,
    build_dual_track_graph,
    run_leiden_clustering
)
from newbin.model import (
    NewBinModel,
    PairwiseContigDataset,
    sample_balanced_pairs,
    model_load
)
from newbin.loss import (
    extract_shape_features,
    MetagenomicPipelineLoss
)

# 确保 Windows 控制台输出不发生 GBK 编码错误
if sys.platform == "win32":
    try:
        import io
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("DebugPipeline")


def main():
    print("=" * 80)
    print("       >>> NewBin 宏基因组分箱模型全流程数据调试 (debug_pipeline.py)")
    print("=" * 80)

    test_data_dir = os.path.join(BASE_DIR, "tests", "test_data")
    fasta_path = os.path.join(test_data_dir, "contigs.fasta")
    bam_path = os.path.join(test_data_dir, "sample.sorted.bam")
    must_link_path = os.path.join(test_data_dir, "constraints", "must_link.csv")
    cannot_link_path = os.path.join(test_data_dir, "constraints", "cannot_link.csv")

    output_dir = os.path.join(BASE_DIR, "debug_output")
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 步骤 1: 提取 k-mer 频率与 Contig 序列长度
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("【步骤 1】从 FASTA 提取 k-mer 频率与序列长度")
    print("-" * 40)
    print(f"输入 FASTA 文件: {fasta_path}")

    # length_threshold 设为 500 bp，保证测试样本都能进入
    kmer_df, contig_lens = generate_kmer_features_from_fasta(
        fasta_path, length_threshold=500, kmer_len=4, return_lengths=True
    )
    print(f"[OK] 成功解析 Contig 数量: {len(kmer_df)}")
    print(f"[OK] k-mer 特征矩阵形状: {kmer_df.shape} (前 136 维)")
    print(f"[OK] 前 3 条 Contig 真实长度:")
    for ctg in list(kmer_df.index)[:3]:
        print(f"   - {ctg}: {contig_lens[ctg]} bp")

    # -------------------------------------------------------------------------
    # 步骤 2: 构造 Coverage 特征并拼装完整 data.csv (138+2M 维)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("【步骤 2】组装 Coverage 特征与最终 data.csv 矩阵")
    print("-" * 40)

    # 模拟 M=2 个测序样本的 Coverage 特征: [level(1), shape(2), log_id(2)] = 5 维
    # (真实场景由 generate_coverage.py 从 BAM 计算产生)
    np.random.seed(42)
    N = len(kmer_df)
    M = 2
    mu_sim = np.random.uniform(5.0, 100.0, size=(N, M))
    log_mu = np.log(mu_sim + 1e-5)
    level = log_mu.mean(axis=1, keepdims=True)
    shape = log_mu - level
    log_id = np.random.uniform(-1.0, 1.0, size=(N, M))

    cov_df = pd.DataFrame(
        np.hstack([level, shape, log_id]),
        index=kmer_df.index,
        columns=['cov_level', 'cov_shape_s1', 'cov_shape_s2', 'cov_log_id_s1', 'cov_log_id_s2']
    )

    # 拼接: k-mer (136) + Coverage (5) + length (1) = 142 维
    full_data = pd.concat([kmer_df, cov_df], axis=1)
    full_data['length'] = full_data.index.map(contig_lens).astype(float)

    data_csv_path = os.path.join(output_dir, "data.csv")
    full_data.to_csv(data_csv_path)

    print(f"[OK] 完整特征矩阵形状: {full_data.shape} ({N} contigs, {full_data.shape[1]} 维特征)")
    print(f"[OK] 特征切片划分:")
    print(f"   - k-mer:   列 0..135 (宽度 136)")
    print(f"   - Coverage:列 136..140 (宽度 {2*M+1}, M={M} 样本)")
    print(f"   - Length:  最后一列 (宽度 1)")
    print(f"[OK] 已持久化至: {data_csv_path}")

    # -------------------------------------------------------------------------
    # 步骤 3: 从 BAM 文件流式提取成对 Reads 跨 Contig 连接 (PE Links)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("【步骤 3】从 BAM 文件解析成对 Reads 跨 Contig 物理连接")
    print("-" * 40)
    print(f"输入 BAM 文件: {bam_path}")

    pe_counts = parse_bam_pe_reads_stream(bam_path, min_mapq=0)
    print(f"[OK] 扫描 BAM 得到跨 Contig 配对读段数: {len(pe_counts)} 对")
    if pe_counts:
        top_pairs = list(pe_counts.items())[:3]
        for (c1, c2), cnt in top_pairs:
            print(f"   - ({c1} <---> {c2}): 共享 {cnt} 对配对 reads")

    # -------------------------------------------------------------------------
    # 步骤 4: 构建成对约束集与 DataLoader (1:2 动态负采样)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("【步骤 4】解析约束集并执行 1:2 平衡负采样")
    print("-" * 40)

    contig2idx = {name: idx for idx, name in enumerate(full_data.index)}

    # 读取必须连 (Must-link)
    must_pairs = []
    if os.path.exists(must_link_path):
        raw_m = pd.read_csv(must_link_path, header=None).values
        for r in raw_m:
            c1, c2 = str(r[0]).strip(), str(r[1]).strip()
            if c1 in contig2idx and c2 in contig2idx and c1 != c2:
                must_pairs.append((contig2idx[c1], contig2idx[c2]))

    # 读取不能连 (Cannot-link)
    cannot_pairs = []
    if os.path.exists(cannot_link_path):
        raw_c = pd.read_csv(cannot_link_path, header=None).values
        for r in raw_c:
            c1, c2 = str(r[0]).strip(), str(r[1]).strip()
            if c1 in contig2idx and c2 in contig2idx and c1 != c2:
                cannot_pairs.append((contig2idx[c1], contig2idx[c2]))

    print(f"[OK] 有效 Must-link 正对数: {len(must_pairs)}")
    print(f"[OK] 有效 Cannot-link 负对数: {len(cannot_pairs)}")

    # 采样平衡成对样本 (正:负 = 1:2)
    pairs, labels = sample_balanced_pairs(
        num_contigs=len(full_data),
        must_pairs=must_pairs,
        cannot_pairs=cannot_pairs,
        neg_ratio=2,
        rng=np.random.default_rng(42)
    )
    print(f"[OK] 平衡采样后总样本对数: {len(pairs)} (正样本={int(np.sum(labels==1))}, 负样本={int(np.sum(labels==0))})")

    # 组装 Dataset & DataLoader
    features_arr = full_data.values.astype(np.float32)
    dataset = PairwiseContigDataset(features_arr, pairs, labels)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 打印第 1 个 Batch 的形态
    batch_x1, batch_x2, batch_labels = next(iter(loader))
    print(f"[OK] 第 1 个 Batch 张量维度透视:")
    print(f"   - x1:     {batch_x1.shape}")
    print(f"   - x2:     {batch_x2.shape}")
    print(f"   - labels: {batch_labels.shape} -> {batch_labels.tolist()} (1.0=正对, 0.0=负对)")

    # -------------------------------------------------------------------------
    # 步骤 5: 模型初始化、前向传播与双轨损失计算
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("【步骤 5】模型网络搭建与双轨损失训练循环 (5 个 Epoch)")
    print("-" * 40)

    num_features = full_data.shape[1]
    device = torch.device('cpu')
    model = NewBinModel(num_features=num_features, embed_dim=100).to(device)

    # 双轨联合损失函数: L_total = L_main + 0.1 * L_co
    criterion = MetagenomicPipelineLoss(main_loss_type="hinge", lambda_co=0.1, margin=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"[OK] 模型架构概要:")
    print(f"   - 输入总维度: {num_features}")
    print(f"   - k-mer 分支: Linear(136 -> 256) -> BN -> LeakyReLU -> Linear(256 -> 64)")
    print(f"   - 长度门控:   log(1+L) -> Linear(1 -> 1) -> Sigmoid -> g(1)")
    print(f"   - Coverage支路: Linear({2*M+1} -> 64) -> BN -> LeakyReLU -> Linear(64 -> 32)")
    print(f"   - 融合网络:   Linear(64+32=96 -> 100 维)")

    model.train()
    for epoch in range(1, 6):
        total_loss_accum = 0.0
        main_loss_accum = 0.0
        co_loss_accum = 0.0
        batches = 0

        for b_x1, b_x2, b_lbl in loader:
            optimizer.zero_grad()

            # 1. 切片提取原始 Shape 特征 d (M维)
            d_shape1 = extract_shape_features(b_x1)
            d_shape2 = extract_shape_features(b_x2)

            # 2. 孪生前向传播，同时返回最终 z 与 Coverage z_c
            (z1, z2), (zc1, zc2) = model(b_x1, b_x2, return_zc=True)

            # 3. 计算双轨损失
            loss_dict = criterion.forward_pairwise(
                z1=z1, z2=z2,
                zc1=zc1, zc2=zc2,
                d_shape1=d_shape1, d_shape2=d_shape2,
                labels=b_lbl
            )

            loss = loss_dict["total_loss"]
            loss.backward()
            optimizer.step()

            total_loss_accum += loss.item()
            main_loss_accum += loss_dict["main_loss"].item()
            co_loss_accum += loss_dict["co_loss"].item()
            batches += 1

        avg_total = total_loss_accum / max(1, batches)
        avg_main = main_loss_accum / max(1, batches)
        avg_co = co_loss_accum / max(1, batches)

        print(f"   Epoch [{epoch}/5] - 总损失: {avg_total:.4f} | 主损失(Hinge): {avg_main:.4f} | 辅助共丰度损失(L_co): {avg_co:.4f}")

    # 保存训练好的模型权重
    model_save_path = os.path.join(output_dir, "model.pt")
    model.save_with_params_to(model_save_path)
    print(f"[OK] 训练完毕，模型权重已导出至: {model_save_path}")

    # -------------------------------------------------------------------------
    # 步骤 6: 全体 Contig 嵌入向量生成与 Leiden 聚类分箱
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("【步骤 6】生成 100 维 Embedding 与双轨图 Leiden 聚类")
    print("-" * 40)

    loaded_model = model_load(model_save_path, device=device)
    loaded_model.eval()

    with torch.no_grad():
        all_x = torch.from_numpy(features_arr).to(device)
        embeddings = loaded_model.embedding(all_x).cpu().numpy()

    print(f"[OK] 全体 Contig 嵌入特征矩阵形状: {embeddings.shape} (固定 100 维)")
    emb_norms = np.linalg.norm(embeddings, axis=1)
    print(f"[OK] 嵌入向量模长统计: 均值={emb_norms.mean():.4f}, 最小={emb_norms.min():.4f}, 最大={emb_norms.max():.4f}")

    # 构建双轨增强图
    g, weights = build_dual_track_graph(
        embedding=embeddings,
        contig_names=list(full_data.index),
        pe_links=pe_counts,
        k_neighbors=min(5, N - 1),
        cannot_pairs=[(full_data.index[p[0]], full_data.index[p[1]]) for p in cannot_pairs]
    )
    print(f"[OK] 双轨增强图构建成功: 节点数={g.vcount()}, 边数={g.ecount()}")

    # 运行 Leiden 社区聚类划分
    cluster_labels = run_leiden_clustering(g, weights, resolution=1.0)
    unique_bins = set(cluster_labels)
    print(f"[OK] Leiden 聚类划分完成: 共划分出 {len(unique_bins)} 个 Bin 簇")

    # 统计各 Bin 分布
    from collections import Counter
    bin_counts = Counter(cluster_labels)
    for b_id, cnt in sorted(bin_counts.items()):
        print(f"   - Bin #{b_id:02d}: 包含 {cnt} 条 Contig")

    # 结果导出为 CSV
    res_df = pd.DataFrame({
        'contig_name': full_data.index,
        'bin_id': cluster_labels
    })
    res_csv = os.path.join(output_dir, "bin_clusters.csv")
    res_df.to_csv(res_csv, index=False)
    print(f"[OK] 聚类结果已保存至: {res_csv}")

    print("\n" + "=" * 80)
    print("       >>> 全流程端到端调试完成！所有模块数据流转顺畅，验证通过！")
    print("=" * 80)


if __name__ == '__main__':
    main()
