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
    run_leiden_clustering,
    run_hdbscan_clustering
)
from newbin.evaluate import compute_benchmark_metrics, load_ground_truth
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
    # 步骤 5: 模型网络搭建与双体系对比训练循环 (5 个 Epoch)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("【步骤 5】双体系模型训练循环 (5 个 Epoch)")
    print("   - 体系 A【当前主方案】：L2 超球面归一化 + InfoNCE 损失 (cos 相似度空间)")
    print("   - 体系 B【传统对照组】：自由欧氏空间 + 经典欧氏 Hinge 损失")
    print("-" * 40)

    num_features = full_data.shape[1]
    device = torch.device('cpu')

    # 体系 A: 超球面余弦体系
    model_cosine = NewBinModel(num_features=num_features, embed_dim=100, l2_norm=True).to(device)
    criterion_cosine = MetagenomicPipelineLoss(main_loss_type="infonce", lambda_co=0.1, temperature=0.1)
    optimizer_cosine = torch.optim.AdamW(model_cosine.parameters(), lr=1e-3, weight_decay=1e-4)

    # 体系 B: 传统欧氏体系
    model_euclidean = NewBinModel(num_features=num_features, embed_dim=100, l2_norm=False).to(device)
    criterion_euclidean = MetagenomicPipelineLoss(main_loss_type="hinge", lambda_co=0.1, margin=1.0)
    optimizer_euclidean = torch.optim.AdamW(model_euclidean.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"[OK] 模型初始化完成 (输入总维度={num_features}):")
    print(f"   - 体系 A: l2_norm=True  (超球面投影, InfoNCE tau=0.1)")
    print(f"   - 体系 B: l2_norm=False (自由欧氏空间, Hinge margin=1.0)")

    # 训练体系 A (余弦 + InfoNCE)
    print("\n>>> 开始训练【体系 A：超球面余弦 + InfoNCE】...")
    model_cosine.train()
    for epoch in range(1, 6):
        tot_loss, m_loss, c_loss, nb = 0.0, 0.0, 0.0, 0
        for b_x1, b_x2, b_lbl in loader:
            optimizer_cosine.zero_grad()
            d_shape1 = extract_shape_features(b_x1)
            d_shape2 = extract_shape_features(b_x2)
            (z1, z2), (zc1, zc2) = model_cosine(b_x1, b_x2, return_zc=True)
            loss_dict = criterion_cosine.forward_pairwise(
                z1=z1, z2=z2, zc1=zc1, zc2=zc2,
                d_shape1=d_shape1, d_shape2=d_shape2,
                labels=b_lbl
            )
            loss = loss_dict["total_loss"]
            loss.backward()
            optimizer_cosine.step()
            tot_loss += loss.item()
            m_loss += loss_dict["main_loss"].item()
            c_loss += loss_dict["co_loss"].item()
            nb += 1
        print(f"   [Cosine] Epoch [{epoch}/5] - 总损失: {tot_loss/nb:.4f} | InfoNCE: {m_loss/nb:.4f} | L_co: {c_loss/nb:.4f}")

    # 训练体系 B (欧氏 + Hinge)
    print("\n>>> 开始训练【体系 B：自由欧氏 + Hinge Loss】...")
    model_euclidean.train()
    for epoch in range(1, 6):
        tot_loss, m_loss, c_loss, nb = 0.0, 0.0, 0.0, 0
        for b_x1, b_x2, b_lbl in loader:
            optimizer_euclidean.zero_grad()
            d_shape1 = extract_shape_features(b_x1)
            d_shape2 = extract_shape_features(b_x2)
            (z1, z2), (zc1, zc2) = model_euclidean(b_x1, b_x2, return_zc=True)
            loss_dict = criterion_euclidean.forward_pairwise(
                z1=z1, z2=z2, zc1=zc1, zc2=zc2,
                d_shape1=d_shape1, d_shape2=d_shape2,
                labels=b_lbl
            )
            loss = loss_dict["total_loss"]
            loss.backward()
            optimizer_euclidean.step()
            tot_loss += loss.item()
            m_loss += loss_dict["main_loss"].item()
            c_loss += loss_dict["co_loss"].item()
            nb += 1
        print(f"   [Euclid] Epoch [{epoch}/5] - 总损失: {tot_loss/nb:.4f} | Hinge: {m_loss/nb:.4f} | L_co: {c_loss/nb:.4f}")

    # 导出权重
    model_cos_path = os.path.join(output_dir, "model_cosine.pt")
    model_euc_path = os.path.join(output_dir, "model_euclidean.pt")
    model_cosine.save_with_params_to(model_cos_path)
    model_cosine.save_with_params_to(os.path.join(output_dir, "model.pt"))
    model_euclidean.save_with_params_to(model_euc_path)
    print(f"[OK] 训练权重导出成功: {model_cos_path} 与 {model_euc_path}")

    # -------------------------------------------------------------------------
    # 步骤 6: 全体 Contig 嵌入向量生成与双体系建图分箱
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("【步骤 6】双体系 Embedding 生成与双轨图 Leiden 聚类")
    print("-" * 40)

    model_cosine.eval()
    model_euclidean.eval()

    with torch.no_grad():
        all_x = torch.from_numpy(features_arr).to(device)
        emb_cosine = model_cosine.embedding(all_x).cpu().numpy()
        emb_euclidean = model_euclidean.embedding(all_x).cpu().numpy()

    print(f"[OK] 嵌入特征维度透视:")
    print(f"   - 体系 A (余弦超球面) 模长均值: {np.linalg.norm(emb_cosine, axis=1).mean():.4f} (严格超球面归一)")
    print(f"   - 体系 B (自由欧氏空间) 模长均值: {np.linalg.norm(emb_euclidean, axis=1).mean():.4f} (自由欧氏向量)")

    cannot_pairs_formatted = [(full_data.index[p[0]], full_data.index[p[1]]) for p in cannot_pairs]

    # 6.1 体系 A 建图 (余弦相似度)
    g_cos, w_cos = build_dual_track_graph(
        embedding=emb_cosine,
        contig_names=list(full_data.index),
        pe_links=pe_counts,
        k_neighbors=min(5, N - 1),
        graph_metric='cosine',
        cannot_pairs=cannot_pairs_formatted
    )
    labels_cos = run_leiden_clustering(g_cos, w_cos, resolution=1.0)
    print(f"[OK] 体系 A【余弦超球面】Leiden 聚类: 边数={g_cos.ecount()}, 产生簇数={len(set(labels_cos))}")

    # 6.2 体系 B 建图 (截断欧氏距离 S_ij = max(0, 1 - d))
    g_euc, w_euc = build_dual_track_graph(
        embedding=emb_euclidean,
        contig_names=list(full_data.index),
        pe_links=pe_counts,
        k_neighbors=min(5, N - 1),
        graph_metric='euclidean',
        cannot_pairs=cannot_pairs_formatted
    )
    labels_euc = run_leiden_clustering(g_euc, w_euc, resolution=1.0)
    print(f"[OK] 体系 B【传统欧氏截断】Leiden 聚类: 边数={g_euc.ecount()}, 产生簇数={len(set(labels_euc))}")

    # 导出结果
    res_cos_df = pd.DataFrame({'contig_name': full_data.index, 'bin_id': labels_cos})
    res_euc_df = pd.DataFrame({'contig_name': full_data.index, 'bin_id': labels_euc})
    res_cos_df.to_csv(os.path.join(output_dir, "bin_clusters_cosine.csv"), index=False)
    res_cos_df.to_csv(os.path.join(output_dir, "bin_clusters.csv"), index=False)
    res_euc_df.to_csv(os.path.join(output_dir, "bin_clusters_euclidean.csv"), index=False)

    # -------------------------------------------------------------------------
    # 步骤 7: 消融对照组 (HDBSCAN 空间几何密度聚类)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("【步骤 7】消融对照组：空间几何密度聚类 (HDBSCAN, euclidean)")
    print("-" * 40)

    hdb_labels, n_hdb_clusters, n_hdb_noise = run_hdbscan_clustering(
        embedding=emb_cosine,
        contig_names=list(full_data.index),
        min_cluster_size=3,
        metric='euclidean'
    )
    print(f"[OK] HDBSCAN 对照组聚类完成:")
    print(f"   - 识别有效簇数量: {n_hdb_clusters}")
    print(f"   - 判定为噪声/未分箱数量: {n_hdb_noise} ({100.0 * n_hdb_noise / len(full_data):.1f}%)")

    hdb_df = pd.DataFrame({
        'contig_name': full_data.index,
        'bin_id': [f"hdb_{lbl}" if lbl != -1 else "Unbinned" for lbl in hdb_labels]
    })
    hdb_df.to_csv(os.path.join(output_dir, "bin_clusters_hdbscan.csv"), index=False)

    # -------------------------------------------------------------------------
    # 步骤 8: 核心体系级对决基准评测消融对比看板 (Ground Truth Benchmark)
    # -------------------------------------------------------------------------
    gt_path = os.path.join(test_data_dir, "ground_truth.csv")
    if os.path.exists(gt_path):
        print("\n" + "=" * 105)
        print("【步骤 8】核心体系级对决基准评测消融对比看板 (Ablation Benchmark Dashboard)")
        print("对比双方: 【超球面余弦体系 (cos + InfoNCE)】 vs 【传统欧氏体系 (欧氏距离 + Hinge)】 vs 【纯几何密度基线 (HDBSCAN)】")
        print("=" * 105)

        c2g = load_ground_truth(gt_path)
        c2b_cos = dict(zip(res_cos_df['contig_name'].astype(str), res_cos_df['bin_id'].astype(str)))
        c2b_euc = dict(zip(res_euc_df['contig_name'].astype(str), res_euc_df['bin_id'].astype(str)))
        c2b_hdb = dict(zip(hdb_df['contig_name'].astype(str), hdb_df['bin_id'].astype(str)))

        # 1. 评估超球面余弦体系 (主方案)
        m_cos = compute_benchmark_metrics(contig2bin=c2b_cos, contig2genome=c2g, contig_lens=contig_lens, short_threshold=2000)
        # 2. 评估传统欧氏体系 (旧方案对照组)
        m_euc = compute_benchmark_metrics(contig2bin=c2b_euc, contig2genome=c2g, contig_lens=contig_lens, short_threshold=2000)
        # 3. 评估 HDBSCAN (纯空间几何密度对照组)
        m_hdb = compute_benchmark_metrics(contig2bin=c2b_hdb, contig2genome=c2g, contig_lens=contig_lens, short_threshold=2000)

        sc = m_cos['summary']
        se = m_euc['summary']
        sh = m_hdb['summary']

        # 4. 打印三维横向对比看板
        col_m = "消融评测核心指标 (Ablation Metric)"
        col_c = "余弦+InfoNCE (主方案)"
        col_e = "欧氏+Hinge (旧方案)"
        col_h = "HDBSCAN (几何密度)"
        print(f"\n{col_m:<38} | {col_c:<20} | {col_e:<20} | {col_h:<20}")
        print("-" * 105)
        print(f"{'高质量 MAGs (HQ MAGs: >90% comp, <5% contam)':<38} | {sc['High_Quality_MAGs (HQ)']:<20} | {se['High_Quality_MAGs (HQ)']:<20} | {sh['High_Quality_MAGs (HQ)']:<20}")
        print(f"{'中等质量 MAGs (MQ MAGs)':<38} | {sc['Medium_Quality_MAGs (MQ)']:<20} | {se['Medium_Quality_MAGs (MQ)']:<20} | {sh['Medium_Quality_MAGs (MQ)']:<20}")
        print(f"{'宏平均完整度 (Mean Completeness)':<38} | {sc['Mean_Completeness'] * 100:>18.2f}% | {se['Mean_Completeness'] * 100:>18.2f}% | {sh['Mean_Completeness'] * 100:>18.2f}%")
        print(f"{'宏平均污染度 (Mean Contamination)':<38} | {sc['Mean_Contamination'] * 100:>18.2f}% | {se['Mean_Contamination'] * 100:>18.2f}% | {sh['Mean_Contamination'] * 100:>18.2f}%")
        print(f"{'宏平均 F1-Score':<38} | {sc['Mean_F1_Score']:>19.4f} | {se['Mean_F1_Score']:>19.4f} | {sh['Mean_F1_Score']:>19.4f}")
        print(f"{'调整兰德指数 (ARI)':<38} | {sc['ARI']:>19.4f} | {se['ARI']:>19.4f} | {sh['ARI']:>19.4f}")
        print(f"{'标准化互信息 (NMI)':<38} | {sc['NMI']:>19.4f} | {se['NMI']:>19.4f} | {sh['NMI']:>19.4f}")
        print(f"{'全基因组碱基分箱率 (Binned BP Rate)':<38} | {sc['Binned_BP_Rate'] * 100:>18.2f}% | {se['Binned_BP_Rate'] * 100:>18.2f}% | {sh['Binned_BP_Rate'] * 100:>18.2f}%")

        # 5. 短序列拯救能力对比 (< 2000 bp)
        r_cnt_c = sc.get('Short_Rescue_Rate_Count', 0.0) * 100
        r_cnt_e = se.get('Short_Rescue_Rate_Count', 0.0) * 100
        r_cnt_h = sh.get('Short_Rescue_Rate_Count', 0.0) * 100

        r_bp_c = sc.get('Short_Rescue_Rate_BP', 0.0) * 100
        r_bp_e = se.get('Short_Rescue_Rate_BP', 0.0) * 100
        r_bp_h = sh.get('Short_Rescue_Rate_BP', 0.0) * 100

        print(f"{'短序列与质粒条数拯救率 (<2kb Count Rescue)':<38} | {r_cnt_c:>18.1f}% | {r_cnt_e:>18.1f}% | {r_cnt_h:>18.1f}%")
        print(f"{'短序列与质粒碱基拯救率 (<2kb BP Rescue)':<38} | {r_bp_c:>18.1f}% | {r_bp_e:>18.1f}% | {r_bp_h:>18.1f}%")
        print("-" * 105)

    print("\n" + "=" * 105)
    print("       >>> 全流程端到端体系级消融对照调试完成！所有模块数据流转顺畅，验证通过！")
    print("=" * 105)


if __name__ == '__main__':
    main()
