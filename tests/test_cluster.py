import unittest
import os
import sys
import logging
import tempfile
import numpy as np
import pandas as pd
import torch

# 将项目目录加入 sys.path
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from newbin.cluster import (
    parse_bam_pe_reads_stream,
    extract_pe_bridging_links,
    build_dual_track_graph,
    run_leiden_clustering,
    run_dual_track_leiden,
    run_hdbscan_clustering,
    cluster
)
from newbin.model import NewBinModel


class TestDualTrackCluster(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        self.test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        self.bam_path = os.path.join(self.test_data_dir, 'sample.sorted.bam')

    def test_parse_bam_pe_reads_stream(self):
        """测试 BAM 流式解析器读取真实 BAM 文件不崩溃"""
        if os.path.exists(self.bam_path):
            counts = parse_bam_pe_reads_stream(self.bam_path, min_mapq=0)
            self.assertIsInstance(counts, dict)

    def test_extract_pe_bridging_links_and_cache(self):
        """测试 PE 物理连通提取、阈值过滤、缓存读写与 Must-link 导出"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, 'pe_links.csv')
            must_link_file = os.path.join(tmpdir, 'pe_must_link.csv')

            # 模拟已有缓存文件写入
            df_cache = pd.DataFrame([
                ['ctg_A', 'ctg_B', 10],
                ['ctg_B', 'ctg_C', 4],
                ['ctg_A', 'ctg_D', 1]  # 低于 theta=3
            ], columns=['contig_1', 'contig_2', 'count'])
            df_cache.to_csv(cache_file, index=False)

            contig_names = {'ctg_A', 'ctg_B', 'ctg_C', 'ctg_D'}

            pe_links = extract_pe_bridging_links(
                bams=None,
                contig_names=contig_names,
                min_count=3,
                cache_file=cache_file,
                must_link_export_file=must_link_file,
                must_link_min_count=5
            )

            # 验证低于 3 的 ctg_A-ctg_D 被过滤
            self.assertEqual(len(pe_links), 2)
            self.assertEqual(pe_links[('ctg_A', 'ctg_B')], 10)
            self.assertEqual(pe_links[('ctg_B', 'ctg_C')], 4)

            # 验证导出 Must-link (只有 count >= 5 的 ctg_A-ctg_B)
            self.assertTrue(os.path.exists(must_link_file))
            with open(must_link_file, 'r') as f:
                lines = f.read().strip().split('\n')
            self.assertEqual(len(lines), 1)
            self.assertIn('ctg_A,ctg_B', lines[0])

    def test_build_dual_track_graph_weights_and_cannot_link(self):
        """测试双轨融合图的公式计算、PE直连增强与 Cannot-link 拦截"""
        num_contigs = 6
        contig_names = [f"c_{i}" for i in range(num_contigs)]

        # 构造正交的 embedding 使部分节点余弦相似度极低
        embedding = np.zeros((num_contigs, 100), dtype=np.float32)
        embedding[0, 0] = 1.0
        embedding[1, 1] = 1.0  # c_0 与 c_1 语义距离极远，余弦相似度 = 0
        embedding[2, 0] = 0.99 # c_2 与 c_0 语义非常接近
        embedding[3, 1] = 0.99 # c_3 与 c_1 语义非常接近
        embedding[4, 2] = 1.0
        embedding[5, 3] = 1.0

        # 在 c_0 与 c_1 之间注入强 PE 物理连接 (count=6, theta=3, alpha=0.9)
        # W_pe = 0.9 * tanh(6 / 3) = 0.9 * tanh(2) ~= 0.8676
        pe_links = {
            ('c_0', 'c_1'): 6
        }
        cannot_pairs = [
            ('c_0', 'c_2')  # 强制切断本该很近的 c_0-c_2
        ]

        g, weights = build_dual_track_graph(
            embedding=embedding,
            contig_names=contig_names,
            pe_links=pe_links,
            k_neighbors=2,
            pe_alpha=0.9,
            pe_theta=3.0,
            cannot_pairs=cannot_pairs
        )

        name2idx = {name: idx for idx, name in enumerate(contig_names)}
        u0, u1 = name2idx['c_0'], name2idx['c_1']
        u2 = name2idx['c_2']

        # 1. 验证 PE 物理直连成功注入 c_0 与 c_1
        eid_01 = g.get_eid(u0, u1, error=False)
        self.assertNotEqual(eid_01, -1, "c_0 与 c_1 应当被 PE 物理边成功连接")
        expected_w_pe = 0.9 * np.tanh(2.0)
        np.testing.assert_allclose(weights[eid_01], expected_w_pe, rtol=1e-3)

        # 2. 验证 Cannot-link 成功切断 c_0 与 c_2 连边
        eid_02 = g.get_eid(u0, u2, error=False)
        self.assertEqual(eid_02, -1, "c_0 与 c_2 应该被 Cannot-link 强制切断")

    def test_run_leiden_clustering(self):
        """测试 Leiden 算法社区划分"""
        from igraph import Graph
        # 创建由两个团 (clique) 组成的图
        g = Graph()
        g.add_vertices(6)
        # 团 1: 0-1, 1-2, 0-2 (高权重)
        # 团 2: 3-4, 4-5, 3-5 (高权重)
        # 跨团弱边: 2-3 (极弱边)
        edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.01], dtype=np.float32)
        g.add_edges(edges)

        labels = run_leiden_clustering(g, weights, resolution=1.0)
        self.assertEqual(len(labels), 6)
        # 团 1 (0, 1, 2) 应具有相同标签
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[1], labels[2])
        # 团 2 (3, 4, 5) 应具有相同标签
        self.assertEqual(labels[3], labels[4])
        self.assertEqual(labels[4], labels[5])
        # 团 1 和团 2 应处于不同社区
        self.assertNotEqual(labels[0], labels[3])

    def test_dual_track_leiden_end_to_end(self):
        """测试端到端双轨 Leiden 聚类流程"""
        num_contigs = 8
        num_features = 146
        contig_names = [f"k141_{i}" for i in range(num_contigs)]
        data_arr = np.random.rand(num_contigs, num_features).astype(np.float32)
        data_arr[:, -1] = 5000.0
        df = pd.DataFrame(data_arr, index=contig_names)
        contig_dict = {name: "ATCG" * 200 for name in contig_names}

        model = NewBinModel(num_features=num_features)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            pe_cache = os.path.join(tmpdir, "pe_links.csv")
            # 伪造 PE 连接边
            df_pe = pd.DataFrame([
                [contig_names[0], contig_names[1], 10],
                [contig_names[2], contig_names[3], 8]
            ], columns=['contig_1', 'contig_2', 'count'])
            df_pe.to_csv(pe_cache, index=False)

            emb, labels = run_dual_track_leiden(
                logger=None,
                model=model,
                data=df,
                device=torch.device('cpu'),
                contig_dict=contig_dict,
                pe_links_file=pe_cache,
                k_neighbors=4,
                resolution=1.0
            )

            self.assertEqual(emb.shape, (num_contigs, 100))
            self.assertEqual(len(labels), num_contigs)
            self.assertTrue(np.all(labels >= 0))

    def test_run_hdbscan_clustering_synthetic(self):
        """测试 HDBSCAN 对照组聚类在模拟高维空间中的有效聚类识别与噪声剔除"""
        # 构造两组明显聚拢的点群与若干孤立噪声点
        np.random.seed(42)
        cluster1 = np.random.normal(loc=2.0, scale=0.1, size=(8, 100))
        cluster2 = np.random.normal(loc=-2.0, scale=0.1, size=(8, 100))
        outliers = np.array([
            [10.0] * 100,
            [-10.0] * 100
        ])
        embedding = np.vstack([cluster1, cluster2, outliers])
        contig_names = [f"ctg_{i}" for i in range(len(embedding))]

        labels, n_clusters, n_noise = run_hdbscan_clustering(
            embedding=embedding,
            contig_names=contig_names,
            min_cluster_size=5,
            metric='euclidean'
        )

        self.assertEqual(len(labels), len(embedding))
        self.assertGreaterEqual(n_clusters, 2)
        self.assertGreaterEqual(n_noise, 1)
        self.assertEqual(n_noise, int(np.sum(labels == -1)))
        # 离群点应被归为未分箱 (-1)
        self.assertEqual(labels[-1], -1)
        self.assertEqual(labels[-2], -1)

    def test_run_hdbscan_clustering_small_sample(self):
        """测试样本量小于 min_cluster_size 时 HDBSCAN 优雅降级返回全噪声"""
        embedding = np.random.randn(3, 100)
        labels, n_clusters, n_noise = run_hdbscan_clustering(
            embedding=embedding,
            min_cluster_size=5
        )
        self.assertEqual(len(labels), 3)
        self.assertEqual(n_clusters, 0)
        self.assertEqual(n_noise, 3)
        self.assertTrue(np.all(labels == -1))

    def test_cluster_pipeline_with_hdbscan_option(self):
        """测试主聚类 pipeline 切换为 HDBSCAN 对照组的分箱端到端执行与文件产出"""
        import argparse
        import logging

        logger = logging.getLogger('TestHDBSCANPipeline')
        num_contigs = 12
        num_features = 138
        contig_names = [f"seq_{i}" for i in range(num_contigs)]

        # 构造两组明显密集的特征数据
        data_arr = np.zeros((num_contigs, num_features), dtype=np.float32)
        data_arr[:6, :50] = 5.0
        data_arr[6:, 50:100] = -5.0
        data_arr[:, -1] = 5000.0  # length 列
        df = pd.DataFrame(data_arr, index=contig_names)
        contig_dict = {name: "ATCG" * 1250 for name in contig_names}

        model = NewBinModel(num_features=num_features)
        model.eval()

        args = argparse.Namespace(
            clustering_algorithm='hdbscan',
            hdbscan_min_cluster_size=4,
            hdbscan_min_samples=None,
            hdbscan_metric='euclidean',
            recluster=False,
            write_pre_reclustering_bins=False,
            output_tag=None,
            output_compression='none'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cluster(
                logger=logger,
                model=model,
                data=df,
                device=torch.device('cpu'),
                is_combined=False,
                n_sample=0,
                out=tmpdir,
                contig_dict=contig_dict,
                args=args,
                binned_length=1000,
                minfasta=2000
            )

            # 验证生成了 contig_bins.tsv
            contig_bins_path = os.path.join(tmpdir, 'contig_bins.tsv')
            self.assertTrue(os.path.exists(contig_bins_path))
            df_bins = pd.read_csv(contig_bins_path, sep='\t')
            self.assertEqual(len(df_bins), num_contigs)
            self.assertIn('contig', df_bins.columns)
            self.assertIn('bin', df_bins.columns)

    def test_build_dual_track_graph_truncated_euclidean(self):
        """测试截断欧氏距离建图: S_ij = max(0, 1.0 - min(d, 1.0)) 对齐 Hinge Loss margin=1.0"""
        contig_names = ['c0', 'c1', 'c2']
        embedding = np.zeros((3, 10), dtype=np.float32)
        # c0 at origin
        # c1 at dist = 0.4 (< 1.0, margin 内)
        embedding[1, 0] = 0.4
        # c2 at dist = 1.5 (>= 1.0, 超出 margin)
        embedding[2, 0] = 1.5

        g, weights = build_dual_track_graph(
            embedding=embedding,
            contig_names=contig_names,
            k_neighbors=2,
            graph_metric='euclidean'
        )

        u0, u1, u2 = 0, 1, 2
        eid_01 = g.get_eid(u0, u1, error=False)
        self.assertNotEqual(eid_01, -1)
        # S_01 应当精确为 1.0 - 0.4 = 0.6
        np.testing.assert_allclose(weights[eid_01], 0.6, rtol=1e-4)

        eid_02 = g.get_eid(u0, u2, error=False)
        if eid_02 != -1:
            # 若包含该边，权重必须截断为 0.0
            np.testing.assert_allclose(weights[eid_02], 0.0, atol=1e-5)

    def test_run_dual_track_leiden_with_euclidean_metric(self):
        """测试端到端双轨 Leiden 聚类支持切换 euclidean 距离体系"""
        logger = logging.getLogger('TestEuclideanLeiden')
        num_contigs = 6
        num_features = 137
        contig_names = [f"seq_{i}" for i in range(num_contigs)]
        data_arr = np.random.rand(num_contigs, num_features).astype(np.float32)
        data_arr[:, -1] = 3000.0
        df = pd.DataFrame(data_arr, index=contig_names)
        contig_dict = {name: "ATCG" * 750 for name in contig_names}

        # 自由欧氏模型: l2_norm=False
        model = NewBinModel(num_features=num_features, l2_norm=False)
        model.eval()

        emb, labels = run_dual_track_leiden(
            logger=logger,
            model=model,
            data=df,
            device=torch.device('cpu'),
            contig_dict=contig_dict,
            k_neighbors=3,
            graph_metric='euclidean'
        )

        self.assertEqual(emb.shape, (num_contigs, 100))
        self.assertEqual(len(labels), num_contigs)
        self.assertTrue(all(l >= 0 for l in labels))


if __name__ == '__main__':
    unittest.main()


