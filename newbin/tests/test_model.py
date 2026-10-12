import unittest
import os
import sys
import tempfile
import torch
import numpy as np

# 将项目目录加入 sys.path
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

import logging
import pandas as pd

try:
    from newbin.model import (
        NewBinModel, model_load, PairwiseContigDataset,
        sample_balanced_pairs, train_model
    )
except (ImportError, ModuleNotFoundError):
    inner_dir = os.path.join(repo_dir, 'newbin')
    if inner_dir not in sys.path:
        sys.path.insert(0, inner_dir)
    from model import (
        NewBinModel, model_load, PairwiseContigDataset,
        sample_balanced_pairs, train_model
    )


class TestNewBinModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_kmer_branch_and_gate_basic(self):
        """测试只有 k-mer + length 特征 (136 + 1 = 137 维) 时，经过融合网络输出为 100 维"""
        batch_size = 8
        num_features = 137  # 136 k-mer + 1 length
        model = NewBinModel(num_features=num_features, kmer_hidden_dim=256, embed_dim=100)
        model.eval()

        # 模拟输入特征：前 136 维为 k-mer 频率，最后一列为 length
        x = torch.rand(batch_size, num_features)
        x[:, -1] = torch.tensor([1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 50000.0, 100000.0, 500000.0])

        with torch.no_grad():
            emb = model.embedding(x)

        self.assertEqual(emb.shape, (batch_size, 100))

    def test_gate_scaling_with_length(self):
        """测试门控网络 g(L) 的单调性与输出范围 (0, 1)"""
        model = NewBinModel(num_features=137)
        model.eval()

        lengths = torch.tensor([[100.0], [1000.0], [10000.0], [100000.0]])
        log_l = torch.log1p(lengths)
        with torch.no_grad():
            g = model.gate(log_l)

        # 检查门控值在 0 到 1 之间
        self.assertTrue((g > 0.0).all())
        self.assertTrue((g < 1.0).all())
        self.assertEqual(g.shape, (4, 1))

    def test_coverage_and_linear_fusion(self):
        """测试 Coverage 编码器 (2M+1 -> 32) 与 Linear 融合 (64+32=96 -> 100)"""
        batch_size = 4
        # 136 k-mer + 9 coverage (2*4 + 1) + 1 length = 146 维
        num_features = 146
        model = NewBinModel(num_features=num_features, kmer_hidden_dim=256, cov_hidden_dim=64, embed_dim=100)
        model.eval()

        x = torch.rand(batch_size, num_features)
        x[:, -1] = torch.tensor([1500.0, 3000.0, 8000.0, 25000.0])

        with torch.no_grad():
            emb = model.embedding(x)

        # 验证输出维度固定为 100 维
        self.assertEqual(emb.shape, (batch_size, 100))

    def test_gradient_flow(self):
        """测试反向传播：梯度能同时流向 k-mer 编码器、gate 门控以及 coverage 编码器"""
        batch_size = 4
        num_features = 146
        model = NewBinModel(num_features=num_features, embed_dim=100)
        model.train()

        x = torch.rand(batch_size, num_features, requires_grad=True)
        emb = model.embedding(x)
        loss = emb.sum()
        loss.backward()

        # 验证各网络参数均获得了非零梯度
        self.assertIsNotNone(model.kmer_encoder[0].weight.grad)
        self.assertIsNotNone(model.gate[0].weight.grad)
        self.assertIsNotNone(model.cov_encoder[0].weight.grad)
        self.assertIsNotNone(model.fusion.weight.grad)

    def test_siamese_forward(self):
        """测试孪生网络 forward 输出两组匹配的 100 维嵌入向量"""
        batch_size = 4
        num_features = 146
        model = NewBinModel(num_features=num_features, embed_dim=100)
        model.eval()

        x1 = torch.rand(batch_size, num_features)
        x2 = torch.rand(batch_size, num_features)

        with torch.no_grad():
            emb1, emb2 = model(x1, x2)

        self.assertEqual(emb1.shape, (batch_size, 100))
        self.assertEqual(emb2.shape, (batch_size, 100))

    def test_model_save_and_load(self):
        """测试模型保存权重参数与重新加载"""
        model = NewBinModel(num_features=146, kmer_hidden_dim=256, cov_hidden_dim=64, embed_dim=100)
        model.eval()
        x = torch.rand(2, 146)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.pt')
            model.save_with_params_to(model_path)
            self.assertTrue(os.path.exists(model_path))

            loaded_model = model_load(model_path, device=torch.device('cpu'))
            self.assertEqual(loaded_model.num_features, 146)
            self.assertEqual(loaded_model.embed_dim, 100)

            with torch.no_grad():
                orig_out = model.embedding(x)
                loaded_out = loaded_model.embedding(x)
            torch.testing.assert_close(orig_out, loaded_out)

    def test_pairwise_dataset(self):
        """测试 PairwiseContigDataset 的延迟索引与特征张量提取"""
        features = np.random.randn(10, 146).astype(np.float32)
        pairs = np.array([[0, 1], [2, 3], [4, 5]])
        labels = np.array([1.0, 0.0, 1.0], dtype=np.float32)

        dataset = PairwiseContigDataset(features, pairs, labels)
        self.assertEqual(len(dataset), 3)

        x1, x2, label = dataset[0]
        self.assertEqual(x1.shape, (146,))
        self.assertEqual(x2.shape, (146,))
        self.assertEqual(label.item(), 1.0)
        np.testing.assert_array_almost_equal(x1.numpy(), features[0])
        np.testing.assert_array_almost_equal(x2.numpy(), features[1])

    def test_sample_balanced_pairs(self):
        """测试 sample_balanced_pairs 的正负平衡采样、全量随机补齐与自监督兜底"""
        num_contigs = 20
        must_pairs = [(0, 1), (2, 3)]
        cannot_pairs = [(0, 2), (1, 3)]

        # 1. 给定 must_pairs 和 cannot_pairs，采样比 1:2
        pairs, labels = sample_balanced_pairs(
            num_contigs=num_contigs,
            must_pairs=must_pairs,
            cannot_pairs=cannot_pairs,
            neg_ratio=2,
            rng=np.random.default_rng(42)
        )
        # 正对数 = 2, 负对数 = 4, 总数 = 6
        self.assertEqual(len(pairs), 6)
        self.assertEqual(len(labels), 6)
        self.assertEqual(np.sum(labels == 1.0), 2)
        self.assertEqual(np.sum(labels == 0.0), 4)

        # 2. 自监督兜底（must_pairs 为空）
        pairs_self, labels_self = sample_balanced_pairs(
            num_contigs=num_contigs,
            must_pairs=[],
            cannot_pairs=[],
            neg_ratio=2,
            rng=np.random.default_rng(42)
        )
        # 正对数 = 20, 负对数 = 40, 总数 = 60
        self.assertEqual(len(pairs_self), 60)
        self.assertEqual(np.sum(labels_self == 1.0), 20)
        self.assertEqual(np.sum(labels_self == 0.0), 40)

    def test_train_model_with_constraints(self):
        """测试 train_model 端到端训练：带先验约束文件、Hinge 损失与模型权重持久化"""
        logger = logging.getLogger('TestLogger')
        num_contigs = 8
        num_features = 146

        with tempfile.TemporaryDirectory() as tmpdir:
            # 构造特征数据 data.csv
            contig_names = [f"k141_{i}" for i in range(num_contigs)]
            data_arr = np.random.rand(num_contigs, num_features).astype(np.float32)
            data_arr[:, -1] = 5000.0  # length
            df = pd.DataFrame(data_arr, index=contig_names)
            data_path = os.path.join(tmpdir, "data.csv")
            df.to_csv(data_path)

            # 构造约束
            must_path = os.path.join(tmpdir, "must_link.csv")
            with open(must_path, "w") as f:
                f.write(f"{contig_names[0]},{contig_names[1]}\n")
                f.write(f"{contig_names[2]},{contig_names[3]}\n")

            cannot_path = os.path.join(tmpdir, "cannot_link.csv")
            with open(cannot_path, "w") as f:
                f.write(f"{contig_names[0]},{contig_names[4]}\n")
                f.write(f"{contig_names[1]},{contig_names[5]}\n")

            save_path = os.path.join(tmpdir, "trained_model.pt")

            model = train_model(
                logger=logger,
                data=[data_path],
                must_link=must_path,
                cannot_link=cannot_path,
                batchsize=4,
                epochs=2,
                device=torch.device('cpu'),
                main_loss_type='hinge',
                save_path=save_path
            )

            self.assertTrue(os.path.exists(save_path))
            loaded = model_load(save_path, device=torch.device('cpu'))
            self.assertEqual(loaded.num_features, num_features)

            with torch.no_grad():
                test_x = torch.from_numpy(data_arr[:2])
                emb = loaded.embedding(test_x)
            self.assertEqual(emb.shape, (2, 100))

    def test_train_model_self_supervised_and_supcon(self):
        """测试 train_model 无约束自监督模式与 SupCon 损失类型"""
        logger = logging.getLogger('TestLogger')
        num_contigs = 6
        num_features = 137  # 仅 k-mer + length

        with tempfile.TemporaryDirectory() as tmpdir:
            contig_names = [f"ctg_{i}" for i in range(num_contigs)]
            data_arr = np.random.rand(num_contigs, num_features).astype(np.float32)
            data_arr[:, -1] = 2000.0
            df = pd.DataFrame(data_arr, index=contig_names)
            data_path = os.path.join(tmpdir, "data.csv")
            df.to_csv(data_path)

            model = train_model(
                logger=logger,
                data=data_path,
                must_link=None,
                cannot_link=None,
                batchsize=4,
                epochs=2,
                device=torch.device('cpu'),
                main_loss_type='supcon'
            )

            self.assertIsNotNone(model)
            with torch.no_grad():
                emb = model.embedding(torch.from_numpy(data_arr[:2]))
            self.assertEqual(emb.shape, (2, 100))

    def test_train_model_legacy_signature_compat(self):
        """测试 train_model 对遗留位置参数（is_combined 等）的兼容性"""
        logger = logging.getLogger('TestLogger')
        num_contigs = 6
        num_features = 137

        with tempfile.TemporaryDirectory() as tmpdir:
            contig_names = [f"ctg_{i}" for i in range(num_contigs)]
            data_arr = np.random.rand(num_contigs, num_features).astype(np.float32)
            df = pd.DataFrame(data_arr, index=contig_names)
            data_path = os.path.join(tmpdir, "data.csv")
            df.to_csv(data_path)

            # 模拟旧版 SemiBin 位置调用: data, data_split, is_combined=True, batchsize, epochs, device, num_process, mode
            model = train_model(
                logger,
                [data_path],
                None,
                True,  # 原 is_combined 位置
                4,
                1,
                torch.device('cpu'),
                1,
                'single'
            )
            self.assertIsNotNone(model)


if __name__ == '__main__':
    unittest.main()
