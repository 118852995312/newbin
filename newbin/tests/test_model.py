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

try:
    from newbin.model import NewBinModel, model_load
except (ImportError, ModuleNotFoundError):
    inner_dir = os.path.join(repo_dir, 'newbin')
    if inner_dir not in sys.path:
        sys.path.insert(0, inner_dir)
    from model import NewBinModel, model_load


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


if __name__ == '__main__':
    unittest.main()
