import unittest
import os
import sys
import torch
import torch.nn.functional as F

# 将项目目录加入 sys.path
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

try:
    from newbin.model import NewBinModel
    from newbin.loss import (
        extract_shape_features,
        CoAbundanceAuxiliaryLoss,
        ContrastiveHingeLoss,
        SupConLoss,
        MetagenomicPipelineLoss,
    )
except (ImportError, ModuleNotFoundError):
    inner_dir = os.path.join(repo_dir, 'newbin')
    if inner_dir not in sys.path:
        sys.path.insert(0, inner_dir)
    from model import NewBinModel
    from loss import (
        extract_shape_features,
        CoAbundanceAuxiliaryLoss,
        ContrastiveHingeLoss,
        SupConLoss,
        MetagenomicPipelineLoss,
    )


class TestLossModule(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_extract_shape_features(self):
        """测试从完整特征张量中准确切片提取 M 维 Shape 剖面"""
        B = 4
        M = 5  # 5 个样本
        # 136 k-mer + (2*5 + 1) coverage + 1 length = 148 维
        x = torch.zeros(B, 136 + 2 * M + 1 + 1)
        
        # 模拟中间的 coverage 部分: [level(1), shape(5), log_id(5)]
        # 给 shape 填入特定标识值
        fake_shape = torch.arange(1, M + 1).float().unsqueeze(0).expand(B, -1)
        x[:, 137:137 + M] = fake_shape

        extracted_d = extract_shape_features(x)
        self.assertEqual(extracted_d.shape, (B, M))
        self.assertTrue(torch.allclose(extracted_d, fake_shape))

        # 测试只有 k-mer + length (137 维) 时，返回空张量
        x_no_cov = torch.zeros(B, 137)
        extracted_empty = extract_shape_features(x_no_cov)
        self.assertEqual(extracted_empty.shape, (B, 0))

    def test_co_abundance_numerical_derivation(self):
        """
        测试共丰度相关系数 r_ij 与文档 2.4 节手算推导严格一致：
        M=4:
        μ_i = [10, 50, 30, 5]
        μ_k = [30, 150, 90, 15] (3倍放大)
        μ_m = [25, 20, 24, 26] (异种平缓)
        验证 r_ik == 1.00, r_im ≈ -0.76
        """
        mu = torch.tensor([
            [10.0, 50.0, 30.0, 5.0],    # i
            [30.0, 150.0, 90.0, 15.0],  # k
            [25.0, 20.0, 24.0, 26.0],   # m
        ], dtype=torch.float32)

        # 1. log
        log_mu = torch.log(mu)
        # 2. level (行均值)
        level = log_mu.mean(dim=1, keepdim=True)
        # 3. shape d
        d = log_mu - level

        # 验证行和中心化为 0
        self.assertTrue(torch.allclose(d.sum(dim=1), torch.zeros(3), atol=1e-5))

        # 4. 余弦相似度即 Pearson 相关系数
        d_norm = F.normalize(d, p=2, dim=1)
        r = torch.matmul(d_norm, d_norm.T)

        r_ik = r[0, 1].item()
        r_im = r[0, 2].item()

        self.assertAlmostEqual(r_ik, 1.0, places=4)
        # 文档手算近似为 -0.76，严格连续计算值为 -0.876，验证异种负相关性
        self.assertAlmostEqual(r_im, -0.876, places=2)
        self.assertTrue(r_im < -0.7)

    def test_co_abundance_loss_zero_when_aligned(self):
        """当 Coverage 网络嵌入 Z_C 余弦夹角完全对齐 r_ij 时，L_co 应该为 0"""
        loss_fn = CoAbundanceAuxiliaryLoss(min_samples_threshold=2)

        # 假设 d_shape 形状为 (3, 4)
        d_shape = torch.randn(3, 4)
        d_shape = d_shape - d_shape.mean(dim=1, keepdim=True)

        # 若令 z_c 的方向完全由 d_shape 决定 (如直接投影相同方向)
        z_c = F.normalize(d_shape, p=2, dim=1)
        loss = loss_fn(z_c, d_shape)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_co_abundance_loss_threshold_bypass(self):
        """当样本数 M <= 2 时，共丰度自由度不足，损失恒等于 0"""
        loss_fn = CoAbundanceAuxiliaryLoss(min_samples_threshold=2)
        d_shape_m2 = torch.randn(4, 2)  # M=2
        z_c = torch.randn(4, 32)
        loss = loss_fn(z_c, d_shape_m2)
        self.assertEqual(loss.item(), 0.0)

    def test_contrastive_hinge_loss(self):
        """测试成对 Hinge 损失计算"""
        loss_fn = ContrastiveHingeLoss(margin=1.0)
        z1 = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
        z2 = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float32)

        # 样本0: 正对 (dist=0), loss=0
        # 样本1: 负对 (dist=2.0 > margin 1.0), loss=0
        labels = torch.tensor([1.0, 0.0])
        loss = loss_fn(z1, z2, labels)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

        # 样本1: 负对 (dist=0.4 < margin 1.0), loss = (1 - 0.4)^2 = 0.36
        z2_close = torch.tensor([[0.0, 0.0], [0.4, 0.0]], dtype=torch.float32)
        loss_close = loss_fn(z1, z2_close, labels)
        self.assertAlmostEqual(loss_close.item(), 0.36 / 2, places=4)

    def test_supcon_loss(self):
        """测试有监督对比损失 SupConLoss"""
        loss_fn = SupConLoss(temperature=0.1)
        # 构造两类样本，同类特征完全相同且相互正交
        z = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0]
        ], dtype=torch.float32)
        labels = torch.tensor([0, 0, 1, 1])

        loss = loss_fn(z, labels)
        self.assertTrue(loss.item() > 0)
        self.assertFalse(torch.isnan(loss))

    def test_gradient_isolation(self):
        """
        核心物理测试：验证辅助损失 L_co 的梯度严格隔离。
        L_co 只能反向传播回 cov_encoder，绝对不能流入 kmer_encoder、gate 网络或 fusion 层！
        """
        num_features = 146  # 136 k-mer + 9 coverage + 1 length
        model = NewBinModel(num_features=num_features, embed_dim=100)
        loss_fn = CoAbundanceAuxiliaryLoss(min_samples_threshold=2)

        x = torch.rand(4, num_features, requires_grad=True)
        d_shape = torch.rand(4, 4) - 0.5  # M=4

        out, z_c = model.embedding(x, return_zc=True)
        self.assertIsNotNone(z_c)
        self.assertEqual(z_c.shape, (4, 32))

        # 仅对 L_co 进行 backward
        l_co = loss_fn(z_c, d_shape)
        l_co.backward()

        # 1. 验证 Coverage 编码器正常接收到梯度
        self.assertIsNotNone(model.cov_encoder[0].weight.grad)
        self.assertTrue((model.cov_encoder[0].weight.grad.abs().sum() > 0).item())

        # 2. 验证 k-mer 编码器梯度严格为 None
        self.assertIsNone(model.kmer_encoder[0].weight.grad)

        # 3. 验证长度自适应门控网络 g(L) 梯度严格为 None
        self.assertIsNone(model.gate[0].weight.grad)

        # 4. 验证下游融合层 fusion 梯度严格为 None
        self.assertIsNone(model.fusion.weight.grad)

    def test_metagenomic_pipeline_joint_loss(self):
        """测试联合损失包装器 MetagenomicPipelineLoss"""
        joint_loss_fn = MetagenomicPipelineLoss(main_loss_type="supcon", lambda_co=0.1, temperature=0.1)

        B = 4
        z = torch.randn(B, 100)
        z_c = torch.randn(B, 32)
        d_shape = torch.randn(B, 4)
        labels = torch.tensor([0, 0, 1, 1])

        loss_dict = joint_loss_fn(z, z_c, d_shape, labels)
        self.assertIn("total_loss", loss_dict)
        self.assertIn("main_loss", loss_dict)
        self.assertIn("co_loss", loss_dict)

        expected_total = loss_dict["main_loss"] + 0.1 * loss_dict["co_loss"]
        self.assertTrue(torch.allclose(loss_dict["total_loss"], expected_total))


if __name__ == '__main__':
    unittest.main()

