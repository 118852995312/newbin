"""
宏基因组分箱联合损失函数模块
============================

根据《宏基因组分箱模型：损失函数计算与代码实现手册》实现：
1. 主损失函数 L_main：
   - ContrastiveHingeLoss: 成对 Hinge 对比损失 (工作在欧氏距离空间)
   - SupConLoss: 有监督对比损失 / InfoNCE (工作在归一化超球面)
2. 共丰度辅助损失 L_co (CoAbundanceAuxiliaryLoss)：
   - 约束 Coverage 小网络输出 Z_C 在余弦空间上对齐原始跨样本对数 Shape 剖面 d 的皮尔逊相关系数 r_ij
3. 联合损失包装器 MetagenomicPipelineLoss：
   - L_total = L_main + lambda_co * L_co
   - 保证梯度隔离：L_co 仅流入 Coverage 分支，不污染 k-mer 与门控网络 g(l)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_shape_features(x: torch.Tensor) -> torch.Tensor:
    """
    从输入特征张量 x 中提取对数中心化 Shape 剖面 d_shape (M 维)。

    输入特征结构（方式 1）:
        - [0..135] (136 维): k-mer 特征
        - [136..136+2M] (2M+1 维): Coverage 特征:
            - 第 0 维: cov_level (1 维)
            - 第 1..M 维: cov_shape (M 维, 内部行均值恒为 0)
            - 第 M+1..2M 维: cov_log_id (M 维)
        - [-1] (1 维): contig 序列长度 length

    Parameters
    ----------
    x : torch.Tensor, shape [batch_size, num_features]

    Returns
    -------
    torch.Tensor, shape [batch_size, M]
        若无 Coverage 特征或 M <= 0 则返回空张量 (B, 0)
    """
    if x.shape[1] <= 137:
        return torch.empty(x.size(0), 0, device=x.device, dtype=x.dtype)

    x_cov = x[:, 136:-1]
    num_cov = x_cov.shape[1]
    if num_cov < 3:  # 2M + 1 且 M >= 1，最少为 3 维
        return torch.empty(x.size(0), 0, device=x.device, dtype=x.dtype)

    num_samples = (num_cov - 1) // 2
    # shape 列位于 level 之后，宽度为 num_samples (M 维)
    d_shape = x_cov[:, 1:1 + num_samples]
    return d_shape


class CoAbundanceAuxiliaryLoss(nn.Module):
    """
    共丰度辅助损失函数 L_co:
    强制 Coverage 小网络输出 Z_C 在余弦空间上复现输入的真实对数皮尔逊相关系数 r_ij。

    数学公式:
        L_co = (1 / |mask|) * sum_{(i != j)} ( cos(Z_C_i, Z_C_j) - r_ij )^2
    其中 r_ij = cos(d_i, d_j) 即预处理后的 Shape 剖面的余弦相似度。
    """

    def __init__(self, min_samples_threshold: int = 2):
        super().__init__()
        self.min_samples_threshold = min_samples_threshold

    def forward(self, z_c: torch.Tensor, d_shape: torch.Tensor) -> torch.Tensor:
        """
        Batch 矩阵形式的共丰度辅助损失计算。

        Parameters
        ----------
        z_c : torch.Tensor, shape (B, D_c)
            Coverage 小网络输出特征 (例如 32 维)
        d_shape : torch.Tensor, shape (B, M)
            原始 Coverage Shape 剖面，M 为测序样本数

        Returns
        -------
        loss_co : torch.Tensor
            标量辅助损失
        """
        if z_c is None or d_shape is None:
            return torch.tensor(0.0, device=z_c.device if z_c is not None else torch.device('cpu'))

        batch_size, num_samples = d_shape.shape

        # 样本数 <= 2 时 shape 特征自由度过低或退化，不施加辅助约束
        if num_samples <= self.min_samples_threshold or batch_size <= 1:
            return torch.tensor(0.0, device=z_c.device, dtype=z_c.dtype)

        # 1. 计算真实目标相关系数矩阵 r (B, B)
        # d_shape 已中心化 (均值为 0)，L2 归一化后做点积严格等价于 Pearson 相关系数
        d_norm = F.normalize(d_shape.detach(), p=2, dim=1)
        r_target = torch.matmul(d_norm, d_norm.T)  # (B, B)

        # 2. 计算网络预测相似度矩阵 s (B, B)
        zc_norm = F.normalize(z_c, p=2, dim=1)
        s_pred = torch.matmul(zc_norm, zc_norm.T)  # (B, B)

        # 3. 计算均方误差 (排除自身对角线元素)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=z_c.device)
        loss_co = torch.mean((s_pred[mask] - r_target[mask]) ** 2)

        return loss_co

    def forward_pairwise(
        self,
        zc1: torch.Tensor,
        zc2: torch.Tensor,
        d_shape1: torch.Tensor,
        d_shape2: torch.Tensor
    ) -> torch.Tensor:
        """
        成对输入模式下的共丰度辅助损失 (用于孪生配对样本)。

        Parameters
        ----------
        zc1, zc2 : torch.Tensor, shape (B, D_c)
        d_shape1, d_shape2 : torch.Tensor, shape (B, M)

        Returns
        -------
        loss_co : torch.Tensor
            标量辅助损失
        """
        if zc1 is None or zc2 is None or d_shape1 is None or d_shape2 is None:
            return torch.tensor(0.0, device=zc1.device if zc1 is not None else torch.device('cpu'))

        batch_size, num_samples = d_shape1.shape
        if num_samples <= self.min_samples_threshold or batch_size == 0:
            return torch.tensor(0.0, device=zc1.device, dtype=zc1.dtype)

        d1_norm = F.normalize(d_shape1.detach(), p=2, dim=1)
        d2_norm = F.normalize(d_shape2.detach(), p=2, dim=1)
        r_target = torch.sum(d1_norm * d2_norm, dim=-1)  # (B,)

        zc1_norm = F.normalize(zc1, p=2, dim=1)
        zc2_norm = F.normalize(zc2, p=2, dim=1)
        s_pred = torch.sum(zc1_norm * zc2_norm, dim=-1)  # (B,)

        loss_co = torch.mean((s_pred - r_target) ** 2)
        return loss_co


class ContrastiveHingeLoss(nn.Module):
    """
    成对对比铰链损失 (Contrastive Hinge Loss / 对齐基线):
    工作在欧氏距离空间，优化目标：
        L = (1 / |P|) * sum [ y * d(z1, z2)^2 + (1 - y) * max(0, margin - d(z1, z2))^2 ]
    其中 y = 1 为 must-link 正样本对，y = 0 为 cannot-link 负样本对。
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z1, z2 : torch.Tensor, shape (B, D)
            两组样本的最终嵌入特征向量
        labels : torch.Tensor, shape (B,)
            1.0 表示 must-link (正对)，0.0 表示 cannot-link (负对)
        """
        dist = torch.norm(z1 - z2, p=2, dim=-1)
        loss_pos = labels * (dist ** 2)
        loss_neg = (1.0 - labels) * (F.relu(self.margin - dist) ** 2)
        return torch.mean(loss_pos + loss_neg)


class SupConLoss(nn.Module):
    """
    有监督对比损失 (Supervised Contrastive Loss / SupCon / InfoNCE):
    在超球面上基于 Softmax 全局互斥场拉近同类别样本、推远异类样本。
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : torch.Tensor, shape (B, D)
            最终融合 Embedding (方法内部自动执行 L2 归一化)
        labels : torch.Tensor, shape (B,)
            物种/类别的整数标签
        """
        device = z.device
        batch_size = z.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        z_norm = F.normalize(z, p=2, dim=1)

        # 构造正样本掩码: labels 相同且非对角线自身
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        mask = mask.masked_fill(diag_mask, 0.0)

        # 相似度矩阵 / tau
        sim = torch.matmul(z_norm, z_norm.T) / self.temperature

        # 数值稳定性：减去每行最大值
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        # 分母 log-sum-exp 排除自身
        logits_mask = (~diag_mask).float()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # 计算正样本均值似然
        pos_per_row = mask.sum(1)
        valid_rows = pos_per_row > 0

        if not valid_rows.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (mask * log_prob).sum(1)[valid_rows] / pos_per_row[valid_rows]
        loss = -mean_log_prob_pos.mean()
        return loss


class MetagenomicPipelineLoss(nn.Module):
    """
    模型整体联合损失包装器:
        L_total = L_main + lambda_co * L_co
    """

    def __init__(
        self,
        main_loss_type: str = "supcon",
        lambda_co: float = 0.1,
        temperature: float = 0.1,
        margin: float = 1.0,
        min_samples_threshold: int = 2
    ):
        super().__init__()
        self.lambda_co = lambda_co
        self.main_loss_type = main_loss_type.lower()

        if self.main_loss_type in ["supcon", "infonce"]:
            self.main_loss_fn = SupConLoss(temperature=temperature)
        elif self.main_loss_type in ["hinge", "contrastive"]:
            self.main_loss_fn = ContrastiveHingeLoss(margin=margin)
        else:
            raise ValueError(f"未知的损失函数类型: {main_loss_type} (仅支持 'supcon' 或 'hinge')")

        self.co_loss_fn = CoAbundanceAuxiliaryLoss(min_samples_threshold=min_samples_threshold)

    def forward(
        self,
        z: torch.Tensor,
        z_c: torch.Tensor,
        d_shape: torch.Tensor,
        labels: torch.Tensor
    ) -> dict:
        """
        Batch 模式联合损失计算 (主要用于 SupCon 模式)。

        Parameters
        ----------
        z : torch.Tensor, shape (B, D)
            模型最终输出的 100 维 embedding
        z_c : torch.Tensor, shape (B, 32)
            Coverage 小网络输出特征
        d_shape : torch.Tensor, shape (B, M)
            输入中对应的原始 Shape 剖面
        labels : torch.Tensor, shape (B,)
            物种类别标签

        Returns
        -------
        dict
            {"total_loss": ..., "main_loss": ..., "co_loss": ...}
        """
        l_main = self.main_loss_fn(z, labels)
        l_co = self.co_loss_fn(z_c, d_shape)
        total_loss = l_main + self.lambda_co * l_co

        return {
            "total_loss": total_loss,
            "main_loss": l_main,
            "co_loss": l_co
        }

    def forward_pairwise(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        zc1: torch.Tensor,
        zc2: torch.Tensor,
        d_shape1: torch.Tensor,
        d_shape2: torch.Tensor,
        labels: torch.Tensor
    ) -> dict:
        """
        孪生成对模式联合损失计算 (用于 Hinge 对比学习)。

        Parameters
        ----------
        z1, z2 : torch.Tensor, shape (B, D)
        zc1, zc2 : torch.Tensor, shape (B, 32)
        d_shape1, d_shape2 : torch.Tensor, shape (B, M)
        labels : torch.Tensor, shape (B,), 1 为 must-link, 0 为 cannot-link
        """
        if self.main_loss_type in ["hinge", "contrastive"]:
            l_main = self.main_loss_fn(z1, z2, labels)
        else:
            # 若成对模式下使用 SupCon，将两组特征与伪标签拼入单一 Batch
            z_all = torch.cat([z1, z2], dim=0)
            # 生成伪分组标签: 同一索引配对标签相同
            b = z1.size(0)
            pair_ids = torch.arange(b, device=z1.device)
            # 对于正对 (labels==1)，共享相同的类别 ID；对于负对，分配不同的类别 ID
            id1 = pair_ids
            id2 = torch.where(labels.long() == 1, pair_ids, pair_ids + b)
            labels_all = torch.cat([id1, id2], dim=0)
            l_main = self.main_loss_fn(z_all, labels_all)

        l_co = self.co_loss_fn.forward_pairwise(zc1, zc2, d_shape1, d_shape2)
        total_loss = l_main + self.lambda_co * l_co

        return {
            "total_loss": total_loss,
            "main_loss": l_main,
            "co_loss": l_co
        }

