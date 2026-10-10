"""
NewBin 模型定义文件（待实现）
===========================

本文件是 NewBin 的模型搭建入口，用于替代 SemiBin 中的：
  - semi_supervised_model.py  (孪生网络架构 + 半监督训练)
  - self_supervised_model.py  (自监督对比学习训练循环)

需要实现的接口
--------------
所有模型类必须满足以下接口，才能被 cluster.py / long_read_cluster.py 使用：

    model.eval()                        # 切换推理模式
    model.embedding(x: torch.Tensor)    # 返回嵌入向量，shape [batch, embed_dim]
    model.save_with_params_to(path)     # 保存模型到磁盘

同时，训练函数 train_model() 需要返回满足上述接口的模型对象。

data.csv 的特征维度
------------------
    - is_combined=False（BAM < 5）：只用前 136 列 k-mer，输入维度为 136
    - is_combined=True （BAM ≥ 5）：k-mer + coverage 全表，输入维度为 136 + N

参考 SemiBin 原架构（可保留或完全替换）
----------------------------------------
    encoder: Linear(num -> 512) -> BatchNorm1d -> LeakyReLU -> Dropout(0.2)
             Linear(512 -> 512) -> BatchNorm1d -> LeakyReLU -> Dropout(0.2)
             Linear(512 -> 100)   # 100 维嵌入
"""

import torch
import torch.nn as nn
import logging

from .loss import (
    extract_shape_features,
    CoAbundanceAuxiliaryLoss,
    ContrastiveHingeLoss,
    SupConLoss,
    MetagenomicPipelineLoss,
)

logger = logging.getLogger('NewBin')


# ===========================================================================
#  TODO: 在这里实现你自己的模型架构
# ===========================================================================

class NewBinModel(nn.Module):
    """
    NewBin 双分支特征编码模型（k-mer 门控编码 + Coverage 编码 + Linear 融合网络）。

    架构特点：
    1. k-mer 分支：两层 MLP（Linear -> BatchNorm1d -> LeakyReLU -> Dropout(0.2) -> Linear），输出 64 维 z_embedding。
    2. 长度门控分支：输入 contig 长度 L，通过 log(1+L) -> Linear(1, 1) -> Sigmoid 得到门控标量 g(1)，
       与 64 维 z_embedding 进行逐元素相乘调制 (g ⊙ z_embedding)。
    3. Coverage 分支：两层 MLP（Linear -> BatchNorm1d -> LeakyReLU -> Dropout(0.2) -> Linear），
       将 2M+1 维 Coverage 编码为 32 维向量 z_C。
    4. 融合网络：拼接门控 k-mer 嵌入 (64 维) 与 Coverage 嵌入 (32 维) 得到 96 维，
       通过单层 Linear 融合输出 100 维的最终嵌入向量。

    Parameters
    ----------
    num_features : int
        输入总特征维度（136 维 k-mer + 可选的 Coverage 维度 + 1 维 length）。
    kmer_hidden_dim : int, optional
        k-mer MLP 隐层维度，默认为 256。
    cov_hidden_dim : int, optional
        Coverage MLP 隐层维度，默认为 64。
    embed_dim : int, optional
        最终输出嵌入向量维度，默认 100。
    """

    def __init__(self, num_features: int, kmer_hidden_dim: int = 256, cov_hidden_dim: int = 64, embed_dim: int = 100):
        super().__init__()
        self.num_features = num_features
        self.kmer_hidden_dim = kmer_hidden_dim
        self.cov_hidden_dim = cov_hidden_dim
        self.embed_dim = embed_dim

        # ---------------------------------------------------------------
        # 1. k-mer 编码器分支 (输入 136 维 -> 隐层 -> 64 维 z_embedding)
        # ---------------------------------------------------------------
        self.kmer_encoder = nn.Sequential(
            nn.Linear(136, kmer_hidden_dim),
            nn.BatchNorm1d(kmer_hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.2),
            nn.Linear(kmer_hidden_dim, 64),
        )

        # ---------------------------------------------------------------
        # 2. 长度自适应门控网络 g(L)
        #    输入: log(1 + L) -> Linear(1, 1) -> Sigmoid -> 权重 g
        # ---------------------------------------------------------------
        self.gate = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid(),
        )

        # ---------------------------------------------------------------
        # 3. Coverage 编码器分支 (输入 2M+1 维 -> 隐层 -> 32 维 z_C)
        # ---------------------------------------------------------------
        # 总特征列数结构（方式 1）：前 136 列为 k-mer，最后一列为 length，中间为 coverage
        self.num_cov_features = max(0, num_features - 136 - 1)
        if self.num_cov_features > 0:
            self.cov_encoder = nn.Sequential(
                nn.Linear(self.num_cov_features, cov_hidden_dim),
                nn.BatchNorm1d(cov_hidden_dim),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(p=0.2),
                nn.Linear(cov_hidden_dim, 32),
            )
            # 融合网络：拼接 64 维 k-mer 与 32 维 coverage -> Linear(96 -> 100)
            self.fusion = nn.Linear(64 + 32, embed_dim)
        else:
            self.cov_encoder = None
            # 无 coverage 时，仅输入 64 维 k-mer -> Linear(64 -> 100)
            self.fusion = nn.Linear(64, embed_dim)

    def embedding(self, x: torch.Tensor, length: torch.Tensor = None, return_zc: bool = False):
        """
        生成嵌入向量（cluster.py / 推理使用，或训练时同时提取 Coverage 嵌入）。

        Parameters
        ----------
        x : torch.Tensor, shape [batch_size, num_features]
            特征张量。按方式 1 约定：前 136 列为 k-mer，最后一列为 contig 长度 length，中间为 coverage。
        length : torch.Tensor, optional, shape [batch_size, 1]
            若单独传入长度张量，则优先使用该参数。
        return_zc : bool, optional
            是否同时返回 Coverage 分支输出 z_c (32 维)。默认为 False。

        Returns
        -------
        out : torch.Tensor, shape [batch_size, embed_dim] (固定为 100 维)
            如果 return_zc=True，则返回元组 (out, z_c)
        """
        # 1. 解析 k-mer、Coverage 与 length 切片
        if length is not None:
            x_kmer = x[:, :136]
            x_cov = x[:, 136:]
            len_val = length.view(-1, 1).float()
        elif x.shape[1] > 136:
            x_kmer = x[:, :136]
            x_cov = x[:, 136:-1]
            len_val = x[:, -1:].float()
        else:
            x_kmer = x
            x_cov = torch.empty(x.size(0), 0, device=x.device)
            len_val = torch.full((x.size(0), 1), 1000.0, device=x.device, dtype=torch.float32)

        # 2. k-mer MLP 编码得到 64 维 z_embedding
        z_kmer = self.kmer_encoder(x_kmer)

        # 3. 门控网络：log(1 + L) -> Linear -> Sigmoid -> g(1)
        log_l = torch.log1p(torch.clamp(len_val, min=0.0))
        g = self.gate(log_l)

        # 4. 门控调制：g(1) * z_embedding(64)
        z_kmer_gated = z_kmer * g

        # 5. Coverage 编码与融合输出 100 维向量
        z_c = None
        if self.cov_encoder is not None and x_cov.shape[1] > 0:
            z_c = self.cov_encoder(x_cov)
            z_concat = torch.cat([z_kmer_gated, z_c], dim=-1)  # 64 + 32 = 96 维
            out = self.fusion(z_concat)                        # 96 -> 100 维
        else:
            out = self.fusion(z_kmer_gated)                    # 64 -> 100 维

        if return_zc:
            return out, z_c
        return out

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        len1: torch.Tensor = None,
        len2: torch.Tensor = None,
        return_zc: bool = False
    ):
        """
        孪生前向传播（训练对比学习时使用）

        Parameters
        ----------
        x1, x2 : torch.Tensor
        len1, len2 : torch.Tensor, optional
        return_zc : bool, optional
            若为 True，返回 ((emb1, emb2), (zc1, zc2))，供联合损失函数使用。
        """
        if return_zc:
            emb1, zc1 = self.embedding(x1, len1, return_zc=True)
            emb2, zc2 = self.embedding(x2, len2, return_zc=True)
            return (emb1, emb2), (zc1, zc2)
        return self.embedding(x1, len1), self.embedding(x2, len2)

    def save_with_params_to(self, path: str):
        """保存模型到磁盘（同时保存构建参数，方便重新加载）"""
        torch.save({
            'model_name': self.__class__.__name__,
            'model_state_dict': self.state_dict(),
            'params': (self.num_features, self.kmer_hidden_dim, self.cov_hidden_dim, self.embed_dim),
        }, path)
        logger.info(f'Model saved to {path}')


def model_load(path: str, device: torch.device) -> NewBinModel:
    """
    从磁盘加载模型

    Parameters
    ----------
    path : str
        模型文件路径（.pt）
    device : torch.device

    Returns
    -------
    NewBinModel（已移到 device 并切换为 eval 模式）
    """
    data = torch.load(path, map_location=device, weights_only=True)
    params = data['params']
    if len(params) == 2:
        num_features, embed_dim = params
        model = NewBinModel(num_features=num_features, embed_dim=embed_dim)
    elif len(params) == 3:
        num_features, kmer_hidden_dim, embed_dim = params
        model = NewBinModel(num_features=num_features, kmer_hidden_dim=kmer_hidden_dim, embed_dim=embed_dim)
    else:
        num_features, kmer_hidden_dim, cov_hidden_dim, embed_dim = params
        model = NewBinModel(num_features=num_features, kmer_hidden_dim=kmer_hidden_dim, cov_hidden_dim=cov_hidden_dim, embed_dim=embed_dim)

    model.load_state_dict(data['model_state_dict'])
    model.to(device)
    model.eval()
    return model


# ===========================================================================
#  TODO: 在这里实现你自己的训练函数
# ===========================================================================

def train_model(logger,
                data,           # list[str]  data.csv 路径列表
                must_link=None, # str 或 list[str]  must_link.csv 路径
                cannot_link=None, # str 或 list[str] cannot_link.csv 路径
                batchsize: int = 2048,
                epochs: int = 15,
                device: torch.device = None,
                num_process: int = 1,
                mode: str = 'single', # 'single' 或 'several'
                **kwargs        # 兼容其他调用参数 (如 is_combined, data_split 等)
                ) -> NewBinModel:
    """
    模型训练函数。

    Parameters
    ----------
    data : list[str]
        data.csv 特征文件路径列表（每个样本一个）
    must_link : str or list[str], optional
        must_link.csv 路径（正样本对）
    cannot_link : str or list[str], optional
        cannot_link.csv 路径（负样本对）
    batchsize : int
    epochs : int
    device : torch.device
    num_process : int
    mode : str
        'single' 表示单样本训练，'several' 表示多样本联合训练
    **kwargs : dict
        兼容参数

    Returns
    -------
    NewBinModel（训练完毕后的模型）
    """
    import os
    import numpy as np
    import pandas as pd

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------------------------------------------------------------
    # 1. 读取训练特征数据与先验约束
    # ---------------------------------------------------------------
    # 1.1 读取特征矩阵 data.csv
    if mode == 'single':
        train_df = pd.read_csv(data[0], index_col=0)
    else:
        train_df = pd.concat([pd.read_csv(f, index_col=0) for f in data])

    train_df.index = train_df.index.astype(str)
    train_data = train_df.values.astype(np.float32)
    contig2idx = {name: idx for idx, name in enumerate(train_df.index)}
    num_features = train_data.shape[1]

    # 1.2 读取 must-link (正对) 约束
    must_pairs = []
    if must_link:
        must_files = [must_link] if isinstance(must_link, str) else must_link
        for mf in must_files:
            if mf and os.path.exists(mf):
                raw_must = pd.read_csv(mf, header=None).values
                for c1, c2 in raw_must:
                    c1, c2 = str(c1).strip(), str(c2).strip()
                    if c1 in contig2idx and c2 in contig2idx and c1 != c2:
                        must_pairs.append((contig2idx[c1], contig2idx[c2]))

    # 1.3 读取 cannot-link (负对) 约束
    cannot_pairs = []
    if cannot_link:
        cannot_files = [cannot_link] if isinstance(cannot_link, str) else cannot_link
        for cf in cannot_files:
            if cf and os.path.exists(cf):
                raw_cannot = pd.read_csv(cf, header=None).values
                for c1, c2 in raw_cannot:
                    c1, c2 = str(c1).strip(), str(c2).strip()
                    if c1 in contig2idx and c2 in contig2idx and c1 != c2:
                        cannot_pairs.append((contig2idx[c1], contig2idx[c2]))

    # 1.4 无向约束去重规范化
    must_pairs = list(set([tuple(sorted(p)) for p in must_pairs]))
    cannot_pairs = list(set([tuple(sorted(p)) for p in cannot_pairs]))

    logger.info(f'特征矩阵加载完成: contig数={len(train_df)}, 特征维度={num_features}')
    logger.info(f'先验约束解析完成: 有效Must-link对数={len(must_pairs)}, 有效Cannot-link对数={len(cannot_pairs)}')

    # ---------------------------------------------------------------
    # 2. 初始化模型
    # ---------------------------------------------------------------
    model = NewBinModel(num_features=num_features).to(device)

    # ---------------------------------------------------------------
    # TODO: 3. 实现你的训练循环
    # ---------------------------------------------------------------
    # 参考 SemiBin self_supervised_model.py 的对比学习：
    #   - Cannot-link：每 epoch 随机错位采样负对（label=0）
    #   - Must-link：data_split 偶数行 & 奇数行配对（label=1）
    #   - 损失：Contrastive Loss = label * d² + (1-label) * relu(1-d)²
    #
    # 你可以自由替换为任何损失函数 / 训练策略。

    logger.warning('train_model() 尚未实现，请在 model.py 中填写训练逻辑。')

    return model
