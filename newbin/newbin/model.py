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

import os
import logging
from typing import Optional, List, Tuple, Union, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
#  Dataset, 采样器与训练循环实现
# ===========================================================================

class PairwiseContigDataset(Dataset):
    """
    成对 Contig 训练数据集（延迟索引加载，节省显存与内存）。

    Parameters
    ----------
    features : np.ndarray, shape (N, D)
        全量 contig 特征矩阵（存放于 CPU 内存中）
    pairs : np.ndarray, shape (P, 2)
        成对索引数组 [[idx_a, idx_b], ...]
    labels : np.ndarray, shape (P,)
        标签数组 (1.0 为 must-link, 0.0 为 cannot-link)
    """
    def __init__(self, features: np.ndarray, pairs: np.ndarray, labels: np.ndarray):
        self.features = features
        self.pairs = pairs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i, j = self.pairs[idx]
        x1 = torch.from_numpy(self.features[i])
        x2 = torch.from_numpy(self.features[j])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x1, x2, label


def sample_balanced_pairs(
    num_contigs: int,
    must_pairs: List[Tuple[int, int]],
    cannot_pairs: List[Tuple[int, int]],
    neg_ratio: int = 2,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    负采样平衡器：控制每个 Epoch 的正负对比例维持在 1 : neg_ratio。

    Parameters
    ----------
    num_contigs : int
        特征表中的 contig 总数 N
    must_pairs : list of (int, int)
        清洗后的先验正对（Must-link）索引列表
    cannot_pairs : list of (int, int)
        清洗后的先验负对（Cannot-link）索引列表
    neg_ratio : int
        每个正样本对对应的负样本对数量（默认 2，即 1:2）
    rng : np.random.Generator, optional
        随机数生成器

    Returns
    -------
    balanced_pairs : np.ndarray, shape (P, 2)
        配对的 contig 索引数组
    balanced_labels : np.ndarray, shape (P,)
        对应的二分类标签 (1.0 为正对，0.0 为负对)
    """
    if rng is None:
        rng = np.random.default_rng()

    if num_contigs < 2:
        raise ValueError(f"contig 总数必须大于等于 2 才能进行成对训练，当前为 {num_contigs}。")

    # 1. 确定正样本对
    if len(must_pairs) > 0:
        pos_pairs = np.array(must_pairs, dtype=np.int64)
    else:
        # 自监督兜底机制：若未提供任何外部 Must-link 先验，
        # 将各 contig 与自身配对作为正样本对。
        # 在模型训练模式 (model.train) 下，Dropout(0.2) 会对同一特征生成两个不同的扰动视图 (SimCLR/SimCSE 范式)。
        pos_pairs = np.stack([
            np.arange(num_contigs, dtype=np.int64),
            np.arange(num_contigs, dtype=np.int64)
        ], axis=1)

    n_pos = len(pos_pairs)
    n_neg_target = max(1, n_pos * max(1, neg_ratio))

    # 2. 负样本对采样与平衡
    if len(cannot_pairs) >= n_neg_target:
        # 先验负对充足，无放回随机抽取目标数量
        chosen_idx = rng.choice(len(cannot_pairs), size=n_neg_target, replace=False)
        neg_pairs = np.array(cannot_pairs, dtype=np.int64)[chosen_idx]
    else:
        # 先验负对不足或为空，全取已有先验负对，其余用全库随机配对补齐
        sampled_prior = [list(p) for p in cannot_pairs]
        needed = n_neg_target - len(sampled_prior)
        idx1 = rng.choice(num_contigs, size=needed, replace=True)
        # 错位加偏移量以确保 idx1 != idx2
        offset = 1 + rng.choice(num_contigs - 1, size=needed, replace=True)
        idx2 = (idx1 + offset) % num_contigs
        rand_neg = np.stack([idx1, idx2], axis=1).tolist()
        all_neg = sampled_prior + rand_neg
        neg_pairs = np.array(all_neg, dtype=np.int64)

    # 3. 组合正负对并打乱
    all_pairs = np.concatenate([pos_pairs, neg_pairs], axis=0)
    all_labels = np.concatenate([
        np.ones(len(pos_pairs), dtype=np.float32),
        np.zeros(len(neg_pairs), dtype=np.float32)
    ], axis=0)

    perm = rng.permutation(len(all_pairs))
    return all_pairs[perm], all_labels[perm]


def train_model(
    logger,
    data: Union[str, List[str]],
    must_link: Optional[Union[str, List[str]]] = None,
    cannot_link: Optional[Union[str, List[str]]] = None,
    batchsize: int = 2048,
    epochs: int = 15,
    device: Optional[torch.device] = None,
    num_process: int = 1,
    mode: str = 'single',
    main_loss_type: str = 'hinge',
    lambda_co: float = 0.1,
    neg_ratio: int = 2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    margin: float = 1.0,
    temperature: float = 0.1,
    save_path: Optional[str] = None,
    **kwargs
) -> NewBinModel:
    """
    NewBin 模型训练主函数。

    Parameters
    ----------
    logger : logging.Logger
        日志记录器
    data : str or list[str]
        data.csv 特征文件路径或路径列表
    must_link : str or list[str], optional
        must_link.csv 约束文件路径（正对）
    cannot_link : str or list[str], optional
        cannot_link.csv 约束文件路径（负对）
    batchsize : int, default 2048
        批次大小
    epochs : int, default 15
        训练轮数
    device : torch.device, optional
        计算设备 (CPU / CUDA)
    num_process : int, default 1
        PyTorch 线程数
    mode : str, default 'single'
        训练模式 ('single' 或 'several')
    main_loss_type : str, default 'hinge'
        主损失函数类型 ('hinge' 或 'supcon')
    lambda_co : float, default 0.1
        共丰度辅助损失权重 λ
    neg_ratio : int, default 2
        正负样本对比例 (1 : neg_ratio)
    lr : float, default 1e-3
        初始学习率
    weight_decay : float, default 1e-4
        权重衰减
    margin : float, default 1.0
        Hinge 损失间隔阈值
    temperature : float, default 0.1
        SupCon 损失温度超参数
    save_path : str, optional
        训练完成后权重保存目标路径 (.pt)
    **kwargs : dict
        兼容遗留参数 (例如 is_combined, data_split 等)

    Returns
    -------
    NewBinModel
        训练完毕的模型实例
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 兼容处理：若第 4 个位置参数传入了布尔值 (遗留的 is_combined)
    if isinstance(cannot_link, bool):
        kwargs['is_combined'] = cannot_link
        cannot_link = kwargs.get('cannot_link_file', None)

    if must_link is None and 'must_link' in kwargs:
        must_link = kwargs['must_link']
    if cannot_link is None and 'cannot_link' in kwargs:
        cannot_link = kwargs['cannot_link']

    # ---------------------------------------------------------------
    # 1. 读取训练特征数据与先验约束
    # ---------------------------------------------------------------
    if isinstance(data, str):
        data = [data]

    if mode == 'single':
        train_df = pd.read_csv(data[0], index_col=0)
    else:
        train_df = pd.concat([pd.read_csv(f, index_col=0) for f in data])

    train_df.index = train_df.index.astype(str)
    train_data = train_df.values.astype(np.float32)
    contig2idx = {name: idx for idx, name in enumerate(train_df.index)}
    num_features = train_data.shape[1]

    if len(train_data) < 2:
        raise ValueError(f"特征表中的 contig 数量必须大于等于 2，当前仅为 {len(train_data)}。")

    # 1.2 读取 must-link (正对) 约束
    must_pairs = []
    if must_link:
        must_files = [must_link] if isinstance(must_link, str) else must_link
        for mf in must_files:
            if mf and isinstance(mf, str) and os.path.exists(mf):
                try:
                    raw_must = pd.read_csv(mf, header=None).values
                    if raw_must.ndim == 2 and raw_must.shape[1] >= 2:
                        for row in raw_must:
                            c1, c2 = str(row[0]).strip(), str(row[1]).strip()
                            if c1 in contig2idx and c2 in contig2idx and c1 != c2:
                                must_pairs.append((contig2idx[c1], contig2idx[c2]))
                except Exception as e:
                    logger.warning(f"读取 must_link 文件 {mf} 出错: {e}")

    # 1.3 读取 cannot-link (负对) 约束
    cannot_pairs = []
    if cannot_link:
        cannot_files = [cannot_link] if isinstance(cannot_link, str) else cannot_link
        for cf in cannot_files:
            if cf and isinstance(cf, str) and os.path.exists(cf):
                try:
                    raw_cannot = pd.read_csv(cf, header=None).values
                    if raw_cannot.ndim == 2 and raw_cannot.shape[1] >= 2:
                        for row in raw_cannot:
                            c1, c2 = str(row[0]).strip(), str(row[1]).strip()
                            if c1 in contig2idx and c2 in contig2idx and c1 != c2:
                                cannot_pairs.append((contig2idx[c1], contig2idx[c2]))
                except Exception as e:
                    logger.warning(f"读取 cannot_link 文件 {cf} 出错: {e}")

    # 1.4 无向约束去重规范化
    must_pairs = list(set([tuple(sorted(p)) for p in must_pairs]))
    cannot_pairs = list(set([tuple(sorted(p)) for p in cannot_pairs]))

    logger.info(f"特征矩阵加载完成: contig数={len(train_df)}, 特征维度={num_features}")
    if len(must_pairs) > 0:
        logger.info(f"先验约束解析完成: 有效Must-link对数={len(must_pairs)}, 有效Cannot-link对数={len(cannot_pairs)}")
    else:
        logger.info(f"未检测到外部 Must-link 先验，启用自监督扰动对构建正样本。有效Cannot-link对数={len(cannot_pairs)}")

    # ---------------------------------------------------------------
    # 2. 初始化网络、联合损失与优化器
    # ---------------------------------------------------------------
    torch.set_num_threads(max(1, num_process))
    model = NewBinModel(num_features=num_features).to(device)

    criterion = MetagenomicPipelineLoss(
        main_loss_type=main_loss_type,
        lambda_co=lambda_co,
        margin=margin,
        temperature=temperature
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-5)

    rng = np.random.default_rng(42)

    logger.info(f"开始训练模型: 轮数={epochs}, 批大小={batchsize}, 损失类型={main_loss_type}, 设备={device}")

    # ---------------------------------------------------------------
    # 3. 训练主循环
    # ---------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        # 每个 epoch 重新均衡采样负对并打乱
        epoch_pairs, epoch_labels = sample_balanced_pairs(
            num_contigs=len(train_data),
            must_pairs=must_pairs,
            cannot_pairs=cannot_pairs,
            neg_ratio=neg_ratio,
            rng=rng
        )

        dataset = PairwiseContigDataset(train_data, epoch_pairs, epoch_labels)
        actual_batch_size = max(1, min(batchsize, len(dataset)))
        drop_last = len(dataset) > actual_batch_size

        dataloader = DataLoader(
            dataset,
            batch_size=actual_batch_size,
            shuffle=True,
            drop_last=drop_last,
            num_workers=0
        )

        model.train()
        total_loss_accum = 0.0
        main_loss_accum = 0.0
        co_loss_accum = 0.0
        num_batches = 0

        for x1, x2, labels in dataloader:
            x1 = x1.to(device=device, dtype=torch.float32)
            x2 = x2.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)

            (z1, z2), (zc1, zc2) = model(x1, x2, return_zc=True)
            d_shape1 = extract_shape_features(x1)
            d_shape2 = extract_shape_features(x2)

            loss_dict = criterion.forward_pairwise(
                z1=z1, z2=z2,
                zc1=zc1, zc2=zc2,
                d_shape1=d_shape1, d_shape2=d_shape2,
                labels=labels
            )

            loss = loss_dict['total_loss']
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss_accum += loss.item()
            m_loss = loss_dict['main_loss']
            c_loss = loss_dict['co_loss']
            main_loss_accum += m_loss.item() if isinstance(m_loss, torch.Tensor) else m_loss
            co_loss_accum += c_loss.item() if isinstance(c_loss, torch.Tensor) else c_loss
            num_batches += 1

        scheduler.step()

        if num_batches > 0:
            avg_total = total_loss_accum / num_batches
            avg_main = main_loss_accum / num_batches
            avg_co = co_loss_accum / num_batches
            cur_lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] - "
                f"Total Loss: {avg_total:.4f} (Main: {avg_main:.4f}, Co: {avg_co:.4f}) - "
                f"LR: {cur_lr:.6f}"
            )

    model.eval()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        model.save_with_params_to(save_path)
        logger.info(f"模型权重已保存至: {save_path}")

    logger.info("模型训练完成。")
    return model
