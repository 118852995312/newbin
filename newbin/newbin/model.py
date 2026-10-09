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

logger = logging.getLogger('NewBin')


# ===========================================================================
#  TODO: 在这里实现你自己的模型架构
# ===========================================================================

class NewBinModel(nn.Module):
    """
    占位模型类 —— 请替换成你自己的网络架构。

    Parameters
    ----------
    num_features : int
        输入特征维度（136 或 136+N）
    embed_dim : int
        嵌入向量维度，默认 100（与 SemiBin 保持一致，以便复用聚类代码）
    """

    def __init__(self, num_features: int, embed_dim: int = 100):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim

        # ---------------------------------------------------------------
        # TODO: 替换为你自己的网络层
        # ---------------------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim),
        )

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        生成嵌入向量（cluster.py / long_read_cluster.py 调用此方法）

        Parameters
        ----------
        x : torch.Tensor, shape [batch_size, num_features]

        Returns
        -------
        torch.Tensor, shape [batch_size, embed_dim]
        """
        return self.encoder(x)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        孪生前向传播（训练时使用，返回两个嵌入向量）
        如果你的模型不需要孪生结构，可以修改此方法。
        """
        return self.embedding(x1), self.embedding(x2)

    def save_with_params_to(self, path: str):
        """保存模型到磁盘（同时保存构建参数，方便重新加载）"""
        torch.save({
            'model_name': self.__class__.__name__,
            'model_state_dict': self.state_dict(),
            'params': (self.num_features, self.embed_dim),
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
    num_features, embed_dim = data['params']
    model = NewBinModel(num_features, embed_dim)
    model.load_state_dict(data['model_state_dict'])
    model.to(device)
    model.eval()
    return model


# ===========================================================================
#  TODO: 在这里实现你自己的训练函数
# ===========================================================================

def train_model(logger,
                data,           # list[str]  data.csv 路径列表
                data_split,     # list[str]  data_split.csv 路径列表
                is_combined: bool,
                batchsize: int,
                epochs: int,
                device: torch.device,
                num_process: int,
                mode: str       # 'single' 或 'several'
                ) -> NewBinModel:
    """
    训练函数占位 —— 请在此实现你的训练逻辑。

    Parameters
    ----------
    data : list[str]
        data.csv 文件路径列表（每个样本一个）
    data_split : list[str]
        data_split.csv 文件路径列表
    is_combined : bool
        True 表示 BAM >= 5（k-mer + coverage），False 表示只用 k-mer
    batchsize : int
    epochs : int
    device : torch.device
    num_process : int
    mode : str
        'single' 表示单样本训练，'several' 表示多样本联合训练

    Returns
    -------
    NewBinModel（已训练完毕，可调用 save_with_params_to() 保存）
    """
    import numpy as np
    import pandas as pd

    # ---------------------------------------------------------------
    # 1. 读取训练数据
    # ---------------------------------------------------------------
    import os
    if mode == 'single':
        train_df = pd.read_csv(data[0], index_col=0)
    else:
        train_df = pd.concat([pd.read_csv(f, index_col=0) for f in data])

    train_split_df = None
    if data_split and len(data_split) > 0 and data_split[0] is not None and os.path.exists(data_split[0]):
        if mode == 'single':
            train_split_df = pd.read_csv(data_split[0], index_col=0)
        else:
            train_split_df = pd.concat([pd.read_csv(f, index_col=0) for f in data_split if os.path.exists(f)])

    if is_combined:
        train_data = train_df.values.astype(np.float32)
    else:
        train_data = train_df.values[:, :136].astype(np.float32)

    if train_split_df is not None:
        train_data_split = train_split_df.values.astype(np.float32)
        if not is_combined:
            train_data_split = train_data_split[:, :136]
    else:
        train_data_split = None

    num_features = train_data.shape[1]
    logger.info(f'Training NewBinModel: num_features={num_features}, '
                f'is_combined={is_combined}, epochs={epochs}, batchsize={batchsize}')

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
