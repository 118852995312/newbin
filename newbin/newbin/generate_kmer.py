# Adapted from https://github.com/BinPro/CONCOCT/blob/develop/scripts/fasta_to_features.py
from itertools import product
from collections import OrderedDict
import numpy as np
import pandas as pd
from .fasta import fasta_iter

def generate_feature_mapping(kmer_len):
    """
    生成 k-mer 到特征索引的映射字典及总特征数。

    考虑到 DNA 双链反向互补对称性，将每个 k-mer 及其反向互补序列映射到相同的特征索引上
    （即合并 canonical k-mer），从而减少特征维度并保持正反链一致性。

    Parameters
    ----------
    kmer_len : int
        k-mer 的长度（例如 4 表示 4-mer / tetranucleotide）。

    Returns
    -------
    kmer_hash : dict
        k-mer 字符串到特征索引的映射字典。
        每个 k-mer 与其反向互补序列对应相同的整数索引。
    counter : int
        独立 k-mer 特征的总数量（即特征向量的维度）。
    """
    BASE_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}
    kmer_hash = {}
    counter = 0
    # 遍历所有由 A, T, G, C 组成的长度为 kmer_len 的组合
    for kmer in product("ATGC", repeat=kmer_len):
        kmer = ''.join(kmer)
        if kmer not in kmer_hash:
            # 分配当前特征索引
            kmer_hash[kmer] = counter
            # 计算反向互补序列并映射到相同索引
            rev_compl = tuple([BASE_COMPLEMENT[x] for x in reversed(kmer)])
            kmer_hash[''.join(rev_compl)] = counter
            counter += 1
    return kmer_hash, counter


def generate_kmer_features_from_fasta(fasta_file, length_threshold, kmer_len):
    """
    从 FASTA 文件中提取每条序列的归一化 k-mer 频率特征。

    读取 FASTA 序列，过滤掉长度低于阈值的序列，统计每条序列的 k-mer 频次（合并反向互补序列），
    并加入微小伪计数（1e-5）后按行进行归一化（使每条序列的频率总和为 1）。

    Parameters
    ----------
    fasta_file : str or Path
        输入的 FASTA 格式文件路径（支持 .gz、.bz2、.xz 等压缩格式）。
    length_threshold : int
        序列最小长度阈值，长度小于该值的序列将被忽略。
    kmer_len : int
        k-mer 的长度（例如 4）。

    Returns
    -------
    df : pandas.DataFrame
        归一化后的 k-mer 频率特征表。
        行索引（index）为序列 ID / contig 名称，列为特征索引，取值为归一化后的频率（浮点型）。
    """
    # 生成 k-mer 索引映射与特征维度
    kmer_dict, nr_features = generate_feature_mapping(kmer_len)
    composition = OrderedDict()
    
    # 遍历 FASTA 文件中的序列
    for h, seq in fasta_iter(fasta_file):
        # 过滤长度小于阈值的序列
        if len(seq) < length_threshold:
            continue
        norm_seq = str(seq).upper()
        # 滑动窗口提取 k-mer 对应的特征索引，忽略含非标准碱基（如 N）的 k-mer
        kmers = [kmer_dict[norm_seq[i:i+kmer_len]]
                for i in range(len(norm_seq) - kmer_len + 1)
                if norm_seq[i:i+kmer_len] in kmer_dict]
        # 统计各特征索引出现的频次，长度不足 nr_features 则补 0
        composition[h] = np.bincount(np.array(kmers, dtype=np.int64), minlength=nr_features)
    
    # 构建 DataFrame，行索引为 contig 名称
    df = pd.DataFrame.from_dict(composition, orient='index', dtype=float)

    # 添加伪计数，避免 0 频率
    df = df.apply(lambda x: x + 1e-5)
    # 按行归一化，得到频率分布
    df = df.div(df.sum(axis=1), axis=0)
    return df

