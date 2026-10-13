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


def _build_lut_4mer(kmer_dict):
    """
    针对 4-mer 构建一个快速单字节查找表（LUT）。
    ASCII 碱基映射为 2-bit：A:0, C:1, G:2, T:3，非标准碱基标记为 255。
    4-mer 对应的 8-bit 整数 (0..255) 直接映射到 canonical 特征索引 (0..135)。
    """
    byte_map = np.full(256, 255, dtype=np.uint8)
    for b, code in [('A', 0), ('a', 0), ('C', 1), ('c', 1), ('G', 2), ('g', 2), ('T', 3), ('t', 3)]:
        byte_map[ord(b)] = code

    lut = np.full(256, -1, dtype=np.int32)
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for kmer, feat_id in kmer_dict.items():
        idx = (base_map[kmer[0]] << 6) | (base_map[kmer[1]] << 4) | (base_map[kmer[2]] << 2) | base_map[kmer[3]]
        lut[idx] = feat_id
    return byte_map, lut


def generate_kmer_features_from_fasta(
        fasta_file,
        length_threshold,
        kmer_len=4,
        split=False,
        split_threshold=0,
        contig_whitelist=None,
        return_lengths=False):
    """
    从 FASTA 文件中提取每条序列的归一化 k-mer 频率特征。

    读取 FASTA 序列，过滤掉长度低于阈值的序列，统计每条序列的 k-mer 频次（合并反向互补序列），
    并加入微小伪计数（1e-5）后按行进行归一化（使每条序列的频率总和为 1）。

    针对 4-mer（默认）引入 2-bit 位运算与查找表向量化加速；
    支持 contig_whitelist 白名单预过滤以针对性提速；
    矩阵归一化在 NumPy 层面原地完成，消除 Pandas apply 性能瓶颈。

    Parameters
    ----------
    fasta_file : str or Path or file-like
        输入的 FASTA 格式文件路径（支持 .gz、.bz2、.xz 等压缩格式）或类文件对象。
    length_threshold : int
        序列最小长度阈值，长度小于该值的序列将被忽略。
    kmer_len : int, optional
        k-mer 的长度（默认为 4）。
    split : bool, optional
        是否将长序列分割为两段（兼容历史接口，默认为 False）。
    split_threshold : int, optional
        当 split=True 时，序列长度达到该阈值才进行分割。
    contig_whitelist : set or container, optional
        可选的目标 contig 名称白名单。如果提供，不在白名单中的 contig 将被快速跳过。
    return_lengths : bool, optional
        是否同时返回序列长度字典 {contig_name: length}（默认为 False）。

    Returns
    -------
    df : pandas.DataFrame (当 return_lengths=False 时)
        归一化后的 k-mer 频率特征表。
    (df, contig_lens) : tuple (当 return_lengths=True 时)
        包含特征表与序列长度字典的元组。
    """
    # 生成 k-mer 索引映射与特征维度
    kmer_dict, nr_features = generate_feature_mapping(kmer_len)

    # 若 k=4 则预生成 2-bit 向量化查找表
    use_fast_4mer = (kmer_len == 4)
    if use_fast_4mer:
        byte_map, lut_256 = _build_lut_4mer(kmer_dict)  ##lut_256 是所有可能kmer计算的十进制数据对应的特征编号 byte_map 是kmer每个字符串对应ASCII的数字

    if contig_whitelist is not None and not isinstance(contig_whitelist, set):
        contig_whitelist = set(contig_whitelist)

    def seq_generator():
        for h, seq in fasta_iter(fasta_file):
            if contig_whitelist is not None and h not in contig_whitelist:
                continue
            if not split:
                yield h, seq
            elif len(seq) >= split_threshold:
                half = len(seq) // 2
                yield (f'{h}_1', seq[:half])
                yield (f'{h}_2', seq[half:])

    names = []
    counts_list = []
    lengths_list = []

    # 遍历 FASTA 序列流
    for h, seq in seq_generator():
        seq_len = len(seq)
        if seq_len < length_threshold:
            continue

        if use_fast_4mer:
            if seq_len < 4:
                cnt = np.zeros(nr_features, dtype=np.int64)
            else:
                seq_bytes = np.frombuffer(seq.encode('ascii', errors='ignore'), dtype=np.uint8)
                encoded = byte_map[seq_bytes]
                b0 = encoded[:-3].astype(np.int32)
                b1 = encoded[1:-2].astype(np.int32)
                b2 = encoded[2:-1].astype(np.int32)
                b3 = encoded[3:].astype(np.int32)
                # 过滤包含非标准碱基（如 N）的窗口
                valid = (b0 < 4) & (b1 < 4) & (b2 < 4) & (b3 < 4)  ##保证数据的有效性
                kmer_indices = (b0 << 6) | (b1 << 4) | (b2 << 2) | b3  ##b0，b1.b2,b3 相同位置进行二进制编码，然后计算出十进制的结果
                feat_ids = lut_256[kmer_indices[valid]]
                ##统计 0∼135 这 136 种 4-mer 特征在当前这条 Contig 序列中各自出现了多少次（频数统计/唱票）。
                cnt = np.bincount(feat_ids, minlength=nr_features)
        else:
            norm_seq = str(seq).upper()
            kmers = [kmer_dict[norm_seq[i:i+kmer_len]]
                    for i in range(len(norm_seq) - kmer_len + 1)
                    if norm_seq[i:i+kmer_len] in kmer_dict]
            cnt = np.bincount(np.array(kmers, dtype=np.int64), minlength=nr_features)

        names.append(h)
        counts_list.append(cnt)
        lengths_list.append(seq_len)

    if not names:
        empty_df = pd.DataFrame(columns=range(nr_features), dtype=float)
        return (empty_df, {}) if return_lengths else empty_df

    # 矢量化添加伪计数并按行归一化
    matrix = np.array(counts_list, dtype=np.float64)
    matrix += 1e-5
    matrix /= matrix.sum(axis=1, keepdims=True)

    df = pd.DataFrame(matrix, index=names, dtype=float)
    if return_lengths:
        return df, dict(zip(names, lengths_list))
    return df

