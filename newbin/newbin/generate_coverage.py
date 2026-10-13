"""
宏基因组 Coverage（测序深度/丰度）特征提取模块
================================================

整体流转逻辑（三步走）：
-----------------------
【前置输入（用户提供）】：
    用户在上游使用比对软件（如 Bowtie2 / BWA / Minimap2）将测序 Reads 比对回 Contigs，
    生成排好序的 `.bam` 文件并传入本程序。

【第一步：调用 bedtools 扫描单个 BAM 文件】 -> 对应函数 `generate_cov()`
    针对某一个样本的 `.bam` 文件，后台启动 `bedtools genomecov -bga -ibam`，
    在每个碱基位置上“竖着数叠了几层 reads”，输出连续区间的覆盖层数流，
    并调用 `calculate_coverage()` 计算该样本下每条 Contig 的均值与方差，保存为中间 CSV。

【第二步：展开位点深度并计算单样本均值 μ 与方差 var】 -> 对应函数 `calculate_coverage()`
    接收 `bedtools` 输出的区间覆盖流 `(contig名, 起始坐标, 结束坐标, 覆盖层数)`，
    将其还原为整条 Contig 每个碱基位点的深度数组 `depth_value`，
    掐掉头尾各 75 bp 的边缘不稳定区后，计算整条 Contig 的平均深度 μ (mean) 和方差 var。

【第三步：跨样本汇总并构造 (2M+1) 维 Coverage 特征】 -> 对应函数 `combine_cov()`
    读取全部 M 个样本算出的 (μ, var) 中间文件，合并为 (N, M) 矩阵，分三阶段构造最终特征：
      1. cov_level  (1 维) : 跨样本对数深度均值 m_i = mean(ln(μ_i + ε))，反映整体丰度高低；
      2. cov_shape  (M 维) : 中心化涨落剖面 d_i = ln(μ_i + ε) - m_i，反映跨样本相对波动趋势（行和恒为0）；
      3. cov_log_id (M 维) : 对数离散度 ln(var_i / (μ_i + ε))，反映单条序列内部覆盖的均匀程度。
    最终拼接为 (N, 2M + 1) 的 DataFrame 返回。
"""

import os
import subprocess
from .atomicwrite import atomic_write


# ===========================================================================
# 【第二步】接收 bedtools 区间深度流，展开为逐位点数组并计算均值 μ 和方差 var
# ===========================================================================
def calculate_coverage(depth_stream, bam_file, edge=75, is_combined=False,
                       contig_threshold=1000, sep=None, sample_contig_threshold=None):
    """
    解析 bedtools genomecov 输出的覆盖度文本流，计算单样本内每条 Contig 的平均深度 μ 与方差 var。

    工作步骤：
    ---------
    1. 按 Contig 名称分组读取 `bedtools genomecov -bga` 的输出行：
       每行格式为: `contig_name \t start_pos \t end_pos \t depth_count`
    2. 将每个区间的长度 `ell = end - start` 与覆盖层数 `v = depth_count` 填入一维数组 `depth_value`，
       还原出该 Contig 从第 0 个碱基到最后一个碱基每个位点的覆盖层数。
    3. 过滤掉总长度小于阈值 `cov_threshold` 的短序列。
    4. 掐头去尾各 `edge` (默认 75 bp) 个碱基（消除 Contig 边缘比对掉落导致的失真）。
    5. 对中间有效区域求均值 `mean()`（即 μ）和方差 `var()`，返回 DataFrame。

    Parameters
    ----------
    depth_stream : iterable of str
        `bedtools genomecov -bga` 的标准输出文本流。
    bam_file : str
        当前处理的 BAM 文件路径（用于生成列名标识）。
    edge : int, default=75
        计算均值与方差时，Contig 两端各自忽略的边缘碱基数（默认 75 bp）。
    is_combined : bool, default=False
        保留兼容参数。
    contig_threshold : int, default=1000
        单样本模式下保留 Contig 的最小长度阈值（单位 bp）。
    sep : str or None, default=None
        多样本联合分箱模式下切分 `样本名:序列名` 的分隔符。
    sample_contig_threshold : dict or None, default=None
        多样本模式下各样本对应的最小长度阈值字典。

    Returns
    -------
    pd.DataFrame, shape (N_valid, 2)
        索引为 `contig_name`，包含两列：
          - `{bam_file}_mean`: 单样本下的平均测序深度 μ
          - `{bam_file}_var` : 单样本下的测序深度方差 var
    """
    import pandas as pd
    import numpy as np
    from itertools import groupby

    contigs = []
    mean_coverage = []
    var = []

    # 2.1 按 Contig 名称对连续输出行进行分组（bedtools 输出已按序列聚集）
    for contig_name, lines in groupby(depth_stream, lambda ell: ell.split('\t', 1)[0]):
        lengths = []
        values = []
        for line in lines:
            line_split = line.strip().split('\t')
            # 区间长度 = 结束坐标 - 起始坐标
            length = int(line_split[2]) - int(line_split[1])
            # 该区间每个碱基上覆盖的 reads 层数
            value = int(float(line_split[3]))
            lengths.append(length)
            values.append(value)

        # 2.2 构建长度等于整条 Contig 的一维数组，将每个位点的覆盖层数展开填平
        depth_value = np.zeros(sum(lengths), dtype=int)
        s = 0
        for ell, v in zip(lengths, values, strict=True):
            depth_value[s:s+ell] = v
            s += ell

        # 2.3 确定当前 Contig 的最小长度过滤阈值
        if sep is None:
            cov_threshold = contig_threshold
        else:
            try:
                sample, contig = contig_name.split(sep)
                cov_threshold = sample_contig_threshold[sample]
            except ValueError:
                raise ValueError(f"Error parsing contig name '{contig_name}' in file {bam_file} (trying to split by {sep} separator)")
            except KeyError:
                raise KeyError(f"Error: {sample} not found (parsing {bam_file}, the original contig name was {contig_name} and using {sep} as separator)")

        # 过滤过短的 Contig
        if len(depth_value) < cov_threshold:
            continue

        # 2.4 掐头去尾各 edge (75 bp)，计算有效区域的平均深度 μ 和方差 var
        depth_value_ = depth_value[edge:-edge]
        mean_coverage.append(depth_value_.mean())
        var.append(depth_value_.var())
        contigs.append(contig_name)

    return pd.DataFrame({
        f'{bam_file}_mean': mean_coverage,
        f'{bam_file}_var': var,
    }, index=contigs)


# ===========================================================================
# 【第一步】针对单个 BAM 文件启动 bedtools，驱动计算并落盘单样本中间文件
# ===========================================================================
def generate_cov(bam_file: str, bam_index, out,
                 is_combined, contig_threshold, logger, sep = None):
    """
    针对单个 BAM 比对文件，调用系统 `bedtools genomecov` 提取覆盖度并保存单样本统计文件。

    工作步骤：
    ---------
    1. 在后台启动子进程执行命令：`bedtools genomecov -bga -ibam <bam_file>`，
       统计 BAM 文件中每条 Contig 各个坐标区间上堆叠的 reads 层数。
    2. 将子进程的标准输出管道 `bed_p.stdout` 实时流式传递给【第二步】`calculate_coverage()`，
       算出该 BAM 样本下所有合格 Contig 的均值 μ 与方差 var。
    3. 检查 `bedtools` 执行状态与结果非空性。
    4. 将结果原子写入磁盘中间文件：`{out}/{bam_fname}_{bam_index}_data_cov.csv`，供【第三步】汇总使用。

    Parameters
    ----------
    bam_file : str
        输入的单个排序后 BAM 文件路径。
    bam_index : int
        当前 BAM 文件在样本列表中的序号（0, 1, ..., M-1）。
    out : str
        输出目录路径。
    is_combined : bool
        是否使用多样本联合特征标识。
    contig_threshold : int or dict
        参与分箱的 Contig 最小长度阈值。
    logger : logging.Logger
        日志记录器。
    sep : str or None, default=None
        多样本模式分隔符。

    Returns
    -------
    bam_file : str
        处理完成的 BAM 文件路径。
    """
    import numpy as np
    logger.debug(f'Processing `{bam_file}`')
    bam_name = os.path.split(bam_file)[-1] + f'_{bam_index}'

    # 1.1 启动 bedtools 子进程，按区间统计每个位点覆盖了多少条 reads (-bga 包含深度为 0 的区间)
    bed_p = subprocess.Popen(
        ['bedtools', 'genomecov',
         '-bga',
         '-ibam', bam_file],
        universal_newlines=True,
        stdout=subprocess.PIPE)

    # 1.2 将输出流实时交给第二步函数 calculate_coverage 计算均值 μ 和方差 var
    contig_cov = calculate_coverage(
        bed_p.stdout,
        bam_file,
        is_combined=is_combined,
        sep=sep,
        contig_threshold=(contig_threshold if sep is None else 1000),
        sample_contig_threshold=(contig_threshold if sep is not None else None))

    # 1.3 校验子进程退出码与结果有效性
    if bed_p.wait() != 0:
        raise OSError(f"Failure in running bedtools ({bam_file})")
    elif len(contig_cov) == 0:
        logger.critical(f"Running `bedtools genomecov` did not return an error, but the result is an empty file (processing {bam_file}). "
                        "Please check your input files: SemiBin expects that they are sorted BAM files.")
        raise OSError(f"Running bedtools returned an empty file ({bam_file})")

    # 1.4 将当前样本的 (mean, var) 表保存为中间 CSV 文件
    with atomic_write(os.path.join(out, f'{bam_name}_data_cov.csv'), overwrite=True) as ofile:
        contig_cov.to_csv(ofile)

    return bam_file


# ===========================================================================
# 【第三步】跨样本汇总 M 个 BAM 的 (μ, var)，计算 (2M+1) 维 Coverage 特征
# ===========================================================================
def combine_cov(cov_dir : str, bam_list, is_combined : bool = True, eps : float = 1e-5):
    """
    汇总所有 M 个 BAM 样本的中间覆盖度文件，计算最终的 (2M + 1) 维对数解耦 Coverage 特征。

    工作步骤：
    ---------
    1. 依次读取【第一步】生成的 M 个中间文件 `{bam_fname}_{bam_index}_data_cov.csv`，
       取所有样本共有的 Contig 交集 `common_index`。
    2. 提取每个样本的均值列和方差列，横向拼成两个 (N, M) 矩阵：
       - `mu`  (N, M): N 条 Contig 在 M 个样本中的平均深度矩阵
       - `var` (N, M): N 条 Contig 在 M 个样本中的深度方差矩阵
    3. 分三个数学阶段构造 (2M + 1) 维特征：
       - 【阶段 1：对数中心化分解】
         * `log_mu = ln(mu + eps)`            : 加伪计数取自然对数，将乘性偏差转为加性偏差
         * `level  = mean(log_mu, axis=1)`    : (N, 1)，单条 Contig 跨样本的平均对数丰度水平
         * `shape  = log_mu - level`          : (N, M)，消除绝对丰度后的跨样本相对涨落剖面（每行和恒为 0）
       - 【阶段 2：对数离散指数 (Index of Dispersion)】
         * `dispersion_index = var / (mu + eps)`
         * `log_id = ln(clip(dispersion_index, eps, None))` : (N, M)，衡量深度分布的离散/均匀程度
       - 【阶段 3：特征横向拼接】
         * 按 `[level (1列), shape (M列), log_id (M列)]` 顺序拼接为 (N, 2M + 1) 的特征表返回。

    Parameters
    ----------
    cov_dir : str
        存放单样本 `*_data_cov.csv` 中间文件的目录。
    bam_list : list of str
        M 个 BAM 文件路径列表。
    is_combined : bool, default=True
        保留接口兼容参数。
    eps : float, default=1e-5
        防止对数 log(0) 或除以 0 的平滑伪计数。

    Returns
    -------
    data_cov : pd.DataFrame, shape (N, 2M + 1)
        包含以下三组列：
          - `cov_level`          : 1 列 (第 0 列)
          - `cov_shape_{sample}` : M 列 (第 1 .. M 列，共丰度辅助损失 L_co 直接切取此区间)
          - `cov_log_id_{sample}`: M 列 (第 M+1 .. 2M 列)
    """
    import pandas as pd
    import numpy as np

    # 3.1 读取每个 BAM 样本在第一步生成的 _data_cov.csv 文件
    cov_dfs = []
    for bam_index, bam_file in enumerate(bam_list):
        bam_fname = os.path.split(bam_file)[-1]
        cov_dfs.append(
            pd.read_csv(f'{cov_dir}/{bam_fname}_{bam_index}_data_cov.csv', index_col=0))

    if not cov_dfs:
        return pd.DataFrame()

    # 3.2 对齐所有样本共有的 Contig 索引
    common_index = cov_dfs[0].index
    for df in cov_dfs[1:]:
        common_index = common_index.intersection(df.index)

    mu_list = []
    var_list = []
    sample_names = []

    # 3.3 收集每个样本的 mean (μ) 和 var 列
    for bam_index, bam_file in enumerate(bam_list):
        bam_fname = os.path.split(bam_file)[-1]
        sample_names.append(bam_fname)
        df = cov_dfs[bam_index].loc[common_index]

        mean_cols = [c for c in df.columns if c.endswith('_mean')]
        var_cols = [c for c in df.columns if c.endswith('_var')]

        if mean_cols and var_cols:
            mean_col = mean_cols[0]
            var_col = var_cols[0]
        else:
            mean_col = df.columns[0]
            var_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        mu_list.append(df[mean_col].values.astype(np.float64))
        var_list.append(df[var_col].values.astype(np.float64))

    # 拼成 (N, M) 矩阵：N 为 Contig 数，M 为 BAM 样本数
    mu = np.column_stack(mu_list)
    var = np.column_stack(var_list)

    # 3.4 阶段 1: mu -> log_mu -> level (1维) + shape (M维，中心化后行均值为0)
    log_mu = np.log(mu + eps)
    level = log_mu.mean(axis=1, keepdims=True)  # (N, 1)
    shape = log_mu - level                       # (N, M)

    # 3.5 阶段 2: var, mu -> Index of Dispersion (离散指数) -> log_id (M维)
    dispersion_index = var / (mu + eps)
    log_id = np.log(np.clip(dispersion_index, eps, None))  # (N, M)

    # 3.6 阶段 3: 按固定顺序 [level, shape, log_id] 拼成 (N, 2M + 1) 输入特征矩阵
    features = np.hstack([level, shape, log_id])

    level_cols = ['cov_level']
    shape_cols = [f'cov_shape_{s}' for s in sample_names]
    log_id_cols = [f'cov_log_id_{s}' for s in sample_names]
    all_cols = level_cols + shape_cols + log_id_cols

    data_cov = pd.DataFrame(features, index=common_index, columns=all_cols)
    data_cov.index = data_cov.index.astype(str)
    return data_cov


# ===========================================================================
# 【备选分支】当用户不传 BAM 而是直接提供预计算好的丰度表 (--abundance) 时调用
# ===========================================================================
def generate_cov_from_abundances(abundances, output, contig_path, contig_threshold=1000, sep=None, sample_contig_threshold=None):
    """
    从外部预计算的丰度文件列表（TSV 格式）直接读取丰度特征（兼容旧版 --abundance 输入模式）。
    """
    import pandas as pd
    import numpy as np
    from .fasta import fasta_iter

    abun_split_list = []
    for abun_file in abundances:
        abun_split_data = pd.read_csv(abun_file, sep='\t', index_col=0, header=None)
        abun_split_data.index.name = None
        abun_split_data.columns = [f'{abun_file}']

        binned_contig = []
        for h, seq in fasta_iter(contig_path):
            if sep is None:
                cov_threshold = contig_threshold
            else:
                try:
                    sample_name, contig_name = h.split(sep)
                    cov_threshold = sample_contig_threshold[sample_name]
                except ValueError:
                    raise ValueError(f"Error parsing contig name '{h}' in file {abun_file} (trying to split by {sep} separator)")
                except KeyError:
                    raise KeyError(f"Error: {sample_name} not found")

            if len(seq) >= cov_threshold:
                binned_contig.append(h + '_1')
                binned_contig.append(h + '_2')
        abun_split_data = abun_split_data.loc[binned_contig]
        abun_split_data = abun_split_data.apply(lambda x: x + 1e-5)
        if sep is None:
            abun_scale = (abun_split_data.mean() / 100).apply(np.ceil) * 100
            abun_split_data = abun_split_data.div(abun_scale)
        abun_split_list.append(abun_split_data)

    abun_split = pd.concat(abun_split_list, axis=1)
    if sep is None:
        with atomic_write(os.path.join(output, 'cov_split.csv'), overwrite=True) as ofile:
            abun_split.to_csv(ofile)

        index_name = abun_split.index.tolist()
        data_index_name = []
        for i in range(len(index_name) // 2):
            data_index_name.append(index_name[2 * i][0:-2])

        abun_split_values = abun_split.values
        abun = (abun_split_values[::2] + abun_split_values[1::2]) / 2
        columns = [f'{abun_file}' for abun_file in abundances]
        abun = pd.DataFrame(abun, index=data_index_name, columns=columns)
        with atomic_write(os.path.join(output, 'cov.csv'), overwrite=True) as ofile:
            abun.to_csv(ofile)
        return abun, abun_split
    else:
        return abun_split

