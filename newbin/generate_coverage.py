import os
import subprocess
from .atomicwrite import atomic_write

def calculate_coverage(depth_stream, bam_file, edge=75, is_combined=False,
                       contig_threshold=1000, sep=None, sample_contig_threshold=None):
    """
    depth_stream : an iterable like the output of bedtools genomecov
    bam_file : str filename
    edge : int
    is_combined : bool
    contig_threshold : int
    sep : str or None
    sample_contig_threshold : dict or None
    """
    import pandas as pd
    import numpy as np
    from itertools import groupby

    contigs = []
    mean_coverage = []
    var = []

    for contig_name, lines in groupby(depth_stream, lambda ell: ell.split('\t', 1)[0]):
        lengths = []
        values = []
        for line in lines:
            line_split = line.strip().split('\t')
            length = int(line_split[2]) - int(line_split[1])
            value = int(float(line_split[3]))
            lengths.append(length)
            values.append(value)
        depth_value = np.zeros(sum(lengths), dtype=int)
        s = 0
        for ell,v in zip(lengths, values, strict=True):
            depth_value[s:s+ell] = v
            s += ell

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
        if len(depth_value) < cov_threshold:
            continue
        depth_value_ = depth_value[edge:-edge]
        mean_coverage.append(depth_value_.mean())
        var.append(depth_value_.var())
        contigs.append(contig_name)

    return pd.DataFrame({
        f'{bam_file}_mean': mean_coverage,
        f'{bam_file}_var': var,
    }, index=contigs)


def generate_cov(bam_file: str, bam_index, out,
                 is_combined, contig_threshold, logger, sep = None):
    """
    Call bedtools and generate coverage file

    bam_file: bam file used
    bam_index: index of bam file
    out: output directory
    is_combined: if using abundance feature in deep learning. True: use
    contig_threshold: threshold of contigs that will be binned
    logger: logger
    sep: separator for multi-sample binning
    """
    import numpy as np
    logger.debug(f'Processing `{bam_file}`')
    bam_name = os.path.split(bam_file)[-1] + f'_{bam_index}'

    bed_p = subprocess.Popen(
        ['bedtools', 'genomecov',
         '-bga',
         '-ibam', bam_file],
        universal_newlines=True,
        stdout=subprocess.PIPE)

    contig_cov = calculate_coverage(
        bed_p.stdout,
        bam_file,
        is_combined=is_combined,
        sep=sep,
        contig_threshold=(contig_threshold if sep is None else 1000),
        sample_contig_threshold=(contig_threshold if sep is not None else None))

    if bed_p.wait() != 0:
        raise OSError(f"Failure in running bedtools ({bam_file})")
    elif len(contig_cov) == 0:
        logger.critical(f"Running `bedtools genomecov` did not return an error, but the result is an empty file (processing {bam_file}). "
                        "Please check your input files: SemiBin expects that they are sorted BAM files.")
        raise OSError(f"Running bedtools returned an empty file ({bam_file})")

    with atomic_write(os.path.join(out, f'{bam_name}_data_cov.csv'), overwrite=True) as ofile:
        contig_cov.to_csv(ofile)

    return bam_file

def combine_cov(cov_dir : str, bam_list, is_combined : bool = True, eps : float = 1e-5, max_samples : int = 5, bam_indices = None, pad_to_max : bool = False):
    """
    Combine coverage files for every BAM and compute level, shape, and log_id features.
    针对固定 M=5 预训练架构：
    - 当样本数 M >= 5 时，按全局平均覆盖深度降序优选前 5 个代表性黄金样本；
    - 当样本数 M < 5 时（例如单样本由自身 BAM 计算 coverage），计算真实样本的 level, shape 与 log_id (2M+1 维)；
      若指定 pad_to_max=True，则自动在右侧填充 0 补齐至 11 维。

    Parameters
    ----------
    cov_dir : where coverage files are stored
    bam_list : list of BAM files
    is_combined : bool (kept for signature compatibility)
    eps : pseudocount for log transformation and dispersion index
    max_samples : maximum number of representative samples to keep (default: 5)
    bam_indices : list of int, optional (explicit indices of BAM files, default: 0..len-1)
    pad_to_max : bool, optional (whether to pad M < max_samples to 2 * max_samples + 1 dimensions, default False)

    Returns
    -------
    data_cov : DataFrame with shape (N, 2M + 1) or (N, 2 * max_samples + 1)
        Columns:
          - cov_level: (N, 1) contig global mean log-depth
          - cov_shape_{sample}: (N, M) relative profile (log_mu - level)
          - cov_log_id_{sample}: (N, M) log dispersion index log(var / (mu + eps))
    """
    import pandas as pd
    import numpy as np

    if bam_indices is None:
        bam_indices = list(range(len(bam_list)))

    cov_dfs = []
    for idx, bam_file in zip(bam_indices, bam_list):
        bam_fname = os.path.split(bam_file)[-1]
        cov_dfs.append(
            pd.read_csv(f'{cov_dir}/{bam_fname}_{idx}_data_cov.csv', index_col=0))

    if not cov_dfs:
        return pd.DataFrame()

    common_index = cov_dfs[0].index
    for df in cov_dfs[1:]:
        common_index = common_index.intersection(df.index)

    mu_list = []
    var_list = []
    sample_names = []

    for i, bam_file in enumerate(bam_list):
        bam_fname = os.path.split(bam_file)[-1]
        sample_names.append(bam_fname)
        df = cov_dfs[i].loc[common_index]

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

    # mu and var raw shapes: (N, M_raw)
    mu_raw = np.column_stack(mu_list)
    var_raw = np.column_stack(var_list)

    # 样本优选与组织 (Depth-First 深度优先 + Canonical 排序)
    # 当样本数 > max_samples (例如 > 5) 时，根据平均覆盖深度降序挑选前 5 个信噪比最高的黄金样本
    if mu_raw.shape[1] > max_samples:
        avg_depths = mu_raw.mean(axis=0)
        top_indices = np.argsort(-avg_depths)[:max_samples]
        mu = mu_raw[:, top_indices]
        var = var_raw[:, top_indices]
        selected_sample_names = [sample_names[i] for i in top_indices]
    else:
        mu = mu_raw
        var = var_raw
        selected_sample_names = sample_names

    # 阶段 1: mu -> log_mu -> level + shape
    log_mu = np.log(mu + eps)
    level = log_mu.mean(axis=1, keepdims=True)  # (N, 1)
    shape_raw = log_mu - level                   # (N, M_curr)

    # 阶段 2: var, mu -> Index of Dispersion -> log_id
    dispersion_index = var / (mu + eps)
    log_id_raw = np.log(np.clip(dispersion_index, eps, None))  # (N, M_curr)

    # 阶段 3: 当 pad_to_max=True 且样本数 M < max_samples 时补齐至 11 维
    m_curr = mu.shape[1]
    n_contigs = mu.shape[0]
    if pad_to_max and m_curr < max_samples:
        pad_width = max_samples - m_curr
        shape_pad = np.zeros((n_contigs, pad_width), dtype=np.float64)
        shape = np.hstack([shape_raw, shape_pad])

        log_id_pad = np.zeros((n_contigs, pad_width), dtype=np.float64)
        log_id = np.hstack([log_id_raw, log_id_pad])

        pad_names = [f'pad_{k+1}' for k in range(pad_width)]
        all_shape_names = selected_sample_names + pad_names
        all_log_id_names = selected_sample_names + pad_names
    else:
        shape = shape_raw
        log_id = log_id_raw
        all_shape_names = selected_sample_names
        all_log_id_names = selected_sample_names

    # 阶段 4: 拼成输入向量
    features = np.hstack([level, shape, log_id])

    level_cols = ['cov_level']
    shape_cols = [f'cov_shape_{s}' for s in all_shape_names]
    log_id_cols = [f'cov_log_id_{s}' for s in all_log_id_names]
    all_cols = level_cols + shape_cols + log_id_cols

    data_cov = pd.DataFrame(features, index=common_index, columns=all_cols)
    data_cov.index = data_cov.index.astype(str)
    return data_cov


def generate_cov_from_abundances(abundances, output, contig_path, contig_threshold=1000, sep=None, sample_contig_threshold=None):
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
