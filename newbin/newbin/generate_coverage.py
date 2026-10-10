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

def combine_cov(cov_dir : str, bam_list, is_combined : bool = True, eps : float = 1e-5):
    """
    Combine coverage files for every BAM and compute level, shape, and log_id features.

    Parameters
    ----------
    cov_dir : where coverage files are stored
    bam_list : list of BAM files
    is_combined : bool (kept for signature compatibility)
    eps : pseudocount for log transformation and dispersion index

    Returns
    -------
    data_cov : DataFrame with shape (N, 2M + 1)
        Columns:
          - cov_level: (N, 1) contig global mean log-depth
          - cov_shape_{sample}: (N, M) relative profile (log_mu - level)
          - cov_log_id_{sample}: (N, M) log dispersion index log(var / (mu + eps))
    """
    import pandas as pd
    import numpy as np

    cov_dfs = []
    for bam_index, bam_file in enumerate(bam_list):
        bam_fname = os.path.split(bam_file)[-1]
        cov_dfs.append(
            pd.read_csv(f'{cov_dir}/{bam_fname}_{bam_index}_data_cov.csv', index_col=0))

    if not cov_dfs:
        return pd.DataFrame()

    common_index = cov_dfs[0].index
    for df in cov_dfs[1:]:
        common_index = common_index.intersection(df.index)

    mu_list = []
    var_list = []
    sample_names = []

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

    # mu and var shapes: (N, M)
    mu = np.column_stack(mu_list)
    var = np.column_stack(var_list)

    # 阶段 1: mu -> log_mu -> level + shape
    log_mu = np.log(mu + eps)
    level = log_mu.mean(axis=1, keepdims=True)  # (N, 1)
    shape = log_mu - level                       # (N, M)

    # 阶段 2: var, mu -> Index of Dispersion -> log_id
    dispersion_index = var / (mu + eps)
    log_id = np.log(np.clip(dispersion_index, eps, None))  # (N, M)

    # 阶段 3: 拼成输入向量 (N, 2M + 1)
    features = np.hstack([level, shape, log_id])

    level_cols = ['cov_level']
    shape_cols = [f'cov_shape_{s}' for s in sample_names]
    log_id_cols = [f'cov_log_id_{s}' for s in sample_names]
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
