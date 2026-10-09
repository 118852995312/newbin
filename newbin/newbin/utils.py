import os
import subprocess
import multiprocessing
import logging
import sys
import random
import contextlib
import pathlib

from .fasta import fasta_iter


@contextlib.contextmanager
def possibly_compressed_write(filename):
    from .atomicwrite import atomic_write
    if filename.endswith('.gz'):
        import gzip
        mode = 'wb'
        transf = lambda f : gzip.open(f, mode='wt')
    elif filename.endswith('.bz2'):
        import bz2
        mode = 'wb'
        transf = lambda f : bz2.open(f, mode='wt')
    elif filename.endswith('.xz'):
        import lzma
        mode = 'wb'
        transf = lambda f : lzma.open(f, mode='wt')
    else:
        mode = 'wt'
        transf = lambda f : f
    with atomic_write(filename, mode=mode, overwrite=True) as f:
        g = transf(f)
        yield g
        if g is not f:
            g.close()


def validate_normalize_args(logger, args):
    '''Validate and normalize command line arguments'''
    exit_with_error = False

    def expect_file(f):
        nonlocal exit_with_error
        if f is not None:
            if not os.path.exists(f):
                logger.error(
                    f"Error: Expected file '{f}' does not exist\n")
                exit_with_error = True

    def expect_file_list(fs):
        if fs is not None:
            for f in fs:
                expect_file(f)

    if hasattr(args, 'num_process'):
        if args.num_process == 0:
            args.num_process = multiprocessing.cpu_count()
            logger.info(f'Setting number of CPUs to {args.num_process}')

        if args.num_process > multiprocessing.cpu_count():
            args.num_process = multiprocessing.cpu_count()
        os.environ['NUMEXPR_NUM_THREADS'] = str(args.num_process)
        os.environ['NUMEXPR_MAX_THREADS'] = str(args.num_process)
        os.environ['OMP_NUM_THREADS'] = str(args.num_process)
        os.environ['OPENBLAS_NUM_THREADS'] = str(args.num_process)
        os.environ['MKL_NUM_THREADS'] = str(args.num_process)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(args.num_process)

    if hasattr(args, 'engine'):
        args.engine = args.engine.lower()
        if args.engine not in ['auto', 'gpu', 'cpu']:
            logger.error(
                "Argument '--engine' needs to be one of auto[default]/gpu/cpu.")
            exit_with_error = True

    if args.cmd in ['generate_sequence_features_single', 'generate_sequence_features_multi', 'single_easy_bin', 'multi_easy_bin']:
        if args.bams and args.abundances:
            logger.error(
                "Cannot use BAM files and abundance files at the same time.")
            exit_with_error = True

    if args.cmd == 'generate_sequence_features_single':
        expect_file(args.contig_fasta)
        if args.bams:
            expect_file_list(args.bams)
        if args.abundances:
            expect_file_list(args.abundances)

    if args.cmd == 'generate_sequence_features_multi':
        expect_file(args.contig_fasta)
        if args.bams:
            expect_file_list(args.bams)
        if args.abundances:
            expect_file_list(args.abundances)

    if args.cmd in ['train_self']:
        if not args.train_from_many:
            if len(args.data) > 1:
                logger.error(
                    "Expected one data.csv file with single mode.")
                exit_with_error = True

            if len(args.data_split) > 1:
                logger.error(
                    "Expected one data_split.csv file with single mode.")
                exit_with_error = True

            expect_file(args.data[0])
            expect_file(args.data_split[0])

        else:
            assert len(args.data) == len(args.data_split), 'Must input same number of data, data_split!'
            expect_file_list(args.data)
            expect_file_list(args.data_split)

    if args.cmd in ['single_easy_bin', 'multi_easy_bin']:
        if args.sequencing_type.lower() in ['short', 'short-read', 'short_reads', 'short-reads', 'short_read']:
            args.sequencing_type = 'short_read'
        elif args.sequencing_type.lower() in ['long', 'long-read', 'long_reads', 'long-reads', 'long_read', 'hybrid']:
            args.sequencing_type = 'long_read'
        else:
            logger.error(
                f"Did not understand sequencing_type argument '{args.sequencing_type}' (should be short_reads or long_reads).")
            exit_with_error = True
        logger.info(f'Binning for {args.sequencing_type}')

    if args.cmd == 'bin':
        if args.model_path is None:
            logger.error(
                "Please provide a model path (--model-path).")
            exit_with_error = True
        if args.model_path is not None:
            expect_file(args.model_path)
        expect_file(args.contig_fasta)
        expect_file(args.data)

    if args.cmd == 'concatenate_fasta':
        expect_file_list(args.contig_fasta)
        contig_name = []
        for contig in args.contig_fasta:
            contig_name.append(os.path.basename(contig).split('.')[0])
        if len(set(contig_name)) != len(contig_name):
            logger.error(
                "Make sure that every contig file has a different name.")
            exit_with_error = True

    if getattr(args, 'train_from_many', False):
        args.mode = 'several'
    elif args.cmd in ['train_self'] and not hasattr(args, 'mode'):
        args.mode = 'single'

    if getattr(args, 'write_pre_reclustering_bins', False) and \
            not getattr(args, 'recluster', True):
        logger.error(
            "Cannot use --write-pre-reclustering-bins with --no-recluster.")
        exit_with_error = True

    if exit_with_error:
        sys.exit(1)


def get_must_link_threshold(contig_lens):
    """
    calculate the threshold length for must link breaking up
    """
    import numpy as np
    contig_lens = np.array(contig_lens)
    contig_lens.sort()
    frac = np.cumsum(contig_lens)/np.sum(contig_lens)
    ix = np.argmax(frac > 0.02) # argmax finds first True element
    threshold = contig_lens[ix]
    return np.clip(threshold, 4000, None)


def maybe_uncompress(fafile, tdir):
    if fafile.endswith('.gz') or \
            fafile.endswith('.bz2') or \
            fafile.endswith('.xz'):
        oname = f'{tdir}/expanded.fa'
        with open(oname, 'wt') as out:
            for header, seq in fasta_iter(fafile):
                out.write(f'>{header}\n{seq}\n')
        return oname
    return fafile


def write_bins(namelist, contig_labels, output, contig_seqs,
               minfasta = 200000, output_tag=None, output_compression='none'):
    '''
    Write binned FASTA files

    Returns: DataFrame with information on the bins
    '''
    from collections import defaultdict
    import pandas as pd

    res = defaultdict(list)
    for label, name in zip(contig_labels, namelist, strict=True):
        if label != -1:
            res[label].append(name)

    os.makedirs(output, exist_ok=True)

    written = []
    for label, contigs in res.items():
        sizes = [len(contig_seqs[contig]) for contig in contigs]
        whole_bin_bp = sum(sizes)

        if whole_bin_bp >= minfasta:
            if output_tag is None:
                ofname = f'bin.{label}.fa'
            else:
                ofname = f'{output_tag}_{label}.fa'
            ofname = os.path.join(output, ofname)
            if output_compression != 'none':
                ofname += '.' + output_compression
            n50, l50 = n50_l50(sizes)
            written.append([ofname, whole_bin_bp, len(contigs), n50, l50])
            with possibly_compressed_write(ofname) as ofile:
                for contig in contigs:
                    ofile.write(f'>{contig}\n{contig_seqs[contig]}\n')
    return pd.DataFrame(written, columns=['filename', 'nbps', 'nr_contigs', 'N50', 'L50'])


def log_binning_stats(logger, bins_info, contig_dict, binned_length=None):
    '''Log summary statistics about the produced bins

    Parameters
    ----------
    logger
        logger to write the summary to
    bins_info : pd.DataFrame
        the DataFrame returned by ``write_bins`` (one row per bin, with
        ``nbps`` and ``nr_contigs`` columns)
    contig_dict : dict
        contig-id -> sequence for the whole input assembly
    binned_length : int, optional
        minimum contig length considered for binning; when given, contigs
        shorter than this are reported as a separate "too short" category
    '''
    n_bins = len(bins_info)
    if n_bins == 0:
        return

    total_contigs = len(contig_dict)
    total_bp = sum(len(s) for s in contig_dict.values())

    binned_contigs = int(bins_info['nr_contigs'].sum())
    binned_bp = int(bins_info['nbps'].sum())

    def _pct(part, whole):
        return 100.0 * part / whole if whole else 0.0

    if binned_length is not None:
        short_contigs = sum(len(s) < binned_length for s in contig_dict.values())
        short_bp = sum(len(s) for s in contig_dict.values() if len(s) < binned_length)
    else:
        short_contigs = short_bp = 0

    logger.info('Binning summary:')
    logger.info(f'  Bins produced: {n_bins}')
    logger.info(
        f'  Contigs binned: {binned_contigs:,} / {total_contigs:,} '
        f'({_pct(binned_contigs, total_contigs):.1f}%)')
    logger.info(
        f'  Basepairs binned: {binned_bp:,} / {total_bp:,} '
        f'({_pct(binned_bp, total_bp):.1f}%)')
    if binned_length is not None:
        eligible_unbinned = total_contigs - binned_contigs - short_contigs
        logger.info(
            f'  Unbinned: {eligible_unbinned:,} contigs '
            f'({total_bp - binned_bp - short_bp:,} bp) passed the length '
            f'filter but were not binned; '
            f'{short_contigs:,} contigs ({short_bp:,} bp) too short '
            f'(< {binned_length:,} bp)')
    else:
        logger.info(
            f'  Unbinned: {total_contigs - binned_contigs:,} contigs '
            f'({total_bp - binned_bp:,} bp)')
    sizes = bins_info['nbps']
    logger.info(
        f'  Bin size (bp): mean {binned_bp // n_bins:,}, '
        f'median {int(sizes.median()):,}, '
        f'min {int(sizes.min()):,}, max {int(sizes.max()):,}')
    logger.info(
        f'  Contigs per bin: mean {binned_contigs / n_bins:.1f}, '
        f'median {int(bins_info["nr_contigs"].median())}')
    logger.info(
        '  Note: these bins are not quality-controlled and should not be '
        'treated as MAGs without further QC (e.g., checking completeness and '
        'contamination with CheckM2, and chimerism with GUNC)')


def set_random_seed(seed):
    import torch
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_fasta(fasta_path: str, ratio: float):
    """
    Returns

    computed_min_length: minimum length of contigs (1000 or 2500, depending on the ratio)
    must_link_threshold: threshold to break up contigs
    contigs: dictionary ID -> contig sequence
    """
    total_bps = 0
    contig_bp_2500 = 0
    contig_length_list = []
    contig_dict = {}

    for h, seq in fasta_iter(fasta_path):
        if 1000 <= len(seq) <= 2500:
            contig_bp_2500 += len(seq)
        contig_length_list.append(len(seq))
        total_bps += len(seq)
        contig_dict[h] = seq

    if total_bps == 0:
        logger = logging.getLogger('NewBin')
        logger.warning(f'No contigs in {fasta_path}')
        return 1000, 4000, contig_dict

    computed_min_length = (
                1000
                if contig_bp_2500 / total_bps < ratio
                else 2500)
    must_link_threshold = get_must_link_threshold(contig_length_list)
    return computed_min_length, must_link_threshold, contig_dict


def split_data(data, sample, separator, is_combined = True):
    """
    split data according their sample in multi-sample binning
    """
    import numpy as np
    part_data = data[data['contig_name'].str.startswith(
        sample + separator)]
    part_data = part_data.set_index('contig_name')
    part_data.index.name = None
    part_data.index = [ix.split(separator)[1] for ix in part_data.index]
    if is_combined:
        abun_scale = (part_data.mean() / 100).apply(np.ceil) * 100
        part_data = part_data.div(abun_scale)
    return part_data


def concatenate_fasta(fasta_files,
                      min_length,
                      output,
                      separator,
                      output_compression='none',
                      keep_full_header=False,
                      ):
    """
    Concatenate multiple FASTA files into one

    Returns name of the concatenated file
    """
    logger = logging.getLogger('NewBin')
    ofname = os.path.join(output, 'concatenated.fa')
    if output_compression != 'none':
        ofname += '.' + output_compression
    with possibly_compressed_write(ofname) as concat_out:
        for fname in fasta_files:
            sample_name = os.path.basename(fname).split('.')[0]
            for fh, seq in fasta_iter(fname, full_header=True):
                h, _, rest = fh.partition(' ')
                if separator in h:
                    logger.error(
                        f"In file {fname}: contig ID '{h}' contains the separator ('{separator}'), please choose another separator.")
                    sys.exit(1)
                if len(seq) >= min_length:
                    header = f'{sample_name}{separator}{h}'
                    if keep_full_header and rest:
                        header += f' {rest}'
                    concat_out.write(f'>{header}\n{seq}\n')
    return ofname

def n50_l50(sizes):
    '''Computes N50 & L50 for a set of contig sizes

    Returns (n50, l50)
    '''
    import numpy as np
    s = np.array(sizes)
    s.sort()
    s = s[::-1]
    l50 = np.where(np.cumsum(s)>=s.sum()//2)[0][0]
    n50 = s[l50]
    return n50, l50+1


def maybe_crams2bams(bams, contig_fasta : str, num_process : int, odir : str):
    '''
    maybe_crams2bams converts CRAM to BAM

    Parameters
    ----------
    bams : list of str
        List of BAM/CRAM files
    contig_fasta : str
        Contig FASTA file
    num_process : int
        Number of processes to use
    odir : str
        Output directory for extracted BAM files

    Returns
    -------
    obams : list of str
        List of BAM files (extracted if CRAM or original)
    '''
    rs = []
    for bam in bams:
        if bam.endswith('.cram'):
            obam = pathlib.PurePath(odir) / pathlib.PurePath(bam).with_suffix('.bam').name
            with open(obam, 'wb') as cram_out:
                subprocess.check_call(
                    ['samtools', 'view',
                     '-bS',
                     '-@', str(num_process),
                     '-T', contig_fasta,
                     bam],
                    stdout=cram_out)
            rs.append(str(obam))
        else:
            rs.append(bam)
    return rs


def maybe_compute_min_length(min_length, fafile, ratio):
    if min_length is not None:
        return min_length
    c_min_len, _, _ = load_fasta(fafile, ratio)
    return c_min_len


def norm_abundance(data):
    import numpy as np
    n = data.shape[1] - 136

    if n >= 20:
        return True
    return (n >= 5 and np.mean(np.sum(data[:, 136:], axis=1)) > 2)
