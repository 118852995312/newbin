# PYTHON_ARGCOMPLETE_OK
import argparse
import logging
import os
from os import path
import multiprocessing as mp
import subprocess
from .atomicwrite import atomic_write
import shutil
import sys
from itertools import groupby
from . import utils
from .utils import validate_normalize_args, \
    set_random_seed, load_fasta, split_data, maybe_crams2bams
from .generate_coverage import generate_cov, combine_cov, generate_cov_from_abundances
from .generate_kmer import generate_kmer_features_from_fasta
from .fasta import fasta_iter
from .newbin_version import __version__
import pandas as pd
Pool = mp.get_context('spawn').Pool


def newbin_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='NewBin — metagenomic binning with a custom deep learning model',
        epilog='See README for usage examples.')

    parser.add_argument('-v', '-V', '--version',
                        action='version',
                        version=__version__,
                        help='Print the version number')

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument('--verbose',
                           required=False,
                           help='Verbose output',
                           dest='verbose',
                           action='store_true')
    verbosity.add_argument('--quiet', '-q',
                           required=False,
                           help='Quiet output',
                           dest='quiet',
                           action='store_true')

    subparsers = parser.add_subparsers(title='NewBin subcommands',
                                       dest='cmd',
                                       metavar='')

    # ------------------------------------------------------------------
    #  generate_sequence_features_single
    # ------------------------------------------------------------------
    gen_single = subparsers.add_parser(
        'generate_sequence_features_single',
        help='Generate sequence features (kmer + abundance) for single-sample / co-assembly mode. '
             'Produces data.csv and data_split.csv.')

    gen_single.add_argument('--kmer',
                            required=False,
                            help='Only output data.csv with k-mer features (no coverage).',
                            dest='kmer',
                            action='store_true')

    # ------------------------------------------------------------------
    #  generate_sequence_features_multi
    # ------------------------------------------------------------------
    subparsers.add_parser(
        'generate_sequence_features_multi',
        help='Generate sequence features for multi-sample mode.')

    # ------------------------------------------------------------------
    #  train_self
    # ------------------------------------------------------------------
    train_self = subparsers.add_parser(
        'train_self',
        help='Train the NewBin model (self-supervised mode).')

    train_self.add_argument('--train-from-many',
                            required=False,
                            help='Train from multiple samples at once.',
                            dest='train_from_many',
                            action='store_true',
                            default=False)

    # ------------------------------------------------------------------
    #  bin  (short-read)
    # ------------------------------------------------------------------
    subparsers.add_parser('bin',
                          help='Cluster contigs using a trained model (short-read).')

    # ------------------------------------------------------------------
    #  bin_long  (long-read)
    # ------------------------------------------------------------------
    subparsers.add_parser('bin_long',
                          help='Cluster contigs using a trained model (long-read).')

    # ------------------------------------------------------------------
    #  single_easy_bin  (all-in-one)
    # ------------------------------------------------------------------
    subparsers.add_parser('single_easy_bin',
                          help='All-in-one: generate features + train + bin (single / co-assembly).')

    # ------------------------------------------------------------------
    #  multi_easy_bin  (all-in-one)
    # ------------------------------------------------------------------
    subparsers.add_parser('multi_easy_bin',
                          help='All-in-one: generate features + train + bin (multi-sample).')

    # ------------------------------------------------------------------
    #  concatenate_fasta
    # ------------------------------------------------------------------
    concat = subparsers.add_parser('concatenate_fasta',
                                   help='Concatenate FASTA files for multi-sample binning.')
    concat.add_argument('-i', '--input-fasta',
                        required=True, nargs='*',
                        help='Input FASTA files.',
                        dest='contig_fasta', default=None)
    concat.add_argument('-o', '--output',
                        required=True,
                        help='Output directory.',
                        dest='output', default=None)

    # ------------------------------------------------------------------
    #  check_install
    # ------------------------------------------------------------------
    subparsers.add_parser('check_install',
                          aliases=['check-install'],
                          help='Check whether required dependencies are present.')

    return parser


# --------------------------------------------------------------------------
#  Shared argument groups (appended to multiple subparsers after creation)
# --------------------------------------------------------------------------

def _add_common_args(p):
    """Add arguments that are shared across most subcommands."""
    p.add_argument('-i', '--input-fasta',
                   required=False,
                   help='Path to the input contigs FASTA file.',
                   dest='contig_fasta',
                   default=None)
    p.add_argument('-o', '--output',
                   required=True,
                   help='Output directory (will be created if non-existent).',
                   dest='output',
                   default=None)
    p.add_argument('-p', '--num-process',
                   required=False,
                   type=int,
                   help='Number of subprocess used in parallel (0 = use all CPUs).',
                   dest='num_process',
                   default=0)
    p.add_argument('--random-seed',
                   required=False,
                   type=int,
                   help='Random seed.',
                   dest='random_seed',
                   default=0)


def _add_binning_args(p):
    p.add_argument('-d', '--data',
                   required=False,
                   help='Path to the data.csv file.',
                   dest='data',
                   default=None)
    p.add_argument('--model-path',
                   required=False,
                   help='Path to a trained model (.pt) file.',
                   dest='model_path',
                   default=None)
    p.add_argument('--max-edges',
                   required=False,
                   type=int,
                   help='Maximum number of edges per contig in the k-NN graph.',
                   dest='max_edges',
                   default=200)
    p.add_argument('--max-node',
                   required=False,
                   type=float,
                   help='Maximum proportion of nodes retained in graph sparsification.',
                   dest='max_node',
                   default=1.0)
    p.add_argument('--minfasta-kb',
                   required=False,
                   type=int,
                   help='Minimum bin size in kilobases.',
                   dest='minfasta_kb',
                   default=200)
    p.add_argument('--recluster',
                   required=False,
                   help='Enable reclustering (default: enabled for short reads).',
                   dest='recluster',
                   action='store_true',
                   default=True)
    p.add_argument('--no-recluster',
                   required=False,
                   help='Disable reclustering.',
                   dest='recluster',
                   action='store_false')
    p.add_argument('--write-pre-reclustering-bins',
                   required=False,
                   help='Also write bins before reclustering.',
                   dest='write_pre_reclustering_bins',
                   action='store_true',
                   default=False)
    p.add_argument('--output-tag',
                   required=False,
                   help='Tag string prepended to bin filenames.',
                   dest='output_tag',
                   default=None)
    p.add_argument('--output-compression',
                   required=False,
                   help='Compression of output bin files (none/gz/bz2/xz).',
                   dest='output_compression',
                   default='none')
    p.add_argument('--orf-finder',
                   required=False,
                   help='ORF finder used for marker gene detection (fast-naive/prodigal/fraggenescan).',
                   dest='orf_finder',
                   default='fast-naive')
    p.add_argument('--prodigal-output-faa',
                   required=False,
                   help='(Deprecated) Path to pre-computed prodigal .faa output.',
                   dest='prodigal_output_faa',
                   default=None)


def _add_training_args(p):
    p.add_argument('--epochs',
                   required=False,
                   type=int,
                   help='Number of training epochs.',
                   dest='epochs',
                   default=15)
    p.add_argument('--batch-size',
                   required=False,
                   type=int,
                   help='Mini-batch size.',
                   dest='batchsize',
                   default=2048)
    p.add_argument('--engine',
                   required=False,
                   help='Compute engine for training (auto/gpu/cpu).',
                   dest='engine',
                   default='auto')
    p.add_argument('-d', '--data',
                   required=False,
                   nargs='*',
                   help='Path(s) to data.csv file(s).',
                   dest='data',
                   default=None)
    p.add_argument('--data-split',
                   required=False,
                   nargs='*',
                   help='Path(s) to data_split.csv file(s).',
                   dest='data_split',
                   default=None)


def _add_feature_args(p):
    p.add_argument('-b', '--input-bam',
                   required=False,
                   nargs='*',
                   help='Path(s) to sorted BAM/CRAM file(s).',
                   dest='bams',
                   default=None)
    p.add_argument('--abundances',
                   required=False,
                   nargs='*',
                   help='Path(s) to abundance files from strobealign-aemb (≥5 required).',
                   dest='abundances',
                   default=None)
    p.add_argument('--min-len',
                   required=False,
                   type=int,
                   help='Minimum contig length to bin (auto-determined if omitted).',
                   dest='min_len',
                   default=None)
    p.add_argument('--ratio',
                   required=False,
                   type=float,
                   help='Ratio threshold for auto min-len selection (1000 vs 2500 bp).',
                   dest='ratio',
                   default=0.05)
    p.add_argument('--ml-threshold',
                   required=False,
                   type=int,
                   help='Must-link threshold (bp). Auto-determined if omitted.',
                   dest='ml_threshold',
                   default=None)


# --------------------------------------------------------------------------
#  Helper: select compute device
# --------------------------------------------------------------------------

def _select_device(engine: str):
    import torch
    if engine == 'gpu':
        if not torch.cuda.is_available():
            logging.getLogger('NewBin').error('GPU requested but CUDA is not available.')
            sys.exit(1)
        return torch.device('cuda')
    elif engine == 'cpu':
        return torch.device('cpu')
    else:  # auto
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --------------------------------------------------------------------------
#  check_install
# --------------------------------------------------------------------------

def check_install(verbose: bool = False):
    logger = logging.getLogger('NewBin')
    missing_deps = []
    deps = {
        'bedtools': 'bedtools --version',
        'hmmsearch': 'hmmsearch -h',
    }
    for dep, cmd in deps.items():
        try:
            subprocess.check_call(cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if verbose:
                p = subprocess.check_output(['which', dep], stderr=subprocess.DEVNULL).decode().strip()
                print(f'\t{dep:16}: {p}')
        except Exception:
            missing_deps.append(dep)
            logger.warning(f'Dependency not found: {dep}')

    # Check torch
    try:
        import torch
        if verbose:
            print(f'\t{"torch":16}: {torch.__version__}')
            if torch.cuda.is_available():
                print(f'CUDA available: {torch.cuda.get_device_name(0)}')
            else:
                print('CUDA not available — running on CPU.')
    except ImportError:
        missing_deps.append('torch')

    if missing_deps:
        print(f'Missing dependencies: {", ".join(missing_deps)}')
        sys.exit(1)
    elif verbose:
        print('Installation OK')


# --------------------------------------------------------------------------
#  Feature generation functions (delegated from semibin intact logic)
# --------------------------------------------------------------------------

def generate_sequence_features_single(logger, contig_fasta,
                                      bams, binned_length,
                                      must_link_threshold, num_process,
                                      output, abundances, only_kmer=False):
    """
    Generate data.csv for single-sample / co-assembly mode.
    """
    import pandas as pd

    if bams is None and abundances is None and not only_kmer:
        logger.error(
            "You need to specify input BAM files or abundance files to calculate coverage features.")
        sys.exit(1)

    if (bams is not None or abundances is not None) and only_kmer:
        logger.info('We will only calculate k-mer features.')

    logger.debug('Generating k-mer features from FASTA file.')
    kmer_whole, contig_lens = generate_kmer_features_from_fasta(
        contig_fasta, binned_length, 4, return_lengths=True)

    if only_kmer:
        kmer_whole['length'] = kmer_whole.index.map(contig_lens).astype(float)
        with atomic_write(os.path.join(output, 'data.csv'), overwrite=True) as ofile:
            kmer_whole.to_csv(ofile)
        return

    if bams:
        is_combined = len(bams) >= 5
        logger.info('Calculating coverage for every sample.')

        with Pool(min(num_process, len(bams))) as pool:
            results = [
                pool.apply_async(
                    generate_cov,
                    args=(bam_file, bam_index, output,
                          is_combined, binned_length, logger, None))
                for bam_index, bam_file in enumerate(bams)]
            for r in results:
                s = r.get()
                logger.info(f'Processed: {s}')

        for bam_index, bam_file in enumerate(bams):
            if not os.path.exists(
                    os.path.join(output,
                                 f'{os.path.split(bam_file)[-1]}_{bam_index}_data_cov.csv')):
                logger.error(f"Generating coverage file failed (for BAM file {bam_file})")
                sys.exit(1)

        data_cov = combine_cov(output, bams, is_combined)

    if abundances:
        if len(abundances) < 5:
            logger.error("Abundances from strobealign-aemb can only be used with at least 5 samples.")
            sys.exit(1)
        logger.info('Reading abundance information from abundance files.')
        data_cov, _ = generate_cov_from_abundances(
            abundances, output, contig_fasta, binned_length)
        is_combined = True

    kmer_whole.index = kmer_whole.index.astype(str)
    data = pd.merge(kmer_whole, data_cov, how='inner',
                    left_index=True, right_index=True, sort=False)
    data['length'] = data.index.map(contig_lens).astype(float)

    with atomic_write(os.path.join(output, 'data.csv'), overwrite=True) as ofile:
        data.to_csv(ofile)


def generate_sequence_features_multi(logger, args):
    """
    Generate data.csv for every sample in multi-sample mode.
    """
    import pandas as pd

    if not args.bams and not args.abundances:
        logger.error("You need to specify input BAM files or abundance files.")
        sys.exit(1)

    n_sample = len(args.bams) if args.bams else len(args.abundances)
    if args.abundances and n_sample < 5:
        logger.error("Abundances from strobealign-aemb can only be used when at least 5 samples are used.")
        sys.exit(1)

    is_combined = n_sample >= 5

    sample_list = []
    has_contigs = False

    os.makedirs(os.path.join(args.output, 'samples'), exist_ok=True)

    def fasta_sample_iter(fn):
        for h, seq in fasta_iter(fn):
            if args.separator not in h:
                raise ValueError(
                    f"Expected contigs to contain separator character ({args.separator}), found {h}")
            sample_name, contig_name = h.split(args.separator, 1)
            yield sample_name, contig_name, seq

    for sample_name, contigs in groupby(fasta_sample_iter(args.contig_fasta),
                                        lambda sn_cn_seq: sn_cn_seq[0]):
        with utils.possibly_compressed_write(
                os.path.join(args.output, 'samples', f'{sample_name}.fa')) as out:
            for _, contig_name, seq in contigs:
                out.write(f'>{contig_name}\n{seq}\n')
                has_contigs = True
        sample_list.append(sample_name)

    if not has_contigs:
        logger.error(f'Input file {args.contig_fasta} is empty. Please check inputs.')
        sys.exit(1)

    binning_threshold = {}
    for sample in sample_list:
        binning_threshold[sample] = utils.maybe_compute_min_length(
            args.min_len, os.path.join(args.output, f'samples/{sample}.fa'), args.ratio)

    if args.bams:
        logger.info('Calculating coverage for every sample.')
        with Pool(min(args.num_process, len(args.bams))) as pool:
            results = [
                pool.apply_async(
                    generate_cov,
                    args=(bam_file, bam_index,
                          os.path.join(args.output, 'samples'),
                          is_combined,
                          binning_threshold, logger, args.separator))
                for bam_index, bam_file in enumerate(args.bams)]
            for r in results:
                s = r.get()
                logger.info(f'Processed: {s}')

        data_cov = combine_cov(
            os.path.join(args.output, 'samples'), args.bams, is_combined)

        data_cov = data_cov.reset_index()
        cols = list(data_cov.columns)
        cols[0] = 'contig_name'
        data_cov.columns = cols

        for sample in sample_list:
            output_path = os.path.join(args.output, 'samples', sample)
            os.makedirs(output_path, exist_ok=True)

            part_data = split_data(data_cov, sample, args.separator, is_combined)
            part_data.to_csv(os.path.join(output_path, 'data_cov.csv'))

            sample_fasta = os.path.join(args.output, f'samples/{sample}.fa')
            kmer_whole, contig_lens = generate_kmer_features_from_fasta(
                sample_fasta, binning_threshold[sample], 4, return_lengths=True)

            sample_cov = pd.read_csv(os.path.join(output_path, 'data_cov.csv'), index_col=0)
            kmer_whole.index = kmer_whole.index.astype(str)
            sample_cov.index = sample_cov.index.astype(str)
            data = pd.merge(kmer_whole, sample_cov, how='inner',
                            left_index=True, right_index=True, sort=False)
            data['length'] = data.index.map(contig_lens).astype(float)

            with atomic_write(os.path.join(output_path, 'data.csv'), overwrite=True) as ofile:
                data.to_csv(ofile)

    if args.abundances:
        logger.info('Reading abundance information from abundance files.')
        abun_split = generate_cov_from_abundances(
            args.abundances, os.path.join(args.output, 'samples'),
            contig_path=args.contig_fasta,
            sep=args.separator,
            sample_contig_threshold=binning_threshold)
        abun_split = abun_split.reset_index()
        cols = list(abun_split.columns)
        cols[0] = 'contig_name'
        abun_split.columns = cols

        for sample in sample_list:
            output_path = os.path.join(args.output, 'samples', sample)
            os.makedirs(output_path, exist_ok=True)
            part_data_split = split_data(abun_split, sample, args.separator, is_combined=True)

            index_name = part_data_split.index.tolist()
            data_index_name = [index_name[2 * t][0:-2] for t in range(len(index_name) // 2)]
            vals = part_data_split.values
            part_data = (vals[::2] + vals[1::2]) / 2
            columns = [f'{a}' for a in args.abundances]
            part_data = pd.DataFrame(part_data, index=data_index_name, columns=columns)
            part_data.to_csv(os.path.join(output_path, 'data_cov.csv'))

            sample_fasta = os.path.join(args.output, f'samples/{sample}.fa')
            kmer_whole, contig_lens = generate_kmer_features_from_fasta(
                sample_fasta, binning_threshold[sample], 4, return_lengths=True)

            data = pd.merge(kmer_whole, part_data, how='inner',
                            left_index=True, right_index=True, sort=False)
            data['length'] = data.index.map(contig_lens).astype(float)

            with atomic_write(os.path.join(output_path, 'data.csv'), overwrite=True) as ofile:
                data.to_csv(ofile)

    return sample_list


# --------------------------------------------------------------------------
#  Training
# --------------------------------------------------------------------------

def training(logger, data, data_split, *, output, device, mode, args):
    """
    Train the NewBin model and save to output/model.pt.

    is_combined is inferred from the last column name of data.csv:
      - ends with '_var'  → is_combined=False (single BAM mode)
      - otherwise         → is_combined=True  (multi-BAM mode)
    """
    from .model import train_model
    import pandas as pd

    if mode == 'single':
        logger.info('Start training from a single sample.')
        data_ = pd.read_csv(data[0], index_col=0)
        col_name = data_.columns[-1].split('_')[-1]
        is_combined = col_name != 'var'
    else:
        logger.info('Start training from multiple samples.')
        is_combined = False

    model = train_model(logger,
                        data,
                        data_split,
                        is_combined,
                        args.batchsize,
                        args.epochs,
                        device,
                        args.num_process,
                        mode)
    model.save_with_params_to(os.path.join(output, 'model.pt'))


# --------------------------------------------------------------------------
#  Binning pre-processing
# --------------------------------------------------------------------------

def binning_preprocess(data_path, model_path, device):
    """Load data.csv and model, infer is_combined and n_sample."""
    import pandas as pd
    from .model import model_load
    logger = logging.getLogger('NewBin')

    data = pd.read_csv(data_path, index_col=0)
    data.index = data.index.astype(str)

    num_cov = data.shape[1] - 136
    if num_cov > 0:
        is_combined = True
        n_sample = max(1, (num_cov - 1) // 2)
    else:
        is_combined = False
        n_sample = 0

    model = model_load(model_path, device)
    return is_combined, n_sample, data, model


def binning_short(logger, data_path, minfasta, binned_length, contig_dict,
                  model_path, output, device, *, args):
    """Short-read binning: embedding + Infomap + optional recluster."""
    from .cluster import cluster
    logger.info('Start binning (short-read).')
    is_combined, n_sample, data, model = binning_preprocess(data_path, model_path, device)
    cluster(logger, model=model, data=data, device=device,
            is_combined=is_combined, n_sample=n_sample,
            out=output, contig_dict=contig_dict,
            args=args, binned_length=binned_length, minfasta=minfasta)


def binning_long(logger, data_path, minfasta, binned_length, contig_dict,
                 model_path, output, device, *, args):
    """Long-read binning: embedding + DBSCAN ensemble."""
    from .long_read_cluster import cluster_long_read
    logger.info('Start binning (long-read).')
    is_combined, n_sample, data, model = binning_preprocess(data_path, model_path, device)
    cluster_long_read(logger, model, data, device, is_combined, n_sample,
                      output, contig_dict, binned_length=binned_length,
                      minfasta=minfasta, args=args)


# --------------------------------------------------------------------------
#  All-in-one commands
# --------------------------------------------------------------------------

def single_easy_binning(logger, args, binned_length, must_link_threshold,
                        contig_dict, device):
    """All-in-one: features → train → bin (single / co-assembly)."""
    os.makedirs(args.output, exist_ok=True)

    # Features
    if args.bams:
        args.bams = maybe_crams2bams(args.bams, args.contig_fasta,
                                     args.num_process, args.output)
    generate_sequence_features_single(
        logger, args.contig_fasta, args.bams, binned_length,
        must_link_threshold, args.num_process, args.output,
        args.abundances)

    # Train
    data_split_path = os.path.join(args.output, 'data_split.csv')
    data_split_list = [data_split_path] if os.path.exists(data_split_path) else None
    training(logger,
             data=[os.path.join(args.output, 'data.csv')],
             data_split=data_split_list,
             output=args.output,
             device=device,
             mode='single',
             args=args)

    # Bin
    if args.sequencing_type == 'short_read':
        binning_short(logger,
                      os.path.join(args.output, 'data.csv'),
                      args.minfasta_kb * 1000,
                      binned_length,
                      contig_dict,
                      os.path.join(args.output, 'model.pt'),
                      args.output,
                      device,
                      args=args)
    else:
        binning_long(logger,
                     os.path.join(args.output, 'data.csv'),
                     args.minfasta_kb * 1000,
                     binned_length,
                     contig_dict,
                     os.path.join(args.output, 'model.pt'),
                     args.output,
                     device,
                     args=args)


def multi_easy_binning(logger, args, device):
    """All-in-one: features → train → bin (multi-sample)."""
    os.makedirs(args.output, exist_ok=True)

    if args.bams:
        args.bams = maybe_crams2bams(args.bams, args.contig_fasta,
                                     args.num_process, args.output)

    sample_list = generate_sequence_features_multi(logger, args)

    for sample in sample_list:
        sample_output = os.path.join(args.output, 'samples', sample)
        _, _, contig_dict = load_fasta(
            os.path.join(args.output, f'samples/{sample}.fa'), args.ratio)
        binned_length = utils.maybe_compute_min_length(
            args.min_len, os.path.join(args.output, f'samples/{sample}.fa'), args.ratio)

        data_split_path = os.path.join(sample_output, 'data_split.csv')
        data_split_list = [data_split_path] if os.path.exists(data_split_path) else None

        training(logger,
                 data=[os.path.join(sample_output, 'data.csv')],
                 data_split=data_split_list,
                 output=sample_output,
                 device=device,
                 mode='single',
                 args=args)

        if args.sequencing_type == 'short_read':
            binning_short(logger,
                          os.path.join(sample_output, 'data.csv'),
                          args.minfasta_kb * 1000,
                          binned_length,
                          contig_dict,
                          os.path.join(sample_output, 'model.pt'),
                          sample_output,
                          device,
                          args=args)
        else:
            binning_long(logger,
                         os.path.join(sample_output, 'data.csv'),
                         args.minfasta_kb * 1000,
                         binned_length,
                         contig_dict,
                         os.path.join(sample_output, 'model.pt'),
                         sample_output,
                         device,
                         args=args)


# --------------------------------------------------------------------------
#  Main entry point
# --------------------------------------------------------------------------

def main():
    parser = newbin_arg_parser()
    args = parser.parse_args()

    if args.cmd is None:
        parser.print_help()
        sys.exit(0)

    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('NewBin')
    if getattr(args, 'verbose', False):
        logger.setLevel(logging.DEBUG)
    elif getattr(args, 'quiet', False):
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    # Validate args
    validate_normalize_args(logger, args)

    # Select device for training/binning
    engine = getattr(args, 'engine', 'auto')
    device = _select_device(engine)
    logger.info(f'Using device: {device}')

    # Prepare output dir
    if hasattr(args, 'output') and args.output is not None:
        os.makedirs(args.output, exist_ok=True)

    # Set random seed
    if hasattr(args, 'random_seed'):
        set_random_seed(args.random_seed)

    # Dispatch commands
    if args.cmd == 'check_install':
        check_install(verbose=getattr(args, 'verbose', False))

    elif args.cmd == 'generate_sequence_features_single':
        c_min_len, must_link_threshold, contig_dict = load_fasta(args.contig_fasta, args.ratio)
        binned_length = args.min_len if args.min_len is not None else c_min_len
        if args.ml_threshold is not None:
            must_link_threshold = args.ml_threshold
        generate_sequence_features_single(
            logger, args.contig_fasta, args.bams, binned_length,
            must_link_threshold, args.num_process, args.output,
            args.abundances, args.kmer)

    elif args.cmd == 'generate_sequence_features_multi':
        generate_sequence_features_multi(logger, args)

    elif args.cmd == 'train_self':
        training(logger,
                 data=args.data,
                 data_split=args.data_split,
                 output=args.output,
                 device=device,
                 mode=args.mode,
                 args=args)

    elif args.cmd == 'bin':
        c_min_len, _, contig_dict = load_fasta(args.contig_fasta, getattr(args, 'ratio', 0.05))
        binned_length = getattr(args, 'min_len', None) or c_min_len
        binning_short(logger, args.data, args.minfasta_kb * 1000,
                      binned_length, contig_dict,
                      args.model_path, args.output, device, args=args)

    elif args.cmd == 'bin_long':
        c_min_len, _, contig_dict = load_fasta(args.contig_fasta, getattr(args, 'ratio', 0.05))
        binned_length = getattr(args, 'min_len', None) or c_min_len
        binning_long(logger, args.data, args.minfasta_kb * 1000,
                     binned_length, contig_dict,
                     args.model_path, args.output, device, args=args)

    elif args.cmd == 'single_easy_bin':
        c_min_len, must_link_threshold, contig_dict = load_fasta(args.contig_fasta, args.ratio)
        binned_length = args.min_len if args.min_len is not None else c_min_len
        if getattr(args, 'ml_threshold', None) is not None:
            must_link_threshold = args.ml_threshold
        single_easy_binning(logger, args, binned_length, must_link_threshold,
                            contig_dict, device)

    elif args.cmd == 'multi_easy_bin':
        multi_easy_binning(logger, args, device)

    elif args.cmd == 'concatenate_fasta':
        from .utils import concatenate_fasta
        ofname = concatenate_fasta(args.contig_fasta, args.min_len,
                                   args.output, args.separator,
                                   getattr(args, 'output_compression', 'none'))
        logger.info(f'Concatenated contigs written to {ofname}')

    else:
        logger.error(f'Unknown subcommand: {args.cmd}')
        sys.exit(1)


if __name__ == '__main__':
    main()
