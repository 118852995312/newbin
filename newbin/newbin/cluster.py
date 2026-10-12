import os
import shutil
import math
import tempfile

from .utils import write_bins, log_binning_stats
from .markers import estimate_seeds

# This is the default in the igraph package
NR_INFOMAP_TRIALS = 10

def cal_kl(m, v, use_ne='auto'):
    # A naive implementation creates a lot of copies of what can
    # become large matrices
    import numpy as np

    m = np.clip(m, 1e-6, None)
    v = np.clip(v, 1.0, None)

    m1 = m.reshape(1, len(m))
    m2 = m.reshape(len(m), 1)

    v1 = v.reshape(1, len(v))
    v2 = v.reshape(len(v), 1)


    if use_ne != 'no':
        try:
            import numexpr as ne

            res = ne.evaluate(
                    '(log(v1) - log(v2))/2 + ( (m1 - m2)**2 + v2 ) / ( 2 * v1 ) - half',
                    {
                        'v1': v1,
                        'v2': v2,
                        'm1': m1,
                        'm2': m2,
                        # numexpr rules are that mixing with floats causes
                        # conversion to float64
                        # Note that this does not happen for integers
                        'half': np.float32(0.5),
                    })
            np.clip(res, 1e-6, 1-1e-6, out=res)
            return res
        except ImportError:
            if use_ne != 'auto':
                raise
    v_div = np.log(v1) - np.log(v2)
    v_div /= 2.0

    m_dif = m1 - m2
    m_dif **=2
    m_dif += v2
    m_dif /= 2 * v1

    v_div += m_dif
    v_div -= 0.5
    np.clip(v_div, 1e-6, 1-1e-6, out=v_div)
    return v_div


# ---------------------------------------------------------------------------
#  双轨增强图构建与 Leiden 社区划分
# ---------------------------------------------------------------------------

def parse_bam_pe_reads_stream(bam_path: str, min_mapq: int = 20):
    """
    流式读取 BAM 文件，提取跨 contig 的配对读段 (RNAME, RNEXT)。
    双引擎设计：
    1. 优先使用系统 samtools view (流式、高效)；
    2. 无外部命令时使用内置纯 Python BGZF 解压与 BAM 二进制解析兜底。
    """
    import subprocess
    from collections import Counter
    import shutil
    import struct
    import zlib

    counts = Counter()

    # 1. 尝试使用 samtools view
    if shutil.which('samtools'):
        cmd = ['samtools', 'view', '-F', '0x904', '-q', str(min_mapq), bam_path]
        try:
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
                for line in proc.stdout:
                    cols = line.strip().split('\t')
                    if len(cols) < 7:
                        continue
                    rname = cols[2]
                    rnext = cols[6]
                    if rnext != '=' and rnext != '*' and rnext != rname:
                        pair = tuple(sorted([rname, rnext]))
                        counts[pair] += 1
                return counts
        except Exception:
            pass

    # 2. 内置纯 Python BGZF 流式二进制解析兜底
    try:
        with open(bam_path, 'rb') as f:
            buf = bytearray()

            def _get_blocks():
                while True:
                    hdr = f.read(18)
                    if len(hdr) < 18 or hdr[:2] != b'\x1f\x8b':
                        break
                    bsize = struct.unpack('<H', hdr[16:18])[0] + 1
                    payload = f.read(bsize - 18)
                    yield zlib.decompress(payload[:-8], -15)

            gen = _get_blocks()

            # 解析 BAM Header
            while len(buf) < 8:
                buf.extend(next(gen))
            l_text = struct.unpack('<I', buf[4:8])[0]
            while len(buf) < 8 + l_text + 4:
                buf.extend(next(gen))
            pos = 8 + l_text
            n_ref = struct.unpack('<I', buf[pos:pos+4])[0]
            pos += 4
            refs = []
            for _ in range(n_ref):
                while len(buf) < pos + 4:
                    buf.extend(next(gen))
                l_name = struct.unpack('<I', buf[pos:pos+4])[0]
                pos += 4
                while len(buf) < pos + l_name + 4:
                    buf.extend(next(gen))
                name = buf[pos:pos+l_name-1].decode('utf-8', errors='ignore')
                pos += l_name + 4  # 跳过 l_ref
                refs.append(name)
            buf = buf[pos:]

            # 流式解析比对记录
            for block in gen:
                buf.extend(block)
                while len(buf) >= 4:
                    bsize = struct.unpack('<I', buf[:4])[0]
                    if len(buf) < 4 + bsize:
                        break
                    rec = buf[4:4+bsize]
                    buf = buf[4+bsize:]
                    if len(rec) < 24:
                        continue
                    refID, _, _, mapq, _, _, flag, _, next_refID = struct.unpack('<iiBBHHHii', rec[:24])
                    if mapq >= min_mapq and (flag & 0x904) == 0:
                        if 0 <= next_refID < len(refs) and 0 <= refID < len(refs) and next_refID != refID:
                            pair = tuple(sorted([refs[refID], refs[next_refID]]))
                            counts[pair] += 1

            while len(buf) >= 4:
                bsize = struct.unpack('<I', buf[:4])[0]
                if len(buf) < 4 + bsize:
                    break
                rec = buf[4:4+bsize]
                buf = buf[4+bsize:]
                if len(rec) < 24:
                    continue
                refID, _, _, mapq, _, _, flag, _, next_refID = struct.unpack('<iiBBHHHii', rec[:24])
                if mapq >= min_mapq and (flag & 0x904) == 0:
                    if 0 <= next_refID < len(refs) and 0 <= refID < len(refs) and next_refID != refID:
                        pair = tuple(sorted([refs[refID], refs[next_refID]]))
                        counts[pair] += 1
    except Exception:
        pass

    return counts


def extract_pe_bridging_links(
    bams=None,
    contig_names=None,
    min_mapq: int = 20,
    min_count: int = 3,
    cache_file=None,
    must_link_export_file=None,
    must_link_min_count: int = 5,
    logger=None
):
    """
    提取跨样本累加的 PE Bridging Reads 跨 contig 物理连通对。
    """
    import pandas as pd
    from collections import Counter

    total_counts = Counter()
    contig_set = set(contig_names) if contig_names is not None else set()

    # 1. 尝试从缓存读取
    if cache_file and os.path.exists(cache_file):
        if logger:
            logger.info(f"从缓存文件读取 PE Bridging Links: {cache_file}")
        try:
            df = pd.read_csv(cache_file)
            for _, row in df.iterrows():
                c1, c2, cnt = str(row.iloc[0]).strip(), str(row.iloc[1]).strip(), int(row.iloc[2])
                if (not contig_set or (c1 in contig_set and c2 in contig_set)) and c1 != c2:
                    pair = tuple(sorted([c1, c2]))
                    total_counts[pair] += cnt
            filtered_pe = {p: c for p, c in total_counts.items() if c >= min_count}

            # 衍生价值：导出高置信度天然 Must-link
            if must_link_export_file:
                strong_pe = [p for p, c in total_counts.items() if c >= must_link_min_count]
                if len(strong_pe) > 0:
                    try:
                        os.makedirs(os.path.dirname(os.path.abspath(must_link_export_file)), exist_ok=True)
                        with open(must_link_export_file, 'w') as f:
                            for p in strong_pe:
                                f.write(f"{p[0]},{p[1]}\n")
                        if logger:
                            logger.info(f"已自动导出 {len(strong_pe)} 对高置信度物理边至: {must_link_export_file}")
                    except Exception as e:
                        if logger:
                            logger.warning(f"导出 Must-link 物理连接对失败: {e}")

            return filtered_pe
        except Exception as e:
            if logger:
                logger.warning(f"读取 PE 缓存文件失败 ({e})，重新从 BAM 提取...")

    if not bams:
        return {}

    bam_list = [bams] if isinstance(bams, str) else bams

    if logger:
        logger.info(f"正在从 {len(bam_list)} 个 BAM 文件流式提取 PE Bridging Reads (MAPQ >= {min_mapq})...")

    for bam in bam_list:
        if not bam or not os.path.exists(bam):
            continue
        sub_counts = parse_bam_pe_reads_stream(bam, min_mapq=min_mapq)
        for pair, cnt in sub_counts.items():
            if not contig_set or (pair[0] in contig_set and pair[1] in contig_set):
                total_counts[pair] += cnt

    # 过滤 C_ij >= min_count
    filtered_pe = {pair: count for pair, count in total_counts.items() if count >= min_count}

    if logger:
        logger.info(f"PE Bridging Reads 提取完成: 满足阈值 θ >= {min_count} 的物理连通对数={len(filtered_pe)}")

    # 导出缓存文件
    if cache_file and len(filtered_pe) > 0:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
            records = [[p[0], p[1], c] for p, c in filtered_pe.items()]
            pd.DataFrame(records, columns=['contig_1', 'contig_2', 'count']).to_csv(cache_file, index=False)
            if logger:
                logger.info(f"PE Bridging Links 已缓存至: {cache_file}")
        except Exception as e:
            if logger:
                logger.warning(f"保存 PE 缓存文件失败: {e}")

    # 衍生价值：导出高置信度天然 Must-link
    if must_link_export_file:
        strong_pe = [p for p, c in total_counts.items() if c >= must_link_min_count]
        if len(strong_pe) > 0:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(must_link_export_file)), exist_ok=True)
                with open(must_link_export_file, 'w') as f:
                    for p in strong_pe:
                        f.write(f"{p[0]},{p[1]}\n")
                if logger:
                    logger.info(f"已自动导出 {len(strong_pe)} 对高置信度物理边至: {must_link_export_file}")
            except Exception as e:
                if logger:
                    logger.warning(f"导出 Must-link 物理连接对失败: {e}")

    return filtered_pe


def build_dual_track_graph(
    embedding,
    contig_names,
    pe_links=None,
    k_neighbors: int = 15,
    pe_alpha: float = 0.9,
    pe_theta: float = 3.0,
    cannot_pairs=None,
    logger=None
):
    """
    构建双轨增强图 (k-NN 全局语义骨架 + 成对 Reads 局部物理连通补强)。
    """
    from sklearn.neighbors import NearestNeighbors
    from igraph import Graph
    import numpy as np

    num_contigs = len(contig_names)
    name2idx = {name: idx for idx, name in enumerate(contig_names)}

    if num_contigs < 2:
        g = Graph(n=num_contigs)
        return g, np.array([], dtype=np.float32)

    # 1. 全局语义骨架 (k-NN 图)
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_emb = embedding / norms

    actual_k = min(k_neighbors + 1, num_contigs)
    nn = NearestNeighbors(n_neighbors=actual_k, metric='cosine')
    nn.fit(norm_emb)
    distances, indices = nn.kneighbors(norm_emb)

    edge_dict = {}

    for i in range(num_contigs):
        for dist, j in zip(distances[i], indices[i]):
            if i == j:
                continue
            cos_sim = max(0.0, float(1.0 - dist))
            pair = tuple(sorted([i, j]))
            if pair not in edge_dict or cos_sim > edge_dict[pair]:
                edge_dict[pair] = cos_sim

    num_knn_edges = len(edge_dict)

    # 2. 局部物理连通补强 (PE Bridging Reads)
    num_pe_injected = 0
    if pe_links:
        for (c1, c2), count in pe_links.items():
            if c1 in name2idx and c2 in name2idx:
                u, v = name2idx[c1], name2idx[c2]
                if u == v:
                    continue
                pair = tuple(sorted([u, v]))
                w_pe = float(pe_alpha * np.tanh(count / pe_theta))
                if pair in edge_dict:
                    edge_dict[pair] = max(edge_dict[pair], w_pe)
                else:
                    edge_dict[pair] = w_pe
                    num_pe_injected += 1

    # 3. 先验互斥对 (Cannot-link) 强制切断
    num_cut = 0
    if cannot_pairs:
        for c1, c2 in cannot_pairs:
            if c1 in name2idx and c2 in name2idx:
                pair = tuple(sorted([name2idx[c1], name2idx[c2]]))
                if pair in edge_dict:
                    del edge_dict[pair]
                    num_cut += 1

    if logger:
        logger.info(
            f"双轨图构建完成: 节点数={num_contigs}, k-NN骨架边={num_knn_edges}, "
            f"PE物理直连边={num_pe_injected}, Cannot-link切断边={num_cut}, 总边数={len(edge_dict)}"
        )

    edges = list(edge_dict.keys())
    weights = np.array(list(edge_dict.values()), dtype=np.float32)

    g = Graph(n=num_contigs)
    if len(edges) > 0:
        g.add_edges(edges)

    return g, weights


def run_leiden_clustering(
    g,
    edge_weights,
    vertex_weights=None,
    resolution: float = 1.0,
    n_iterations: int = 5,
    objective_function: str = 'modularity'
):
    """
    运行 Leiden 算法完成社区切分。
    """
    import numpy as np

    num_nodes = g.vcount()
    if num_nodes == 0:
        return np.array([], dtype=int)
    if g.ecount() == 0:
        return np.arange(num_nodes, dtype=int)

    weights_list = edge_weights.tolist() if isinstance(edge_weights, np.ndarray) else edge_weights
    node_weights_list = vertex_weights.tolist() if vertex_weights is not None else None

    clustering = g.community_leiden(
        objective_function=objective_function,
        weights=weights_list,
        resolution=resolution,
        n_iterations=n_iterations,
        node_weights=node_weights_list
    )

    contig_labels = np.zeros(num_nodes, dtype=int)
    for cluster_id, community_nodes in enumerate(clustering):
        for node in community_nodes:
            contig_labels[node] = cluster_id

    return contig_labels


def run_dual_track_leiden(
    logger,
    model,
    data,
    *,
    device,
    contig_dict,
    bams=None,
    k_neighbors: int = 15,
    pe_min_count: int = 3,
    pe_alpha: float = 0.9,
    pe_theta: float = 3.0,
    pe_min_mapq: int = 20,
    pe_links_file=None,
    pe_cache_file=None,
    pe_must_link_file=None,
    cannot_link=None,
    resolution: float = 1.0,
    num_process: int = 1,
    **kwargs
):
    """
    双轨增强图 + Leiden 聚类核心入口。
    """
    import torch
    import numpy as np
    import pandas as pd

    contig_names = list(data.index.astype(str))

    # 1. 深度模型推理生成 100 维特征嵌入 (统一输入分布)
    with torch.no_grad():
        model.eval()
        x_tensor = torch.from_numpy(data.values.astype(np.float32)).to(device)
        embedding = model.embedding(x_tensor).detach().cpu().numpy()

    # 2. 提取/读取 PE 物理连接边
    pe_links = {}
    if pe_links_file and os.path.exists(pe_links_file):
        pe_links = extract_pe_bridging_links(
            bams=None,
            contig_names=set(contig_names),
            cache_file=pe_links_file,
            logger=logger
        )
    elif bams:
        pe_links = extract_pe_bridging_links(
            bams=bams,
            contig_names=set(contig_names),
            min_mapq=pe_min_mapq,
            min_count=pe_min_count,
            cache_file=pe_cache_file,
            must_link_export_file=pe_must_link_file,
            logger=logger
        )

    # 3. 解析 Cannot-link 约束
    cannot_pairs = []
    if cannot_link:
        c_files = [cannot_link] if isinstance(cannot_link, str) else cannot_link
        for cf in c_files:
            if cf and os.path.exists(cf):
                try:
                    raw_c = pd.read_csv(cf, header=None).values
                    for row in raw_c:
                        c1, c2 = str(row[0]).strip(), str(row[1]).strip()
                        if c1 in contig_names and c2 in contig_names:
                            cannot_pairs.append((c1, c2))
                except Exception as e:
                    if logger:
                        logger.warning(f"读取 cannot_link 文件 {cf} 出错: {e}")

    # 4. 构建双轨融合图
    g, edge_weights = build_dual_track_graph(
        embedding=embedding,
        contig_names=contig_names,
        pe_links=pe_links,
        k_neighbors=k_neighbors,
        pe_alpha=pe_alpha,
        pe_theta=pe_theta,
        cannot_pairs=cannot_pairs,
        logger=logger
    )

    # 5. Leiden 社区切分 (节点以 contig 长度加权)
    length_weight = np.array([len(contig_dict.get(name, '')) for name in contig_names], dtype=float)
    if length_weight.sum() == 0:
        length_weight = None

    contig_labels = run_leiden_clustering(
        g=g,
        edge_weights=edge_weights,
        vertex_weights=length_weight,
        resolution=resolution
    )

    return embedding, contig_labels


# 保持向后兼容的别名
run_embed_infomap = run_dual_track_leiden


def recluster_bins(logger, data, *,
                   embedding,
                   contig_labels,
                   minfasta,
                   contig_dict,
                   binned_length,
                   orf_finder,
                   num_process,
                   random_seed,
                   **kwargs):
    """
    对已完成初次聚类的 bin 基于单拷贝核心标记基因进行二次重聚类。
    """
    from sklearn.cluster import KMeans
    import numpy as np
    from collections import defaultdict

    embedding_new = embedding

    total_size = defaultdict(int)
    for i, c in enumerate(contig_labels):
        total_size[c] += len(contig_dict[data.index[i]])
    with tempfile.TemporaryDirectory() as tdir:
        cfasta = os.path.join(tdir, 'concatenated.fna')
        with open(cfasta, 'wt') as concat_out:
            for ix, h in enumerate(data.index):
                bin_ix = contig_labels[ix]
                if total_size[bin_ix] < minfasta:
                    continue
                concat_out.write(f'>bin{bin_ix:06}.{h}\n')
                concat_out.write(contig_dict[data.index[ix]] + '\n')

        seeds = estimate_seeds(
            cfasta,
            binned_length,
            num_process,
            multi_mode=True,
            orf_finder=orf_finder)

        if not seeds:
            logger.warning('No bins found in the concatenated fasta file.')
            return contig_labels

    name2ix = {name: ix for ix, name in enumerate(data.index)}
    contig_labels_reclustered = np.empty_like(contig_labels)
    contig_labels_reclustered.fill(-1)
    next_label = 0
    for bin_ix in range(contig_labels.max() + 1):
        seed = seeds.get(f'bin{bin_ix:06}', [])
        num_bin = len(seed)

        if num_bin > 1 and total_size[bin_ix] >= minfasta:
            contig_indices = [i for i, ell in enumerate(contig_labels) if ell == bin_ix]
            re_bin_features = embedding_new[contig_indices]

            seed_index = [name2ix[s] for s in seed]
            length_weight = np.array(
                [len(contig_dict[name])
                    for name, ell in zip(data.index, contig_labels, strict=True)
                        if ell == bin_ix])
            seeds_embedding = embedding_new[seed_index]
            kmeans = KMeans(
                n_clusters=num_bin,
                init=seeds_embedding,
                n_init=1,
                random_state=random_seed)
            kmeans.fit(re_bin_features, sample_weight=length_weight)
            for i, label in enumerate(kmeans.labels_):
                contig_labels_reclustered[contig_indices[i]] = next_label + label
            next_label += num_bin
        else:
            contig_labels_reclustered[contig_labels == bin_ix] = next_label
            next_label += 1
    assert contig_labels_reclustered.min() >= 0
    return contig_labels_reclustered


def cluster(logger, model, data, device, is_combined,
            n_sample, out, contig_dict, *, args,
            binned_length, minfasta):
    """
    使用双轨增强图与 Leiden 算法完成分箱聚类。
    """
    import pandas as pd
    from os import path

    k_neighbors = getattr(args, 'k_neighbors', 15)
    pe_min_count = getattr(args, 'pe_min_count', 3)
    pe_alpha = getattr(args, 'pe_alpha', 0.9)
    pe_theta = getattr(args, 'pe_theta', 3.0)
    pe_min_mapq = getattr(args, 'pe_min_mapq', 20)
    resolution = getattr(args, 'leiden_resolution', 1.0)
    pe_links = getattr(args, 'pe_links', None)
    bams = getattr(args, 'bams', None)
    cannot_link = getattr(args, 'cannot_link', None)
    pe_cache = os.path.join(out, 'pe_links.csv')
    pe_must_link = os.path.join(out, 'pe_must_link.csv')

    embedding, contig_labels = run_dual_track_leiden(
        logger, model, data,
        device=device,
        contig_dict=contig_dict,
        bams=bams,
        k_neighbors=k_neighbors,
        pe_min_count=pe_min_count,
        pe_alpha=pe_alpha,
        pe_theta=pe_theta,
        pe_min_mapq=pe_min_mapq,
        pe_links_file=pe_links,
        pe_cache_file=pe_cache,
        pe_must_link_file=pe_must_link,
        cannot_link=cannot_link,
        resolution=resolution,
        num_process=args.num_process
    )

    if args.write_pre_reclustering_bins or not args.recluster:
        output_bin_path = os.path.join(out,
                    'output_prerecluster_bins' if args.recluster else 'output_bins')
        if os.path.exists(output_bin_path):
            logger.warning(f'Previous output directory `{output_bin_path}` found. Over-writing it.')
            shutil.rmtree(output_bin_path)
        os.makedirs(output_bin_path, exist_ok=True)

        bin_files = write_bins(data.index.tolist(),
                            contig_labels,
                            output_bin_path,
                            contig_dict,
                            minfasta=minfasta,
                            output_tag=args.output_tag,
                            output_compression=args.output_compression)
        if not len(bin_files):
            logger.warning('No bins were created. Please check your input data.')
            return
        if not args.recluster:
            logger.info(f'Number of bins: {len(bin_files)}')
            bin_files.to_csv(os.path.join(out, 'bins_info.tsv'), index=False, sep='\t')
            pd.DataFrame({'contig': data.index, 'bin': contig_labels}).to_csv(
                os.path.join(out, 'contig_bins.tsv'), index=False, sep='\t')
            log_binning_stats(logger, bin_files, contig_dict, binned_length)
        n_pre_bins = len(bin_files)
    else:
        from collections import defaultdict
        total_size = defaultdict(int)
        for i, c in enumerate(contig_labels):
            total_size[c] += len(contig_dict[data.index[i]])
        n_pre_bins = sum((total_size[bin_ix] >= minfasta for bin_ix in range(contig_labels.max() + 1)))

    if args.recluster:
        logger.info(f'Number of bins prior to reclustering: {n_pre_bins}')
        logger.debug('Reclustering...')

        if n_pre_bins == 0:
            import numpy as np
            logger.warning('No bins were created. Please check your input data.')
            contig_labels_reclustered = np.full(len(data.index), fill_value=-1, dtype=int)
        else:
            contig_labels_reclustered = recluster_bins(
                logger,
                data,
                embedding=embedding,
                contig_labels=contig_labels,
                contig_dict=contig_dict,
                minfasta=minfasta,
                binned_length=binned_length,
                num_process=args.num_process,
                orf_finder=args.orf_finder,
                random_seed=args.random_seed
            )
        output_recluster_bin_path = path.join(out,
                        ('output_recluster_bins'
                            if args.write_pre_reclustering_bins
                            else 'output_bins'))
        if os.path.exists(output_recluster_bin_path):
            logger.warning(f'Previous output directory `{output_recluster_bin_path}` found. Over-writing it.')
            shutil.rmtree(output_recluster_bin_path)
        os.makedirs(output_recluster_bin_path, exist_ok=True)
        outputs = write_bins(data.index.tolist(),
                            contig_labels_reclustered,
                            output_recluster_bin_path,
                            contig_dict,
                            minfasta=minfasta,
                            output_tag=args.output_tag,
                            output_compression=args.output_compression)
        logger.info(f'Number of bins after reclustering: {len(outputs)}')
        outputs.to_csv(os.path.join(out, 'recluster_bins_info.tsv'), index=False, sep='\t')
        pd.DataFrame({'contig': data.index, 'bin': contig_labels_reclustered}).to_csv(
            os.path.join(out, 'contig_bins.tsv'), index=False, sep='\t')
        log_binning_stats(logger, outputs, contig_dict, binned_length)
    logger.info('Binning finished')
