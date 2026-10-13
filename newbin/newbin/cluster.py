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
    【一句话总结】：
    像安检扫描仪一样，从头到尾流式扫描一个 `.bam` 文件里的所有 Read，专门挑出
    “一头贴在 Contig A、另一头贴在 Contig B”的跨 Contig 配对 Reads (PE Bridging Reads)，
    并统计每一对 `(Contig A, Contig B)` 一共被连了多少次。

    一、BAM 文件的来源与上下游关系（为什么它是 Contig 的前一步数据）：
    ------------------------------------------------------------------
    1. 原始测序 (.fastq) : 测序仪打碎 DNA（约 300~500bp 片段），测出两端各 150bp 的成对短序列 Reads (R1/R2)，
                           这是拼接 Contig 前最原始的测序数据；
    2. 序列拼接 (.fasta) : 组装软件将重叠的 Reads 拼接成较长的 Contig 序列（若中间空白区未知则被迫断开成两条 Contig）；
    3. 回贴比对 (.bam)   : 将第 1 步的原始 Reads 重新比对（Mapping）回第 2 步拼好的 Contig 上，
                           生成本函数输入的 `.bam` 文件（与 Step 2 计算 Coverage 丰度使用的是同一个 BAM 文件，一物两用）。
                           因此，BAM 文件既保留了拼接前的原始配对 Reads，又记录了每条 Read 最终归属的 Contig 名称！

    二、BAM 文件内部记录的数据格式：
    --------------------------------
    BAM 是 SAM 比对格式的二进制压缩版（BGZF），内部包含两部分核心数据：
    1. 文件头 (Header) —— Contig 名称字典 (`refs`)：
       - 记录所有 Contig 的名称列表，例如 `refs = ['g1k_0', 'g1k_1', 'g1k_2', ...]`，将整数编号 `refID` 映射为 Contig 名。
    2. 比对记录列表 (Alignment Records) —— 每条 Read 的归属信息：
       每一行（或每条二进制记录）代表一条短读段 Read，核心字段包括：
       - `QNAME`              : 这条 Read 的名称（配对的左端 R1 和右端 R2 拥有相同的 QNAME）
       - `FLAG` (`flag`)      : 比对状态标记位（如 0x4 表示未比对上，0x100/0x800 表示次级/补充比对）
       - `RNAME` (`refID`)    : ★ 当前这条 Read 比对到的目标 Contig 名称（例如 `'g1k_0'`）
       - `MAPQ` (`mapq`)      : 比对质量打分（分数越高，说明这条 Read 贴到该 Contig 的可信度越高）
       - `RNEXT` (`next_refID`): ★ 与当前 Read 配对的另一端 Mate Read 比对到的目标 Contig 名称（例如 `'g1k_1'`）

    三、函数内部是怎么一步步工作的（双引擎设计）：
    ----------------------------------------------
    代码会自动检测环境，从两条做同一件事的路线中二选一执行：
    1. 路线 1（引擎一，优先）：若系统装有 `samtools`，调用 `samtools view -F 0x904 -q min_mapq` 将二进制 BAM
       转为按制表符分隔的文本流，逐行提取第 3 列 `rname` 和第 7 列 `rnext`：
       - 若 `rnext == '='`（两端在同一条 Contig 上，占 99% 以上情况）或 `rnext == '*'`（另一端未贴上） -> 直接跳过；
       - 若 `rnext != rname`，说明抓到一次跨 Contig 搭桥 -> 排序为无向对 `(contig_A, contig_B)` 并计数 +1。
    2. 路线 2（引擎二，兜底）：若在 Windows 等未安装 `samtools` 的环境下，自动使用内置纯 Python 二进制解析（分 3 步）：
       - 第 1 步 `_get_blocks()` : 流式分块读取并解压 BAM 的 BGZF 压缩块（随读随解压，哪怕几十 GB 的 BAM 也不会撑爆内存）；
       - 第 2 步 步骤 A (读文件头): 从开头解析出 Contig 数量 `n_ref`，构建编号到名称的列表 `refs = ['g1k_0', 'g1k_1', ...]`；
       - 第 3 步 步骤 B (读记录)  : 循环解包每条 Read 记录的前 24 个字节，拿出 `refID`、`next_refID`、`mapq`、`flag`，
                                   当质量达标、主比对正常且 `next_refID != refID` 时，用 `refs[refID]` 和 `refs[next_refID]`
                                   查表还原出两条 Contig 的名称并计数 +1。

    四、算出来的返回结果 `counts` 在后续流程中有什么用？
    ----------------------------------------------------
    1. 给神经网络当“标准答案”（Must-Link 正样本对）：
       在训练阶段（Step 4 & Step 5），被多条配对 Reads 连接的两条 Contig 会作为正样本对，引导神经网络将它们的特征向量拉近。
    2. 在最终聚类图上“焊死”物理强边（Step 6）：
       在构建双轨 KNN 图进行 Leiden 聚类时，直接在这两条 Contig 之间加上高权重的物理边 `w_pe = α * tanh(count / θ)`，
       确保它们被划分进同一个菌株（Bin）中。

    Parameters
    ----------
    bam_path : str
        单个样本的 `.bam` 比对文件路径（与 Step 2 计算 Coverage 传入的 BAM 文件相同）。
    min_mapq : int, default=20
        最小比对质量阈值，过滤掉贴得模棱两可、可能贴错位置的劣质 Read。

    Returns
    -------
    counts : collections.Counter
        键为按字母排序的 Contig 名称二元组 `(contig_1, contig_2)`，值为跨序列配对 Reads 的计数。
        示例: `Counter({('g1k_0', 'g1k_1'): 15, ('g1k_3', 'g1k_8'): 6})`
    """
    import subprocess
    from collections import Counter
    import shutil
    import struct
    import zlib

    counts = Counter()

    # -----------------------------------------------------------------------
    # 路线 1 (引擎 1): 优先尝试调用系统 samtools view (流式文本解析)
    #   -F 0x904 : 过滤掉未比对上 (0x4)、次级比对 (0x100) 和补充比对 (0x800) 的记录
    #   -q       : 要求最小比对质量分数 MAPQ >= min_mapq
    # -----------------------------------------------------------------------
    if shutil.which('samtools'):
        cmd = ['samtools', 'view', '-F', '0x904', '-q', str(min_mapq), bam_path]
        try:
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
                for line in proc.stdout:
                    cols = line.strip().split('\t')
                    if len(cols) < 7:
                        continue
                    rname = cols[2]  # 第 3 列 (RNAME): 当前这端 Read 贴在哪个 Contig 上 (如 'g1k_0')
                    rnext = cols[6]  # 第 7 列 (RNEXT): 配对的另一端 Read 贴在哪个 Contig 上 (如 'g1k_1'、'=' 或 '*')
                    # '=' 表示两端贴在同一条 Contig 上（绝大多数情况），跳过；
                    # '*' 表示配对的另一端根本没比对上任何 Contig，跳过；
                    # 只有当 rnext 是另一个不同的 Contig 名字时，才说明抓到了一条跨 Contig 搭桥！
                    if rnext != '=' and rnext != '*' and rnext != rname:
                        # 用 sorted 排序，确保 ('g1k_0', 'g1k_1') 和 ('g1k_1', 'g1k_0') 归为同一个无向对
                        pair = tuple(sorted([rname, rnext]))
                        counts[pair] += 1
                return counts
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # 路线 2 (引擎 2): 内置纯 Python BGZF 流式二进制解析兜底（Windows 等无 samtools 环境自动走这里）
    # -----------------------------------------------------------------------
    try:
        with open(bam_path, 'rb') as f:
            buf = bytearray()

            # 第 1 步: 流式分块解压 BAM 的 BGZF (分块 Gzip) 压缩数据块，随用随取，避免大文件撑爆内存
            def _get_blocks():
                while True:
                    hdr = f.read(18)
                    if len(hdr) < 18 or hdr[:2] != b'\x1f\x8b':
                        break
                    bsize = struct.unpack('<H', hdr[16:18])[0] + 1
                    payload = f.read(bsize - 18)
                    yield zlib.decompress(payload[:-8], -15)

            gen = _get_blocks()

            # 第 2 步 (步骤 A): 解析 BAM 文件头 (Header)，构建 Contig 编号映射表 refs
            #   二进制 BAM 为了节省体积，不在每条 Read 里重复存 Contig 字符串，而是存整数下标 refID。
            #   这里先把所有 Contig 的名字按顺序读进列表 refs = ['g1k_0', 'g1k_1', ...]，
            #   后续通过 refs[refID] 即可查表还原出对应的 Contig 名称。
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
                pos += l_name + 4  # 跳过 l_ref (该 Contig 的序列长度)
                refs.append(name)
            buf = buf[pos:]

            # 第 3 步 (步骤 B): 逐条解析 BAM 二进制比对记录 (Alignment Records)
            #   每条 Read 的核心比对信息都在记录开头的前 24 个字节里，通过 struct.unpack 拆出 4 个关键数字：
            #   - refID      : 当前 Read 所属 Contig 的编号 (如 0) -> 查表 refs[refID] 得 Contig 名
            #   - next_refID : 配对 Read 所属 Contig 的编号 (如 1) -> 查表 refs[next_refID] 得 Contig 名
            #   - mapq       : 比对质量分数 (要求 >= min_mapq)
            #   - flag       : 状态标记位 (要求 (flag & 0x904) == 0，即主比对且成功比对上)
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
                    # 解包比对记录前 24 字节：提取当前 Contig 编号 refID、质量 mapq、标记位 flag、配对端 Contig 编号 next_refID
                    refID, _, _, mapq, _, _, flag, _, next_refID = struct.unpack('<iiBBHHHii', rec[:24])
                    # 校验质量合格、主比对记录，且两端贴在不同的 Contig 上 (next_refID != refID)
                    if mapq >= min_mapq and (flag & 0x904) == 0:
                        if 0 <= next_refID < len(refs) and 0 <= refID < len(refs) and next_refID != refID:
                            # 查表还原成 Contig 名字并按字母排序，给该无向 Contig 对计数 +1
                            pair = tuple(sorted([refs[refID], refs[next_refID]]))
                            counts[pair] += 1

            # 处理解压缓冲区中最后剩余的比对记录（逻辑同上）
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

    一、参数 `k_neighbors` 的核心作用与原理：
    -------------------------------------------
    1. 什么是 `k_neighbors`（默认值 15）：
       规定在 100 维 Embedding 特征空间中，每条 Contig 只跟与自己特征最相似的前 k 个“最亲密邻居”连线，
       用来构建一张稀疏的 k-NN 邻近图。

    2. 为什么必须限制 k 值（若不限制会有什么灾难）：
       - 避免内存爆炸 (OOM)：真实宏基因组往往有数万（如 50,000）条 Contig，若两两全连接，边数高达
         50000 * 49999 / 2 ≈ 12.5 亿条边，内存直接被撑爆；
       - 形成清晰的物种边界：同物种 Contig 互为前 k 个最近邻自然抱团成簇；若全连接，不同物种间也会连满
         微弱噪声边，导致整个图变成“一锅粥”，下游 Leiden 聚类算法将完全无法切分出菌株边界。

    3. 调参直觉与 trade-off：
       - k 设得太小（如 k=2~3）：图过于稀疏，同物种 Contig 容易断开，导致 Bin 被打得过于零碎（分箱过度碎片化）；
       - k 设得太大（如 k=50~100）：图过于密集，容易引入跨物种的弱相似虚假边，导致不同细菌被混在一起（污染率 Contamination 飙升）；
       - 默认值 15：宏基因组图分箱（包括 SemiBin 等）公认的黄金经验值，兼顾连通性与高纯度。

    4. 代码细节：为什么代码中用 `actual_k = min(k_neighbors + 1, num_contigs)`？
       因为在特征空间计算近邻时，距离最近的第 1 名永远是 Contig 自身（距离为 0）。所以必须多取 1 个名额，
       把第 1 名（自己）跳过后，剩下的恰好是前 k 个不重复的邻居。

    二、双轨图的图层融合逻辑：
    --------------------------
    - 轨道 1 (语义骨架): 基于 100 维 Embedding 的余弦相似度构建 k-NN 边，边权为 cos_sim ∈ [0, 1]；
    - 轨道 2 (物理增强): 将 BAM 中提取的跨 Contig PE Reads 计数 count 通过平滑饱和函数
                         w_pe = α * tanh(count / θ) 注入为强物理边；
    - 约束剪枝 (Cannot-link): 强制删除先验互斥对之间的连边，防止不同物种被错误聚到同一个 Bin。

    Parameters
    ----------
    embedding : np.ndarray, shape (N, 100)
        全体 Contig 的 100 维特征向量矩阵
    contig_names : list of str
        Contig 名称列表
    pe_links : dict, optional
        跨 Contig 配对 Reads 计数字典 {('contig_A', 'contig_B'): count}
    k_neighbors : int, default=15
        每个节点连向的最相似邻居数量（控制图稀疏度）
    pe_alpha : float, default=0.9
        物理边权重上限饱和系数
    pe_theta : float, default=3.0
        物理边权重饱和速度缩放参数
    cannot_pairs : list of (str, str), optional
        先验不能连的 Contig 名称对
    logger : logging.Logger, optional

    Returns
    -------
    g : igraph.Graph
        构建完成的双轨增强无向图
    weights : np.ndarray, dtype=np.float32
        图上每条边对应的权重数组 (取值范围 0~1)
    """
    from sklearn.neighbors import NearestNeighbors
    from igraph import Graph
    import numpy as np

    num_contigs = len(contig_names)
    name2idx = {name: idx for idx, name in enumerate(contig_names)}

    if num_contigs < 2:
        g = Graph(n=num_contigs)
        return g, np.array([], dtype=np.float32)

    # -----------------------------------------------------------------------
    # 轨道 1: 全局语义骨架 (基于特征余弦相似度的 k-NN 图)
    # -----------------------------------------------------------------------
    # L2 归一化特征向量，使欧氏空间点积等价于余弦相似度
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_emb = embedding / norms

    # 取 k_neighbors + 1 个近邻（因为包含自身，自身排在第 1 位）
    actual_k = min(k_neighbors + 1, num_contigs)

    # -----------------------------------------------------------------------
    # 无监督 KNN 最近邻搜索（类似“地图搜附近”的三步过程，非模型训练）：
    #   1. NearestNeighbors(...) : 配置搜索规则（按余弦距离 metric='cosine' 找最近的 actual_k 个点）；
    #   2. nn.fit(norm_emb)      : 将全体 Contig 的 100 维特征坐标导入并保存在底库中（只存坐标做索引，无参数训练）；
    #   3. nn.kneighbors(...)    : 正式查询，输出 indices (邻居编号, shape [N, actual_k]) 与 distances (余弦距离)。
    # -----------------------------------------------------------------------
    nn = NearestNeighbors(n_neighbors=actual_k, metric='cosine')
    nn.fit(norm_emb)  # 将所有 Contig 坐标存入搜索底库
    distances, indices = nn.kneighbors(norm_emb)  # 查出每条 Contig 最近的 actual_k 个邻居与距离

    edge_dict = {}

    for i in range(num_contigs):
        for dist, j in zip(distances[i], indices[i]):
            # 排除自身到自身的自环边 (i == j)
            if i == j:
                continue
            # 余弦距离转为余弦相似度：cos_sim = 1 - cosine_distance
            cos_sim = max(0.0, float(1.0 - dist))
            pair = tuple(sorted([i, j]))
            # 若两个方向都检索到了对方，保留最大的相似度作为无向边权重
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
