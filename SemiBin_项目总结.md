# SemiBin 项目完整文档

> 本文档分四部分：
> - **Part I**：项目整体框架结构总结（定位、功能、架构、工作流程、关键技术）
> - **Part II**：输入数据详细说明（特征矩阵、k-mer、覆盖度、标记基因约束、模型输入）
> - **Part III**：模型训练详解（调度入口、自监督训练循环、损失函数、网络架构、半监督旧模式）
> - **Part IV**：嵌入与聚类详解（k-NN 图构建、Infomap、重聚类、长读长 DBSCAN）

---

# Part I · 项目整体框架结构总结

## 一、项目定位

**SemiBin** 是一款基于**深度学习**的**宏基因组分箱（Metagenomic Binning）**命令行工具。它的核心任务是将宏基因组组装得到的 **contigs（重叠群）** 聚类成不同的 **基因组箱（bins / MAGs，宏基因组组装基因组）**。

- 论文发表：**Nature Communications** (2022, SemiBin1) 和 **Bioinformatics** (2023, SemiBin2)
- 当前版本：**SemiBin2**（v2.5.0+），仅保留 `SemiBin2` 命令
- 开发语言：**Python 3.11+**
- 核心深度学习框架：**PyTorch**
- 许可证：MIT
- 上游仓库：[BigDataBiology/SemiBin](https://github.com/BigDataBiology/SemiBin)

---

## 二、核心功能

| 功能模块 | 说明 |
|---------|------|
| **自监督对比学习** | SemiBin2 默认模式，无需外部参考数据库即可训练模型 |
| **半监督学习** | 旧版模式，利用 GTDB 分类信息生成 cannot-link 约束（已不推荐） |
| **单样本分箱** | 每个样本独立组装、独立分箱 |
| **共组装分箱** | 多样本合并组装后统一分箱 |
| **多样本分箱** | 各样本独立组装，但利用跨样本丰度信息联合分箱 |
| **长读长支持** | 针对三代测序（PacBio/ONT）的 ensemble DBSCAN 聚类算法 |
| **预训练模型** | 内置 12 种生态环境的预训练模型，开箱即用 |
| **GPU 加速** | 自动检测 CUDA，支持 GPU 训练 |

12 个预训练环境：`human_gut` · `dog_gut` · `ocean` · `soil` · `cat_gut` · `human_oral` · `mouse_gut` · `pig_gut` · `built_environment` · `wastewater` · `chicken_caecum` · `global`

> ⚠️ 预训练模型**只能用于单样本、单 BAM** 的场景；共组装/多样本必须自监督训练。

---

## 三、代码架构

### 3.1 目录结构

```
SemiBin/
├── SemiBin/                      # 主 Python 包
│   ├── main.py                   # CLI 入口 & 子命令调度（~1700 行）
│   ├── semi_supervised_model.py  # 孪生网络架构 & 半监督训练
│   ├── self_supervised_model.py  # 自监督对比学习训练
│   ├── cluster.py                # 短读长聚类（Infomap 图聚类）
│   ├── long_read_cluster.py      # 长读长聚类（ensemble DBSCAN）
│   ├── generate_kmer.py          # K-mer 序列特征提取
│   ├── generate_coverage.py      # BAM → 覆盖度/丰度特征
│   ├── markers.py                # 标记基因识别（HMMER）
│   ├── orffinding.py             # ORF 预测（prodigal/fraggenescan/内置）
│   ├── naive_orffinder.py        # 内置 fast-naive ORF finder（默认）
│   ├── gtdb.py                   # GTDB 参考数据库集成（已弃用）
│   ├── fasta.py                  # FASTA 文件解析（支持 .gz/.bz2/.xz）
│   ├── atomicwrite.py            # 原子写文件工具
│   ├── utils.py                  # 工具函数（约束生成、模型加载等）
│   ├── semibin_version.py        # 版本号
│   ├── citation.py               # 引用格式（BIBTEX / RIS / CHICAGO）
│   ├── marker.hmm                # 标记基因 HMM 模型
│   ├── models/                   # 12 个预训练 PyTorch 模型 (.pt)
│   └── skills/semibin/           # 给 AI 助手用的"使用说明书"（Claude Code Skills）
├── docs/                         # MkDocs 文档
├── test/                         # 单元测试
├── integration-tests/            # 端到端集成测试
├── script/                       # 辅助脚本
├── pyproject.toml                # 项目配置 & 依赖
├── pixi.lock                     # pixi 环境锁文件
├── mkdocs.yml                    # 文档站点配置
├── AGENTS.md                     # 给 AI 编码助手看的项目说明
└── roadmap.md                    # 开发路线图
```

### 3.2 核心类与模型

| 文件 | 核心类/函数 | 作用 |
|------|------------|------|
| `semi_supervised_model.py` | `Semi_encoding_single`, `Semi_encoding_multiple` | **孪生神经网络**（Siamese Network），编码器-解码器结构，将 contig 特征映射到 100 维嵌入空间 |
| `self_supervised_model.py` | `train_self()` | **对比学习损失**（Contrastive Loss），利用 must-link/cannot-link 对训练 |
| `cluster.py` | `run_embed_infomap()` | 使用 **igraph + Infomap** 进行图聚类（短读长） |
| `long_read_cluster.py` | `cluster_long_read()` | 使用 **DBSCAN ensemble** 进行长读长聚类 |

**网络结构**：`Linear → BatchNorm → LeakyReLU → Dropout × 3 层 → 100 维嵌入`，输出统一为 `[batch_size, 100]`。

---

## 四、工作流程

SemiBin 的标准流程分为 **4 个阶段**：

```
输入: contigs.fa + sorted.bam (或 abundance.txt)
         │
         ▼
┌─────────────────────┐
│  1. 特征生成         │  generate_sequence_features_*
│     - K-mer 组成     │  → data.csv, data_split.csv
│     - 覆盖度/丰度    │
│     - 标记基因       │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  2. 模型训练         │  train_self (默认) / train_semi (弃用)
│     - 自监督对比学习 │  → model.pt
│     - must/cannot    │
│       -link 约束     │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  3. 嵌入 + 聚类      │  bin_short / bin_long
│     - 100 维嵌入     │  → contig_bins.tsv
│     - k-NN 图构建    │
│     - Infomap/DBSCAN │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  4. 重聚类 (可选)    │  recluster (短读长默认开启)
│     - 基于标记基因   │  → output_bins/
│       细化分箱       │
└─────────────────────┘
         │
         ▼
输出: output/output_bins/*.fa.gz  (最终的 MAG bins)
```

### 4.1 主要子命令

| 类别 | 命令 | 说明 |
|------|------|------|
| **一键命令** | `single_easy_bin` | 单样本 / 共组装分箱（最常用） |
| | `multi_easy_bin` | 多样本分箱 |
| **预处理** | `concatenate_fasta` | 多样本 FASTA 合并（加样本前缀） |
| | `split_contigs` | 为 strobealign-aemb 流程拆分 contigs |
| **分步执行** | `generate_sequence_features_single/multi` | 单独跑特征生成 |
| | `train_self` | 单独跑自监督训练 |
| | `bin_short` / `bin_long` | 单独跑聚类 |
| **工具** | `check_install` | 检查依赖是否安装完整 |
| | `citation` | 打印引用格式 |
| | `update_model` | 升级旧模型文件到新格式 |
| **已弃用** | `train_semi`, `predict_taxonomy`, `download_GTDB`, `generate_cannot_links` | 半监督学习相关 |

---

## 五、关键技术

| 技术点 | 实现 |
|--------|------|
| **深度学习框架** | PyTorch（自动检测 CUDA） |
| **对比学习** | NT-Xent Loss（自监督） |
| **图聚类** | igraph + Infomap（短读长） |
| **密度聚类** | scikit-learn DBSCAN ensemble（长读长） |
| **K-mer 计算** | 内置 Rust/NumPy 加速，4-mer canonical 频率 |
| **覆盖度计算** | bedtools genomecov 解析 |
| **ORF 预测** | fast-naive（默认，无外部依赖）/ prodigal / fraggenescan |
| **标记基因** | HMMER + 内置 `marker.hmm` |
| **重聚类** | 基于单拷贝标记基因的二级聚类 |

---

## 六、AI 辅助文档

SemiBin 仓库为 AI 助手提供了两层文档：

### 6.1 `AGENTS.md`（项目根目录）
给所有 AI 编码助手看的项目说明：开发命令、测试运行方式、目录约定。

### 6.2 `SemiBin/skills/semibin/`（Claude Code Skills 包）

这是**给 AI 助手用的"使用说明书"**，**不是给用户用的，也不影响 SemiBin2 程序功能**。来自上游官方仓库，10 月 8 日刚同步。

```
SemiBin/skills/semibin/
├── SKILL.md                  # 主入口：决策树 + 命令模板 + 排错指南
└── references/
    ├── subcommands.md        # 所有 SemiBin2 子命令的参数表
    └── developer-guide.md    # 如何在代码里新增子命令
```

**SKILL.md 的 front-matter**：
```yaml
name: semibin
description: Auto-activate when the user mentions SemiBin, SemiBin2, metagenomic binning, MAGs…
```

**作用机制**：当 Claude Code 检测到用户提到 SemiBin2 / 宏基因组分箱 / MetaBAT2/MaxBin/CONCOCT 对比时，会自动加载这个文件，让 AI **第一次回答就给出正确的命令行**，避免常见误用（例如把 `--environment` 和多 BAM 一起用、忘记 BAM 要先 sort+index、长读长忘记加 `--sequencing-type long_read`）。

**包含内容**：
- 选 `single_easy_bin` vs `multi_easy_bin` 的决策树
- 复制即用的命令模板（单样本 / 共组装 / 多样本 / 长读长）
- 输入要求（FASTA 压缩格式、BAM 必须 sorted + indexed）
- 输出文件位置（`output/output_bins/` 是默认最终目录）
- 12 个预训练环境列表
- 10 条常见报错 + 修复方法
- Sanity check 清单
- 下游 CheckM2 / GTDB-Tk 质量评估建议

> 删除 `skills/` 目录不会影响 SemiBin2 的任何功能。

---

## 七、外部依赖

| 依赖 | 用途 | 必需性 |
|------|------|--------|
| **PyTorch** | 神经网络训练 | 必需 |
| **bedtools** | 覆盖度计算 | 必需（用 BAM 时） |
| **samtools** | BAM 处理（用户自行 sort+index） | 必需 |
| **HMMER** | 标记基因识别（重聚类用） | 短读长默认需要 |
| **prodigal / fraggenescan** | ORF 预测 | 可选（默认 fast-naive 不需要） |
| **mmseqs2 + GTDB** | 半监督学习的 cannot-link | 已弃用，不需要 |
| **strobealign-aemb** | 丰度估计（替代 BAM） | 可选（≥5 样本时） |

---

## 八、安装与运行

```bash
# 推荐：用 pixi 全局安装
pixi global install semibin

# 或用 conda
conda install -c conda-forge -c bioconda semibin

# 验证安装
SemiBin2 check_install --verbose

# 最简单的运行
SemiBin2 single_easy_bin \
    --environment human_gut \
    -i contigs.fa \
    -b sample.sorted.bam \
    -o output
```

---

# Part II · 输入数据详细说明

## 一、模型输入的数据构成

SemiBin 的神经网络的输入是 **`data.csv`**（训练用特征矩阵），由两类特征拼接而成：

| 特征类型 | 列数 | 来源 |
|---------|------|------|
| **k-mer 组成** | 固定 **136 列** | contig 序列的 4-mer 频率（canonical k-mer，正反向互补合并） |
| **覆盖度 / 丰度** | **N × 1 或 N × 2 列** | BAM 文件经 bedtools 计算，或 strobealign-aemb 丰度表 |

总列数 = **136 + coverage 列数**，行数 = 过滤后的 contig 数量。

### 1.1 模型实际吃进去的输入

真正进神经网络的就是**一个 contig 的特征向量**，batch 后是 `[batch_size, 特征维度]`。只有两种情况（每次运行只会是其中一种，由 BAM 数量决定）：

```python
# 情况1：BAM < 5（单样本）— 只吃 k-mer，coverage 被切掉
输入向量 = [kmer_0, kmer_1, ..., kmer_135]                        # 136 维
Tensor:  [batch_size, 136]

# 情况2：BAM >= 5（共组装/多样本）— k-mer + coverage
输入向量 = [kmer_0, ..., kmer_135, cov_S1, cov_S2, ..., cov_SN]   # 136+N 维
Tensor:  [batch_size, 136+N]
```

- 默认 `batch_size=2048`，输出统一为 `[batch_size, 100]` 的嵌入向量，之后用嵌入构建 k-NN 图做聚类
- 注意：`data.csv` 文件本身总是带 coverage 列的（单样本模式是 138 列），但单样本模式训练时只取前 136 列（`train_data[:, :136]`）；`data_split.csv` 不直接进模型，仅用于构造训练对

---

## 二、k-mer 特征（前 136 列）

- 计算代码：[`SemiBin/SemiBin/generate_kmer.py`](SemiBin/SemiBin/generate_kmer.py)
- 4-mer 一共 4⁴ = 256 种，但**正反向互补视为相同**（如 `AAAA` = `TTTT`），去重后剩 **136 种** → 136 列，列名是整数 0~135
- 每个 contig 一行，值为频率（先加 1e-5 伪计数，再按行归一化，**每行和为 1**）
- 只保留长度 ≥ 1000bp 的 contig
- 劈分版本（`data_split.csv`）：长 contig 从中点切成 `{contig}_1` / `{contig}_2` 两段，分别计算 k-mer

```csv
,0,1,2,...,135
k141_63080,0.0089,0.0001,0.0074,...,0.0052
k141_0,0.0055,0.0031,0.0091,...,0.0011
```

---

## 三、覆盖度特征（后 N 列）

计算代码：**[`SemiBin/SemiBin/generate_coverage.py`](SemiBin/SemiBin/generate_coverage.py)**，4 个主要函数：

| 函数 | 作用 |
|------|------|
| `calculate_coverage()` | 核心计算：解析 `bedtools genomecov` 输出，按 contig 计算平均覆盖度和方差 |
| `generate_cov()` | 调用外部命令 `bedtools genomecov -bga -ibam <file>`，输出单个 BAM 的 CSV |
| `combine_cov()` | 把多个 BAM 的 CSV 横向合并成一个大表 |
| `generate_cov_from_abundances()` | 从 strobealign-aemb 丰度表读取（替代 BAM，要求 ≥5 个文件） |

### 3.1 计算细节（`calculate_coverage()`）

1. `bedtools genomecov -bga` 流式输出（每行：contig 名、起始、终止、深度值）
2. 按 contig 分组，重建逐碱基深度数组
3. **去掉首尾各 75bp**（`edge=75`），减少边缘效应
4. 计算 **mean（平均覆盖度）** 和 **var（方差）**
5. 加 **1e-5 伪计数**避免除零；`is_combined` 模式按 `abun_scale = (mean/100).ceil()*100` 归一化

### 3.2 两种模式的列数差异（关键！）

列数由 `is_combined` 决定，判定条件在 [`main.py:829`](SemiBin/SemiBin/main.py#L829)：**`is_combined = len(bams) >= 5`**

| | `is_combined=False`（BAM 数 < 5） | `is_combined=True`（BAM 数 ≥ 5） |
|---|---|---|
| **每个 BAM 的列数** | **2 列**（`_mean` + `_var`） | **1 列**（`_cov`，归一化均值，丢弃方差） |
| **coverage 总列数** | **2N** | **N** |
| **data.csv 总列数** | 136 + 2N | 136 + N |
| **coverage 是否进神经网络** | ❌ 训练时切掉（`train_data[:, :136]`），只用 k-mer | ✅ 和 k-mer 拼接后一起进模型 |
| **是否劈分 contig 算 split coverage** | ❌ | ✅（`_1`/`_2` 两段，用于 must-link 约束） |

> 即：一个"其他样本"（一个 BAM）对应一组 coverage —— 样本少时每组 2 列（均值+方差），样本多时（≥5）每组 1 列（归一化均值）。

### 3.3 中间文件与最终文件

**每个 BAM 一份中间 CSV**（命名规则：`{bam文件名}_{序号}_data_cov.csv`，如 `S1.sorted.bam_0_data_cov.csv`）：

单样本模式（is_combined=False），两列：
```csv
contig,S1.sorted.bam_mean,S1.sorted.bam_var
k141_63080,2.581828,0.606942
k141_0,2.627917,0.252244
```

共组装/多样本模式（is_combined=True），一列：
```csv
contig,S1.sorted.bam_cov
k141_63080,0.025818
k141_0,0.026279
```

劈分覆盖度（仅 is_combined 生成 `*_data_split_cov.csv`）：
```csv
contig,S1.sorted.bam_cov
k141_63080_1,0.026098
k141_63080_2,0.025542
```

**丰度表路径**的输出：`cov_split.csv`（劈分丰度）和 `cov.csv`（`_1`/`_2` 取平均合并回完整 contig），每个丰度文件一列：
```csv
contig,S1_abundance.txt,S2_abundance.txt,S3_abundance.txt,...
k141_63080,0.0312,0.0187,0.0255,...
```

> 注：以上数值取自 [`test/test_cov.py`](SemiBin/test/test_cov.py) 的真实测试用例。

---

## 四、标记基因（marker genes）— 不进特征矩阵，但决定训练约束

标记基因**不是 `data.csv` 里的列**，而是**生成 must-link / cannot-link 约束**的依据。这些约束决定了自监督训练时哪些 contig 对应该被拉近（must-link）、哪些应该被推远（cannot-link），所以它是模型的**第三类输入信号**。

### 4.1 标记基因从哪来

打包在仓库里的一个文件：[`SemiBin/SemiBin/marker.hmm`](SemiBin/SemiBin/marker.hmm)（约 16 MB）。

| 属性 | 值 |
|------|---|
| **格式** | HMMER3 profile-HMM 库 |
| **总数** | **111 个单拷贝标记基因** |
| **来源** | **TIGRFAM 96 个** + **Pfam 15 个** |
| **构建时间** | 2012 年 10 月（与 CheckM 的 ~104 个细菌单拷贝基因集是同时期、同性质的东西） |
| **是否需要外部数据库** | ❌ 不需要，也不需要联网 |

典型条目：`PGK (PF00162.14)`、`Ribosomal_L23 (PF00276.15)`、`TIGR00001`、`TIGR00002` …

> 想换一套标记基因？代码里**没有自定义参数**，`marker.hmm` 路径写死在 [`markers.py:165`](SemiBin/SemiBin/markers.py#L165)，需要改代码。

### 4.2 识别流程

代码：[`SemiBin/SemiBin/markers.py`](SemiBin/SemiBin/markers.py) 的 `estimate_seeds()` 函数。

```
contigs.fa
    │
    ▼  1. ORF 预测（orffinding.py，默认 fast-naive）
蛋白序列 (.faa)
    │
    ▼  2. hmmsearch --cut_tc 用 marker.hmm 扫
markers.hmmout
    │
    ▼  3. 解析 + 过滤 (get_marker())
{contig: [marker_gene, ...]}
```

关键过滤条件（[`markers.py:58`](SemiBin/SemiBin/markers.py#L58)）：
- ORF 上匹配区域的覆盖率 **> 40%**：`(qend - qstart) / qlen > 0.4`
- contig 长度 ≥ `min_contig_len`
- 同一 contig 同一基因去重
- TIGRFAM ID 归一化（[`markers.py:28-33`](SemiBin/SemiBin/markers.py#L28-L33)）：`TIGR00388→TIGR00389`、`TIGR00471→TIGR00472`、`TIGR00408→TIGR00409`、`TIGR02386→TIGR02387`

**缓存机制**：结果缓存在 `output/markers.hmmout` + `markers.hmmout.json`（指纹 = fasta 大小 + mtime + `binned_length` + `orf_finder`），同一输入不会重复跑 hmmsearch。

### 4.3 标记基因怎么变成训练约束

| 约束类型 | 生成规则 | 在训练里的作用 |
|---------|---------|---------------|
| **Must-link**（必须同 bin） | 同一 contig 的 `_1` / `_2` 两个 split 段 | 强制模型把它们嵌入到相近位置 |
| **Cannot-link**（必须不同 bin） | 同一 contig 上**同时出现两个不同的标记基因** → 该 contig 不可能和"含同种标记基因的其他 contig"同 bin | 在嵌入空间里推开这些 contig 对 |

> Cannot-link 的直觉：**单拷贝标记基因**在同一个基因组里只该出现一次。如果两个 contig 各自带同一个标记基因，它们就不可能来自同一个基因组。

### 4.4 在三个阶段被用到

| 阶段 | 用到的标记基因信息 | 代码位置 |
|------|-------------------|---------|
| **自监督训练** | must-link / cannot-link 对 | [`self_supervised_model.py`](SemiBin/SemiBin/self_supervised_model.py) |
| **估计 bin 数 / 选 seed contig** | 携带标记基因多的 contig 作为聚类种子 | `estimate_seeds()` |
| **重聚类 (recluster)** | 看每个 bin 内标记基因的**单拷贝完整性**，决定是否拆开混合 bin | [`cluster.py`](SemiBin/SemiBin/cluster.py)（短读长默认开启） |

### 4.5 一句话总结

> **标记基因不是 `data.csv` 的特征列，而是训练前计算好的"先验约束"**——告诉对比学习哪些 contig 该靠近、哪些该远离。它来自打包在仓库里的 111 个 TIGRFAM+Pfam HMM profile，用 `hmmsearch` 识别，结果缓存为 `markers.hmmout`。

---

## 五、最终的模型输入矩阵

k-mer 与 coverage 按 contig 名 `pd.merge`（inner join）得到 `data.csv`：

```csv
,0,1,2,...,135,S1.sorted.bam_cov,S2.sorted.bam_cov
k141_63080,0.0089,0.0001,...,0.0052,0.025818,0.018732
k141_0,0.0055,0.0031,...,0.0011,0.026279,0.018945
```

- **行索引**：contig 名
- **前 136 列**：k-mer 频率（行和为 1）
- **后 N 列**：coverage（单样本模式为 2N 列 mean/var）

`data_split.csv` 结构相同，行是劈分后的 `{contig}_1` / `{contig}_2`。

### 模型的"三类输入"总览

| 输入 | 形态 | 进神经网络？ | 作用 |
|------|------|--------------|------|
| **k-mer 频率** | `data.csv` 前 136 列 | ✅ 直接进 | 序列组成特征 |
| **覆盖度/丰度** | `data.csv` 后 N 列（≥5 样本时） | ✅ 仅在 `is_combined=True` 时进 | 跨样本丰度共变性 |
| **标记基因** | `markers.hmmout` → must/cannot-link 对 | ❌ 不进网络，作为**训练约束** | 告诉对比学习哪些 contig 该拉近/推远 |

---

## 六、两种模型架构分别吃什么数据

代码：[`SemiBin/SemiBin/semi_supervised_model.py`](SemiBin/SemiBin/semi_supervised_model.py)

| 模型类 | 输入 | 输入维度 | 使用场景 |
|--------|------|---------|---------|
| `Semi_encoding_single` | **纯 k-mer**（`train_data[:, :136]`，coverage 列被切掉） | **136** | is_combined=False（BAM < 5） |
| `Semi_encoding_multiple` | **k-mer + coverage 全表** | **136 + N** | is_combined=True（BAM ≥ 5） |

两者网络结构相同：`Linear → BatchNorm → LeakyReLU → Dropout × 3 层 → 100 维嵌入`，仅输入维度和 decoder 末端不同（`Softmax` vs `Sigmoid`）。

**12 个预训练模型（`SemiBin/models/*.pt`）全部是 `Semi_encoding_single`**，只吃 136 维 k-mer。代码强制校验（[`main.py:1186-1189`](SemiBin/SemiBin/main.py#L1186-L1189)）：

```python
if environment is not None:
    if data.shape[1] != 138:   # 136 k-mer + 单 BAM 的 mean/var 两列
        logger.error("The provided pretrained model can only be used for single-sample binning!")
```

所以预训练模型**只能用于单样本、单 BAM** 的场景；共组装/多样本（列数随 N 变化）必须自己训练模型。

### 训练时的数据预处理（is_combined=True）

[`self_supervised_model.py:62-69`](SemiBin/SemiBin/self_supervised_model.py#L62-L69)：coverage 列数 ≥ 5 且总和够大时（`norm_abundance()`），对整表做**列归一化 + L1 行归一化**，再喂入模型。

---

# Part III · 模型训练详解

> 对应工作流程的**第 2 阶段**：`data.csv` + `data_split.csv` → `model.pt`。
> 核心代码集中在 **3 个文件**：[`main.py`](SemiBin/SemiBin/main.py)（调度）、[`self_supervised_model.py`](SemiBin/SemiBin/self_supervised_model.py)（默认训练循环）、[`semi_supervised_model.py`](SemiBin/SemiBin/semi_supervised_model.py)（网络架构 + 已弃用的半监督训练）。

## 一、调度入口：main.py

| 位置 | 代码 | 作用 |
|------|------|------|
| [`main.py:1651-1660`](SemiBin/SemiBin/main.py#L1651-L1660) | `elif args.cmd == 'train_self':` | `SemiBin2 train_self` 子命令的分支 |
| [`main.py:1100-1157`](SemiBin/SemiBin/main.py#L1100-L1157) | `def training(...)` | 训练的**总入口函数**：判断 `is_combined`、按 `args.training_type` 分派、保存模型 |
| [`main.py:1675-1698`](SemiBin/SemiBin/main.py#L1675-L1698) | `single_easy_bin` / `multi_easy_bin` | 一键命令内部也走同一个 `training()` 流程 |

### 1.1 `training()` 函数的关键逻辑

```python
# main.py:1113-1121  判断单样本还是多样本
if mode == 'single':
    data_ = pd.read_csv(data[0], index_col=0)
    col_name = data_.columns[-1].split('_')[-1]
    is_combined = col_name != 'var'      # 末列是 _var（mean/var 双列）→ 单样本
else:
    is_combined = False

# main.py:1123-1156  按训练类型分派
if args.training_type == 'semi':
    model = train_semi(...)              # 半监督（已弃用，需 mmseqs2 + GTDB）
else:
    model = train_self(...)              # 自监督（默认）
model.save_with_params_to(os.path.join(output, 'model.pt'))
```

- `is_combined` 不直接数 BAM，而是看 `data.csv` **最后一列名**：以 `_var` 结尾（BAM < 5 时的 mean/var 双列模式）→ `False`，否则 → `True`
- 训练产物统一保存为 **`output/model.pt`**

---

## 二、自监督训练（默认路径）：self_supervised_model.py

代码：[`SemiBin/SemiBin/self_supervised_model.py`](SemiBin/SemiBin/self_supervised_model.py)（约 137 行），核心函数 **`train_self()`**。

### 2.1 对比损失函数（Contrastive Loss）

[`self_supervised_model.py:8-16`](SemiBin/SemiBin/self_supervised_model.py#L8-L16)：

```python
def loss_function(embedding1, embedding2, label):
    relu = torch.nn.ReLU()
    d = torch.norm(embedding1 - embedding2, p=2, dim=1)   # 嵌入对的欧氏距离
    square_pred = torch.square(d)                         # must-link：d²
    margin_square = torch.square(relu(1 - d))             # cannot-link：relu(1-d)²
    supervised_loss = torch.mean(
        label * square_pred + (1 - label) * margin_square)
    return supervised_loss
```

| 约束 | label | 损失项 | 效果 |
|------|-------|--------|------|
| **Must-link** | 1 | `d²` | 距离越小损失越小 → **拉近** |
| **Cannot-link** | 0 | `relu(1-d)²` | 距离 < 1（margin）时有惩罚 → **推远到 ≥1** |

### 2.2 训练对的构造：`train_input_1` / `train_input_2` / `train_labels`

每个 epoch、每个样本，都会重新拼出三个等长数组（[`self_supervised_model.py:96-118`](SemiBin/SemiBin/self_supervised_model.py#L96-L118)）。`train_input_1[i]` 与 `train_input_2[i]` 是**一对** contig 特征向量，`train_labels[i]` 标记这一对是 must-link（1）还是 cannot-link（0）。

#### (a) Cannot-link 部分（前半段，label=0）—— 每 epoch 随机重采样

```python
# self_supervised_model.py:99-102
indices1 = np.random.choice(data_length, size=n_cannot_link)          # 随机抽 contig
indices2 = indices1 + 1 + np.random.choice(data_length - 1,
                                           size=n_cannot_link)
indices2 %= data_length                                               # 环形错位，保证 ≠ indices1
```

- `data_length` = 完整 contig 数（`data.csv` 行数）
- `indices1`：均匀随机抽 `n_cannot_link` 个 contig 下标（**可重复**）
- `indices2`：在 `indices1` 基础上加一个 **1 ~ data_length-1 的随机偏移**再取模——效果等价于"随机抽另一个不同的 contig"，但纯向量化实现
- **保证 `indices2[i] ≠ indices1[i]`**：偏移量 ∈ [1, data_length-1]，永远不会是 0，所以不会把自己配成 cannot-link
- 注意：这里用的是 `train_data`（**完整 contig** 的特征），不是劈分段

#### (b) Must-link 部分（后半段，label=1）—— 劈分段两两配对

```python
# self_supervised_model.py:106, 109
train_data_split[::2]    # 偶数行：所有 contig 的 _1 段
train_data_split[1::2]   # 奇数行：所有 contig 的 _2 段
```

`data_split.csv` 的行是**交错排列**的（`c0_1, c0_2, c1_1, c1_2, …`），所以偶数行和奇数行错位一位后天然就是「同一 contig 的两半」一一对应。这部分**每个 epoch 都相同**，不重新采样。

#### (c) 拼接与打标

```python
# self_supervised_model.py:104-111
train_input_1 = np.concatenate((train_data[indices1],      # ← cannot-link 端点1（完整 contig）
                                train_data_split[::2]))    # ← must-link 端点1（_1 段）
train_input_2 = np.concatenate((train_data[indices2],      # ← cannot-link 端点2（完整 contig）
                                train_data_split[1::2]))   # ← must-link 端点2（_2 段）
train_labels = np.zeros(len(train_input_1), dtype=np.float32)
train_labels[len(indices1):] = 1                           # 前半 0（cannot），后半 1（must）
```

最终每个 epoch 的训练集结构：

```
下标        [0, n_cannot_link)                [n_cannot_link, n_cannot_link + n_must_link/2)
input_1     train_data[indices1]  完整 contig  data_split[::2]   劈分 _1 段
input_2     train_data[indices2]  完整 contig  data_split[1::2]  劈分 _2 段
label       0 (cannot-link)                   1 (must-link)
```

| 类型 | 来源数据 | 配对方式 | 数量 |
|------|---------|---------|------|
| **Cannot-link**（label=0） | `train_data`（完整 contig） | 随机错位配对，**每 epoch 重采样** | `n_cannot_link = min(n_must_link×1000 // 2, 4,000,000)`（上限 400 万） |
| **Must-link**（label=1） | `train_data_split`（劈分段） | 同一 contig 的 `_1`/`_2` 固定配对 | `n_must_link / 2`（`data_split.csv` 行数的一半） |

#### (d) 数量级直观感受

`n_cannot_link` 被设计成 must-link 的 **~500 倍**（`n_must_link×1000//2`，再除以 2 是因为 must-link 数本身按行数算、每 2 行才是一对）。举例：1 万个劈分 contig → must-link 1 万对，cannot-link 上限内取 500 万 → 截到 400 万。也就是说**负对占绝对多数**——对比学习靠海量随机负对把嵌入空间"撑开"，少量正对只负责把同一 contig 的两半锚在一起。

#### (e) 送进模型

```python
# self_supervised_model.py:112-131
dataset = feature_Dataset(train_input_1, train_input_2, train_labels)
train_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=True)

for train_input1, train_input2, train_label in train_loader:
    embedding1, embedding2 = model.forward(train_input1, train_input2)  # 孪生编码器各跑一遍
    supervised_loss = loss_function(embedding1, embedding2, train_label)
    ...
```

- `shuffle=True` 会把 cannot/must 对**打乱混合**进每个 batch（拼接时的前后分段只在拼接瞬间有意义）
- `model.forward(input1, input2)` 用**同一个编码器**分别编码两个输入 → 两个 100 维嵌入 → 交给 2.1 的对比损失

> ⚠️ **与 SemiBin1 的最大区别**：自监督模式**不用标记基因生成 cannot-link**，完全靠随机负对 + 劈分正对；标记基因只在重聚类阶段用。每个 epoch 重采样负对，相当于在线数据增强，避免模型记住固定的负样本。

### 2.3 训练超参数与流程

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)  # 每 epoch lr ×0.9
# DataLoader: batch_size=2048, shuffle=True, drop_last=True
# 默认 epochs=15, batchsize=2048（均可通过 --epochs / --batch-size 调整）
```

- 数据在训练前**一次性预计算**到内存（`train_data` / `train_data_split`），每个 epoch 只重采样 cannot-link 索引
- `is_combined=False`：`train_data = train_data[:, :136]`（只用 k-mer）
- `is_combined=True`：`norm_abundance()` → 列归一化 + sklearn L1 行归一化后，k-mer + coverage 全表进模型

---

## 三、网络架构定义：semi_supervised_model.py

> ⚠️ 文件名带 "semi-supervised" 是历史遗留——**两个模型类（网络结构）定义在这里，自监督训练也复用它**。

### 3.1 孪生网络（Siamese Network）

[`semi_supervised_model.py:12-105`](SemiBin/SemiBin/semi_supervised_model.py#L12-L105)：

```
编码器 encoder1（两个输入共享同一份权重！）:
  Linear(num → 512) → BatchNorm1d → LeakyReLU → Dropout(0.2)
  Linear(512 → 512) → BatchNorm1d → LeakyReLU → Dropout(0.2)
  Linear(512 → 100)                                # 100 维嵌入

解码器 decoder1（仅半监督模式用于重构损失）:
  Linear(100 → 512) → BatchNorm1d → LeakyReLU → Dropout(0.2)
  Linear(512 → 512) → BatchNorm1d → LeakyReLU → Dropout(0.2)
  Linear(512 → num) → Softmax(single) / Sigmoid(multiple)
```

```python
def forward(self, input1, input2):
    return self.encoder1(input1), self.encoder1(input2)   # 同一编码器跑两次 → 孪生
```

| 模型类 | 输入维度 num | decoder 末端 | 使用场景 |
|--------|-------------|-------------|---------|
| `Semi_encoding_single` | **136**（纯 k-mer） | Softmax | `is_combined=False`（BAM < 5） |
| `Semi_encoding_multiple` | **136 + N** | Sigmoid | `is_combined=True`（BAM ≥ 5） |

### 3.2 模型保存与加载

- **保存**：`save_with_params_to(path)` → `torch.save({'model_name', 'model_state_dict', 'params'})`，`params[0]` 记录输入维度，加载时据此重建对应架构
- **加载**：`model_load(path, device)`（[`semi_supervised_model.py:108-145`](SemiBin/SemiBin/semi_supervised_model.py#L108-L145)）——先用 `weights_only=True` 安全加载；失败则回退到旧版 pickle 方式（兼容早期 `.h5` 模型，并提示用 `SemiBin2 update_model` 转换）

### 3.3 半监督版损失（对比）

[`semi_supervised_model.py:148-169`](SemiBin/SemiBin/semi_supervised_model.py#L148-L169)：比自监督版**多加一项自动编码器重构损失**：

```python
loss = supervised_loss + 0.5*MSE(decoder_x_1, raw_x_1) + 0.5*MSE(decoder_x_2, raw_x_2)
```

这就是 decoder 存在的意义——只在半监督模式参与训练；自监督模式只用编码器。

---

## 四、半监督训练（已弃用）：train_semi()

[`semi_supervised_model.py:200-375`](SemiBin/SemiBin/semi_supervised_model.py#L200-L375)，SemiBin1 的旧模式，`SemiBin2 train_semi` 子命令仍保留但官方不推荐。

与 `train_self()` 的关键差异：

| 方面 | `train_self()`（默认） | `train_semi()`（弃用） |
|------|----------------------|----------------------|
| **Cannot-link 来源** | 每 epoch 随机采样 | ① mmseqs2 + GTDB 注释生成的 cannot-link CSV<br>② 每 epoch 调 `estimate_seeds()` 跑 hmmsearch，用携带标记基因的 **seed contig 两两配对** |
| **额外损失** | 无 | 自动编码器重构损失（0.5·MSE ×2） |
| **额外训练回路** | 无 | 有一路**无标签自编码器训练**（全部 contig 各自与自身配对，只训重构） |
| **外部依赖** | 无 | mmseqs2 + GTDB 参考数据库 |
| **标记基因参与训练？** | ❌ 否 | ✅ 是（seed cannot-link） |

---

## 五、训练在整体流程中的位置

```
generate_sequence_features_*  →  data.csv + data_split.csv
            │
            ▼
      training()              ← main.py:1100（本步）
       ├─ train_self()        ← self_supervised_model.py（默认）
       └─ train_semi()        ← semi_supervised_model.py（弃用）
            │
            ▼
        model.pt  ──→  bin_short / bin_long：model_load() 加载 → embedding() 生成 100 维嵌入 → 聚类
```

**一句话总结**：训练逻辑 = [`self_supervised_model.py`](SemiBin/SemiBin/self_supervised_model.py) 的 `train_self()`（当前默认），网络结构在 [`semi_supervised_model.py`](SemiBin/SemiBin/semi_supervised_model.py)（`Semi_encoding_single/multiple` 孪生网络），调度在 [`main.py`](SemiBin/SemiBin/main.py) 的 `training()` 函数（[`main.py:1100`](SemiBin/SemiBin/main.py#L1100)）。

---

# Part IV · 嵌入与聚类详解

> 对应工作流程的**第 3 阶段**：`model.pt` + `data.csv` → `contig_bins.tsv` + `output_bins/`。
> 核心代码：**短读长**在 [`cluster.py`](SemiBin/SemiBin/cluster.py)（Infomap 图聚类），**长读长**在 [`long_read_cluster.py`](SemiBin/SemiBin/long_read_cluster.py)（ensemble DBSCAN），由 `main.py` 的 `binning_short()` / `binning_long()` 调度。

## 一、调度入口

| 位置 | 函数 | 作用 |
|------|------|------|
| [`main.py:1213-1238`](SemiBin/SemiBin/main.py#L1213-L1238) | `binning_short()` | 短读长：`binning_preprocess()` → `cluster()` |
| [`main.py:1195-1211`](SemiBin/SemiBin/main.py#L1195-L1211) | `binning_long()` | 长读长：`binning_preprocess()` → `cluster_long_read()` |
| [`main.py:1160`](SemiBin/SemiBin/main.py#L1160) | `binning_preprocess()` | 读 `data.csv`、判断 `is_combined`、`model_load()` 加载 `model.pt` 或预训练模型 |

---

## 二、短读长：cluster.py

### 2.1 嵌入生成 + 双 k-NN 图交集（`run_embed_infomap()`）

[`cluster.py:64-168`](SemiBin/SemiBin/cluster.py#L64-L168)，SemiBin 聚类的核心设计：

```python
# cluster.py:90-113
# ① 模型推理：contig 特征 → 100 维嵌入
with torch.no_grad():
    model.eval()
    embedding = model.embedding(x.float()).detach().cpu().numpy()

# ② 构建两张 k-NN 图（默认 max_edges=200 个邻居）
embedding_matrix = kneighbors_graph(embedding,         # 学习到的嵌入空间
                    n_neighbors=min(max_edges, n-1), mode='distance', p=2)
kmer_matrix      = kneighbors_graph(train_data_input,  # 原始特征空间（k-mer 或 k-mer+cov）
                    n_neighbors=min(max_edges, n-1), mode='distance', p=2)

# ③ 取交集：只保留"两个空间里互为近邻"的边
kmer_matrix.data.fill(1.)
embedding_matrix = embedding_matrix.multiply(kmer_matrix)
```

> **为什么要双图交集**：两个 contig 只有在**学习到的嵌入空间**和**原始 k-mer 空间**里都互为近邻才连边。嵌入可能因训练误差把不相干的 contig 拉到一起，用原始特征交叉验证可以砍掉这类假边。

### 2.2 边权处理与自适应稀疏化

```python
# cluster.py:115-128
np.clip(embedding_matrix.data, None, 1., out=embedding_matrix.data)
embedding_matrix.data = 1 - embedding_matrix.data      # 距离 → 相似度

# 自适应阈值：从 0 起每次 +0.05，
# 直到"仍有边的 contig 比例"低于 max_node（默认 1.0）为止
threshold = 0
while threshold < 1:
    threshold += 0.05
    n_above = np.sum(max_axis1 > threshold)
    if round(n_above / num_contigs, 2) < max_node:
        break
embedding_matrix.data[embedding_matrix.data <= threshold] = 0   # 砍弱边
```

### 2.3 单样本模式的 coverage KL 边权（关键补偿！）

[`cluster.py:129-141`](SemiBin/SemiBin/cluster.py#L129-L141)：

```python
if not is_combined:
    for k in range(n_sample):
        kl = cal_kl(depth[:,2*k], depth[:, 2*k + 1])   # 每个样本 (mean, var) → 高斯 KL 散度矩阵
        kl_matrix -= kl                                 # 累加
    kl_matrix += n_sample; kl_matrix /= n_sample        # KL → [0,1] 相似度
    embedding_matrix = embedding_matrix.multiply(kl_matrix)   # 边权 × 覆盖度相似度
```

- `cal_kl()`（[`cluster.py:13-60`](SemiBin/SemiBin/cluster.py#L13-L60)）把每对 contig 的 `(mean, var)` 当作高斯分布，计算对称化 KL 散度，再截断到 `[0, 1]` 作为相似度（优先用 numexpr 加速）
- **含义**：单样本时 coverage 没进神经网络（训练只用前 136 维 k-mer），所以在这里以**边权**的形式把覆盖度信息补偿回来；共组装模式 coverage 已进模型，跳过此步

### 2.4 Infomap 图聚类

```python
# cluster.py:155-163
g = Graph()
g.add_vertices(np.arange(num_contigs))
g.add_edges(edges)                       # 只取上三角的边（无向图去重）
result = g.community_infomap(
            edge_weights=edge_weights,   # 边权 = 相似度
            vertex_weights=length_weight,  # 顶点权 = contig 长度（bp）
            trials=NR_INFOMAP_TRIALS)    # 10 次取最优
```

用 **igraph 的 Infomap** 社区发现算法：边权 = 上一步算出的相似度，顶点权 = contig 长度，输出每个 contig 的社区标签 `contig_labels`。

### 2.5 重聚类：`recluster_bins()`（标记基因在此发挥作用！）

[`cluster.py:171-255`](SemiBin/SemiBin/cluster.py#L171-L255)，短读长**默认开启**（`--no-recluster` 可关闭）：

```
每个 bin
   │
   ▼ ① 把 bin 内 contig 拼成临时 fasta，跑 estimate_seeds()（hmmsearch）
找"种子 contig" = 携带 ≥2 个不同标记基因的 contig
   │
   ├─ 种子数 ≤ 1  →  保持原样（bin 内部大概率是单基因组）
   └─ 种子数 > 1  →  ② 这个 bin 混进了多个基因组，需要拆：
                       KMeans(n_clusters=种子数,
                              init=种子 contig 的嵌入,   ← 用标记基因种子做初始中心
                              n_init=1,
                              sample_weight=contig 长度)
```

**单样本模式的额外处理**（[`cluster.py:185-195`](SemiBin/SemiBin/cluster.py#L185-L195)）：拆 bin 前把**覆盖度加权拼接到嵌入上**，让 KMeans 同时考虑序列组成和丰度：

```python
embedding_new = np.concatenate((embedding, depth_mean * weight), axis=1)
# weight = 2*10*ceil(scaling/10)，scaling = mean(|embedding|) / mean(depth_mean)
# 即把 depth 缩放到与嵌入同一量级
```

### 2.6 结果写出（`cluster()`）

[`cluster.py:258-348`](SemiBin/SemiBin/cluster.py#L258-L348)：

| 模式 | 输出目录 | 汇总文件 |
|------|---------|---------|
| 默认（recluster 开启） | `output_bins/` | `recluster_bins_info.tsv` + `contig_bins.tsv` |
| `--no-recluster` | `output_bins/` | `bins_info.tsv` + `contig_bins.tsv` |
| `--write-pre-reclustering-bins` | 重聚类前：`output_prerecluster_bins/`<br>重聚类后：`output_recluster_bins/` | 两份 |

- 实际写 fasta 的是 [`utils.py`](SemiBin/SemiBin/utils.py) 的 `write_bins()`：按 bin 分组写 `.fa.gz`，**过滤掉总长 < `minfasta`（默认 200 kbp）的 bin**
- `contig_bins.tsv`：每个 contig → bin 编号的映射表

---

## 三、长读长：long_read_cluster.py

`cluster_long_read()`（[`long_read_cluster.py:51-162`](SemiBin/SemiBin/long_read_cluster.py#L51-L162)），长读长（PacBio/ONT）用 **ensemble DBSCAN** 代替 Infomap——长读长 contig 更长更少，图方法不再是最佳选择。

### 3.1 嵌入 + 标记基因扫描

```python
# long_read_cluster.py:67-99
embedding = model.embedding(x)                          # 同样的模型推理
if not is_combined:
    embedding_new = np.concatenate((embedding, np.log(np.clip(depth, 1e-6, None))), axis=1)  # 拼接 log 覆盖度
estimate_seeds(...)                                     # hmmsearch → markers.hmmout
contig2marker = get_marker(..., contig_to_marker=True)  # {contig: [marker, ...]}
```

注意：这里的标记基因不是约束，而是后面**给候选 bin 打分的依据**。

### 3.2 多 eps DBSCAN 扫描

```python
# long_read_cluster.py:108-120
dist_matrix = kneighbors_graph(embedding_new, n_neighbors=min(200, n-1), mode='distance')

dbscan_results = []
for eps_value in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]:
    _, labels = dbscan(dist_matrix, eps=eps_value, min_samples=5,
                       metric='precomputed', sample_weight=length_weight)
    dbscan_results.append(labels.tolist())
```

对 **12 个不同的 eps** 各跑一次 DBSCAN（`min_samples=5`，样本权重 = contig 长度），得到 12 组候选聚类——这就是 "ensemble" 的含义。

### 3.3 贪心提取最优 bin：`get_best_bin()`

[`long_read_cluster.py:12-49`](SemiBin/SemiBin/long_read_cluster.py#L12-L49)，用**标记基因质量**从 12 组结果里挑 bin：

```python
for max_contamination in [0.1, 0.2, 0.3, 0.4, 0.5, 1]:   # 污染容忍度逐级放宽
    for res_labels in dbscan_results:                     # 遍历 12 组 DBSCAN 结果
        for bin_contig in 每个 bin:
            recall       = len(set(marker_list)) / 107                          # 完整度：独特标记数/107
            contamination = (len(marker_list) - len(set(marker_list))) / len(marker_list)  # 污染度：重复标记比例
            if contamination <= max_contamination:
                F1 = 2*recall*(1-contamination) / (recall + (1-contamination))   # F1 打分
```

- **最宽松优先**：先尝试最严的污染阈值（≤0.1），找不到合格 bin 才放宽
- **F1 = 完整度 vs (1−污染度) 的调和平均**；同分取总长更小的 bin
- 主循环（[:125-143](SemiBin/SemiBin/long_read_cluster.py#L125-L143)）每提取一个最优 bin，就把它从候选池里删掉，再重新挑，直到提不出合格 bin 或剩余总长 < minfasta

### 3.4 输出

与短读长相同：`output_bins/` + `bins_info.tsv` + `contig_bins.tsv`。**长读长没有重聚类步骤**（`--no-recluster` / `--max-edges` / `--max-node` 对 `bin_long` 无效）。

---

## 四、两种聚类路径对比

| | 短读长（`bin_short`） | 长读长（`bin_long`） |
|---|---|---|
| **核心算法** | Infomap 图社区发现 | ensemble DBSCAN（12 组 eps） |
| **图/距离** | 双 k-NN 图交集（嵌入 ∩ k-mer） | 单 k-NN 距离矩阵 |
| **coverage 补偿** | KL 散度边权（单样本） | log(depth) 拼接进嵌入（单样本） |
| **标记基因角色** | 重聚类：seed → KMeans 拆混合 bin | 选 bin：F1 评分贪心提取 |
| **重聚类** | ✅ 默认开启 | ❌ 无 |
| **代码** | [`cluster.py`](SemiBin/SemiBin/cluster.py) | [`long_read_cluster.py`](SemiBin/SemiBin/long_read_cluster.py) |

**一句话总结**：**嵌入** = `model.embedding()` 把每个 contig 映射到 100 维 → **建图** = 嵌入空间 ∩ k-mer 空间双 k-NN 图取交集（单样本再乘 coverage KL 边权）→ **聚类** = 短读长 Infomap + 标记基因 KMeans 重聚类，长读长 ensemble DBSCAN + 标记基因 F1 贪心提取。
