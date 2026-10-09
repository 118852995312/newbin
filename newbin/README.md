# NewBin: Metagenomic Binning with Custom Deep Learning Models

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**NewBin** 是一个基于深度学习的**宏基因组分箱（metagenomic binning）**工具。代码骨架源自 **SemiBin2**，但将模型训练代码剥离为可替换的占位接口，方便用户使用自定义深度学习架构。

- 保留了 SemiBin2 完整的数据处理流水线：**k-mer 特征 + 覆盖度特征 → 训练嵌入模型 → 聚类分箱**
- 支持**短读（short-read）**和**长读（long-read）**数据
- 支持**单样本（single-sample）**、**共组装（co-assembly）**和**多样本（multi-sample）**分箱模式

---

## 安装

### 使用 pixi（推荐，一键安装所有依赖）

[pixi](https://pixi.sh/) 会同时安装 Python 依赖和外部工具（bedtools、hmmer、prodigal 等）。

```bash
# 1. 克隆仓库
git clone https://github.com/yourname/newbin.git
cd newbin

# 2. 一键安装所有依赖
pixi install

# 3. 激活环境
pixi shell

# 4. 验证安装
newbin --help
newbin check_install --verbose
```

**不安装直接运行**（零环境配置）：

```bash
pixi exec -c conda-forge -c bioconda -s newbin newbin --help
```

### 使用 conda（传统方式）

```bash
# 1. 创建环境
conda create -n newbin python=3.12
conda activate newbin

# 2. 安装外部工具
conda install -c bioconda bedtools hmmer prodigal samtools -y

# 3. 安装 NewBin
pip install .
```

### 使用 uv/pip（仅 Python 包）

```bash
# 创建虚拟环境
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 安装
uv pip install -e .

# 注意：还需手动安装外部工具
# conda install -c bioconda bedtools hmmer prodigal samtools -y
```

---

## 快速开始

### 单样本/共组装分箱（一键模式）

```bash
newbin single_easy_bin \
    --input-fasta contigs.fa \
    --input-bam mapped_reads.sorted.bam \
    --output output
```

长读数据添加 `--sequencing-type=long_read`：

```bash
newbin single_easy_bin \
    --input-fasta contigs.fa \
    --input-bam mapped_reads.sorted.bam \
    --sequencing-type long_read \
    --output output
```

### 多样本分箱（一键模式）

1. 合并多样本 FASTA（contig 名需包含样本前缀，如 `S1:Contig_1`）：

```bash
newbin concatenate_fasta \
    --input-fasta sample1.fa sample2.fa sample3.fa \
    --output combined
```

2. 运行多样本分箱：

```bash
newbin multi_easy_bin \
    --input-fasta combined/concatenated.fa \
    --input-bam *.sorted.bam \
    --output output
```

---

## 子命令详解

```
newbin <subcommand> [options]

核心命令：
  generate_sequence_features_single   生成单样本特征（k-mer + coverage）
  generate_sequence_features_multi    生成多样本特征
  train_self                          自监督训练模型
  bin                                 短读分箱（Infomap 聚类）
  bin_long                            长读分箱（DBSCAN 集成）

一体化命令：
  single_easy_bin                     特征 → 训练 → 分箱（单样本）
  multi_easy_bin                      特征 → 训练 → 分箱（多样本）

工具命令：
  concatenate_fasta                   合并多样本 FASTA
  check_install                       检查依赖是否完整
```

### 常用参数

| 参数 | 说明 |
|------|------|
| `-i, --input-fasta` | 输入 contig FASTA 文件 |
| `-b, --input-bam` | 输入 BAM 文件（可多个） |
| `-o, --output` | 输出目录 |
| `-p, --num-process` | 并行进程数（0 = 全部 CPU） |
| `--minfasta-kb` | 最小 bin 大小（默认 200 kb） |
| `--max-edges` | k-NN 图最大边数（默认 200） |
| `--sequencing-type` | `short_read` 或 `long_read` |
| `--engine` | `auto` / `gpu` / `cpu` |
| `--epochs` | 训练轮数（默认 15） |
| `--batch-size` | 批次大小（默认 2048） |
| `--random-seed` | 随机种子 |

---

## 输出说明

运行完成后，输出目录包含：

```
output/
├── data.csv                    # 特征矩阵（k-mer + coverage）
├── data_split.csv              # must-link split 特征
├── model.pt                    # 训练好的模型
├── markers.hmmout              # 标记基因检测结果
├── contig_bins.tsv             # contig → bin 映射表
├── bins_info.tsv               # bin 统计信息（N50、大小等）
├── output_bins/                # 分箱结果 FASTA
│   ├── bin.0.fa
│   ├── bin.1.fa
│   └── ...
└── output_recluster_bins/      # 重聚类后的 bin（短读模式）
    ├── recluster.0.fa
    └── ...
```

---

## 项目结构

```
newbin/
├── newbin/                     # Python 包
│   ├── __init__.py
│   ├── main.py                 # CLI 入口与工作流编排
│   ├── model.py                # 模型定义（占位，待实现）
│   ├── cluster.py              # 短读分箱：嵌入 + Infomap
│   ├── long_read_cluster.py    # 长读分箱：DBSCAN 集成
│   ├── generate_kmer.py        # k-mer 特征计算
│   ├── generate_coverage.py    # 覆盖度特征计算
│   ├── markers.py              # 标记基因检测
│   ├── orffinding.py           # ORF 预测
│   ├── naive_orffinder.py      # 内置 ORF 查找器
│   ├── utils.py                # 工具函数
│   ├── fasta.py                # FASTA 解析
│   ├── atomicwrite.py          # 原子写文件
│   ├── newbin_version.py       # 版本号
│   └── marker.hmm              # 107 个标记基因 HMM 模型
├── pyproject.toml              # 项目配置（含 pixi 环境）
├── requirements.txt            # Python 依赖
└── README.md                   # 本文件
```

---

## 核心设计

| 特性 | 说明 |
|------|------|
| `is_combined` 判断 | 通过 `data.csv` 最后一列名判断（`_var` 结尾 = 单 BAM，否则多样本） |
| `must_link_threshold` | 自动计算：累积长度超过 2% 处的 contig 长度，下限 4000 bp |
| `binned_length` | 自动选择 1000 或 2500 bp，基于 `--ratio` 阈值 |
| 标记基因 | 107 个单拷贝基因，用 `hmmsearch` + `marker.hmm` 检测 |
| ORF 查找 | 默认 `fast-naive`（内置），可选 `prodigal` / `fraggenescan` |
| 并行 | `multiprocessing.Pool`，CPU 数自动检测 |
| 设备 | `auto/gpu/cpu`，自动检测 CUDA |

---

## 开发

### 运行测试

```bash
pixi run -e test-py312 pytest
```

### 代码结构

- `main.py`：CLI 参数解析、子命令分发、流程编排
- `model.py`：**模型接口定义**，当前为占位实现，需用户自行实现训练循环
- `cluster.py` / `long_read_cluster.py`：聚类算法（短读 Infomap / 长读 DBSCAN）
- `generate_*.py`：特征工程（k-mer、覆盖度）

### 自定义模型

编辑 `newbin/model.py`，实现以下接口：

```python
class YourModel(nn.Module):
    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        """返回嵌入向量，shape [batch, embed_dim]"""
        ...

def train_model(logger, data, data_split, is_combined, batchsize, epochs, device, num_process, mode):
    """训练并返回模型"""
    ...
```

---

## 引用

如果你使用了 NewBin，请引用 SemiBin2 的原始论文：

> Pan, S.; Zhao, XM; Coelho, LP. [SemiBin2: self-supervised contrastive learning leads to better MAGs for short- and long-read sequencing](https://doi.org/10.1093/bioinformatics/btad209). *Bioinformatics* Volume 39, Issue Supplement_1, June 2023, Pages i21–i29.

---

## 许可证

MIT License
