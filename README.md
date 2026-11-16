# NewBin: 新一代深度学习宏基因组分箱框架 (Metagenomic Binning)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)

**NewBin** 是一个基于深度表征学习与双轨增强图聚类的高精度**宏基因组分箱（Metagenomic Binning）**工具。它在特征工程、度量学习体系以及图聚类切分上进行了全面升级：
- **特征工程深度重构**：采用 2-bit 向量化查找表（LUT）极速提取 136 维 Canonical 4-mer；提出全新 $2M+1$ 维 Coverage 架构（解耦全局测序深度基准 Level、相对起伏剖面 Shape 及位置级离散度 $\log\mathrm{ID}$），在数学上精确抹平同种属内拷贝数倍数偏差（$c_i$）。
- **余弦度量与对比学习**：全面拥抱单位超球面嵌入空间（L2 Normalization），采用多正对优化的 InfoNCE / SupCon 损失函数与难负样本挖掘（Hard Negative Mining）。
- **双轨图聚类（Leiden + PE 物理先验）**：以余弦高维表征构建 k-NN 稀疏骨架，创造性地融合**双端测序（Paired-end Bridging Reads）物理连通先验**，利用 Leiden 算法实现高质量社区划分（并支持 HDBSCAN 空间密度聚类消融对照）。
- **支持全场景分箱**：完美覆盖短读长（Short-read）、长读长（Long-read）、单样本/共组装（Single-sample / Co-assembly）与多样本（Multi-sample）分箱。

---

## 目录
- [一、 完整分箱流程（整体分 5 步）](#一-完整分箱流程整体分-5-步)
- [二、 安装指南](#二-安装指南)
- [三、 快速上手（一键模式）](#三-快速上手一键模式)
- [四、 命令行子命令与分步执行指南](#四-命令行子命令与分步执行指南)
- [五、 输入与输出数据规范](#五-输入与输出数据规范)
- [六、 项目代码结构](#六-项目代码结构)

---

## 一、 完整分箱流程（整体分 5 步）

整个 NewBin 的处理流水线严格解耦为五个标准化步骤：

```
[步骤 1: 特征工程 (Feature Generation)]
  FASTA 序列 + BAM 比对深度
    ├── 136 维 4-mer 组成特征 (2-bit LUT 极速算法) + contig 长度
    └── 2M + 1 维 Coverage 特征 (Level 1维 + Shape M维 + log_ID M维)
    └── 水平拼接生成标准 data.csv
          ↓
[步骤 2: 模型训练 (Model Training)]
  data.csv + 先验约束 (must_link / cannot_link)
    ├── 基于超球面 L2 归一化 + InfoNCE / SupCon 对比学习 (或 Hinge 损失)
    └── 训练特征编码器，输出权重 model.pt
          ↓
[步骤 3: 向量嵌入与降维 (Embedding Inference)]
  model.pt + data.csv
    └── 前向推理将每条 contig 映射到 100 维超球面连续表征空间
          ↓
[步骤 4: 图构建与聚类分箱 (Clustering & Binning)]
  100 维 Embedding (+ 双端 PE Reads 物理边)
    ├── 主方案 (短读长): 余弦 k-NN 图构建 + PE 物理边注入 + Leiden 社区划分
    ├── 对照组方案 (长读长/消融): HDBSCAN 多尺度层级空间密度聚类
    └── 过滤碎片 (<200kb) 导出最终 MAGs (bin.X.fa) 并统计 N50 / L50
          ↓
[步骤 5: 基准质量评估 (Evaluation, 可选)]
  binning 结果 + 真实物种注释 (Ground Truth, 如 ctg2genome.tsv)
    └── 自动评估 Completeness (完整度)、Purity (纯度)、Contamination (污染度)、F1-score
```

### 1. 步骤 1：特征提取（Feature Generation）
- **序列组成**：使用 2-bit 向量化查找表（LUT）极速提取 136 维 Canonical 4-mer 相对频率，并附带序列实际长度 `length`。
- **测序深度**：
  - **多样本共丰度模式（$M \ge 5$）**：优选 5 个高深度样本计算 11 维 Coverage（1 维 Level + 5 维 Shape + 5 维 $\log\mathrm{ID}$），彻底抹除重复区域拷贝数倍数偏差；
  - **单样本模式（$M < 5$）**：无缝归约为主样本自身深度（Level + $\log\mathrm{ID}$），保留单样本测序丰度与平整度信息。
- **输出**：生成唯一标准特征表 `data.csv`。

### 2. 步骤 2：模型训练（Model Training）
- 读取 `data.csv` 以及外部准备好的高质量先验约束对：`--must-link`（正对，同基因组）与 `--cannot-link`（负对，跨物种冲突）。
- 在单位超球面上优化特征编码器，通过拉近正对、推开负对学习高判别性表征，保存为 `model.pt`。

### 3. 步骤 3：特征嵌入推理（Embedding Inference）
- 加载 `model.pt`，对全体 contig 进行前向推理，映射为 100 维超球面归一化向量（`[N, 100]`）。

### 4. 步骤 4：图构建与聚类分箱（Clustering & Binning）
- **主方案（双轨增强图聚类）**：基于超球面余弦相似度构建稀疏近邻图（$k$-NN），注入双端测序（PE Bridging Reads）物理共存边，由 **Leiden 算法** 进行社群切分，大幅提高质粒与短序列召回率。
- **对照方案（空间密度聚类）**：采用 **HDBSCAN** 进行多尺度空间密度聚类。
- **结果导出**：过滤总碱基 $< 200\text{ kb}$ 的无效碎片，将同类 contigs 导出为独立的 FASTA 文件（`bin.X.fa`），输出分箱质量报告。

### 5. 步骤 5：质量基准评测（Evaluation）
- 读取预测结果与真实参考标签（Ground Truth，如 CAMI II 的 `ctg2genome.tsv`），按国际标准自动计算长度加权的完整度、纯度、污染度与 F1 得分。

---

## 二、 安装指南

### 方式 A：使用 pixi（推荐，一键安装所有环境及生信依赖）
```bash
# 1. 克隆仓库
git clone https://github.com/118852995312/newbin.git
cd newbin/newbin

# 2. 一键安装 Python 及生信工具链 (bedtools, samtools, hmmer 等)
pixi install

# 3. 激活环境
pixi shell

# 4. 验证环境
newbin check_install --verbose
```

### 方式 B：使用 conda（传统方式）
```bash
# 1. 创建环境并安装底层生信工具
conda create -n newbin python=3.12 -y
conda activate newbin
conda install -c bioconda -c conda-forge bedtools samtools hmmer igraph -y

# 2. 安装 NewBin
cd newbin/newbin
pip install -e .
```

---

## 三、 快速上手（一键模式）

### 1. 单样本 / 共组装模式（All-in-one）
```bash
newbin single_easy_bin \
    -i contigs.fa \
    -b sample.sorted.bam \
    -o output_dir
```

长读长数据增加 `--sequencing-type long_read`：
```bash
newbin single_easy_bin \
    -i contigs.fa \
    -b sample.sorted.bam \
    --sequencing-type long_read \
    -o output_dir
```

### 2. 多样本模式（All-in-one）
如果各样本的 FASTA 独立，先将其拼接（contig 名称自动注入样本前缀，例如 `SampleA:k141_1`）：
```bash
newbin concatenate_fasta \
    -i sample1.fa sample2.fa sample3.fa \
    -o combined_dir
```
一键运行多样本流水线：
```bash
newbin multi_easy_bin \
    -i combined_dir/concatenated.fa \
    -b sample1.sorted.bam sample2.sorted.bam sample3.sorted.bam \
    -o output_multi
```

---

## 四、 命令行子命令与分步执行指南

NewBin 提供了高度可定制的模块化子命令：

```text
newbin <subcommand> [options]

核心工作流：
  generate_sequence_features_single   [步骤 1] 生成单样本特征表 (data.csv)
  generate_sequence_features_multi    [步骤 1] 生成多样本特征表 (每个样本独立 data.csv)
  train_self                          [步骤 2] 训练对比学习特征编码器 (model.pt)
  bin                                 [步骤 3+4] 短读长图聚类分箱 (Leiden / Infomap)
  bin_long                            [步骤 3+4] 长读长密度聚类分箱 (HDBSCAN)

一键一体化命令：
  single_easy_bin                     单样本模式：特征提取 → 模型训练 → 聚类分箱
  multi_easy_bin                      多样本模式：样本拆分 → 特征提取 → 独立训练分箱

评测与辅助工具：
  evaluate                            [步骤 5] 对齐 CAMI / AMBER 标准计算完整度/污染度
  concatenate_fasta                   合并多样本 FASTA 并添加前缀分隔符
  check_install                       检查系统依赖工具是否完整
```

### 分步执行典型范例：

```bash
# 步骤 1：提取特征矩阵 (data.csv)
newbin generate_sequence_features_single \
    -i contigs.fa \
    -b sample.sorted.bam \
    -o output_features

# 步骤 2：基于先验约束训练模型
newbin train_self \
    -d output_features/data.csv \
    --must-link constraints/must_link.csv \
    --cannot-link constraints/cannot_link.csv \
    --metric-system cosine \
    --loss-type infonce \
    --epochs 20 \
    -o output_model

# 步骤 3 & 4：聚类分箱 (Leiden 算法)
newbin bin \
    -i contigs.fa \
    -d output_features/data.csv \
    --model-path output_model/model.pt \
    --clustering-algorithm leiden \
    --metric-system cosine \
    -o output_bins

# 步骤 5：基准评测
newbin evaluate \
    -b output_bins \
    -g ground_truth/ctg2genome.tsv \
    -f contigs.fa \
    -o eval_report
```

---

## 五、 输入与输出数据规范

### 1. 输入数据规范
- **Contigs FASTA (`-i`)**：过滤 $< 1000\text{ bp}$ 的序列。多样本模式下 Header 必须带有分隔符（如 `>SampleA:contig_1`）。
- **比对文件 BAM (`-b`)**：必须为 Coordinate-sorted 且具备索引（`.bam.bai`）的 BAM 文件。
- **先验约束文件（`--must-link` / `--cannot-link`）**：无表头 CSV 格式，每行两个 contig ID（如 `contig_A,contig_B`）。

### 2. 输出目录结构
```text
output/
├── data.csv                    # 完整特征矩阵 (136维 k-mer + 2M+1维 Coverage + 1维 长度)
├── model.pt                    # 训练好的 PyTorch 模型权重
├── contig_bins.tsv             # Contig 到 Bin 映射表
├── bins_info.tsv               # 分箱统计摘要 (N50, L50, 大小)
└── output_bins/                # 分箱 FASTA 结果目录
    ├── bin.0.fa
    ├── bin.1.fa
    └── ...
```

---

## 六、 项目代码结构

```text
newbin/
├── docs/                       # 技术与规格白皮书
│   ├── constraints_and_kmer_spec.md
│   └── 工作记录与后续开发规划.md
├── newbin/
│   ├── main.py                 # CLI 入口与工作流编排
│   ├── model.py                # 深度学习网络与对比损失训练循环
│   ├── cluster.py              # 短读分箱：余弦 k-NN 图构建 + Leiden/Infomap
│   ├── long_read_cluster.py    # 长读分箱：HDBSCAN 空间密度聚类
│   ├── generate_kmer.py        # 2-bit LUT 极速 136 维 4-mer 特征计算
│   ├── generate_coverage.py    # 2M+1 维 Coverage (Level/Shape/log_ID) 计算
│   ├── evaluate.py             # CAMI / AMBER 标准质量评估模块
│   └── utils.py                # 工具函数与数据清洗
├── tests/                      # 单元测试与测试数据集
│   └── test_kmer_and_cov.py    # 特征工程数值与数学等价性测试
├── pyproject.toml              # 包配置与 Pixi 环境定义
└── requirements.txt            # Python 依赖清单
```
