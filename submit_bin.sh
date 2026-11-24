#!/bin/bash
#SBATCH --job-name=newbin_bin        # 任务名称: 分箱与评估
#SBATCH --partition=a100             # 使用 A100 显卡队列 (加速 Embedding 向量计算)
#SBATCH --nodes=1                    # 单节点
#SBATCH --ntasks-per-node=1          # 单进程
#SBATCH --cpus-per-task=16           # 分配 16 核 CPU (供 Leiden 并发聚类与评估)
#SBATCH --gres=gpu:1                 # 申请 1 块 A100 GPU
#SBATCH --mem=32G                    # 分配 32GB 内存
#SBATCH --output=newbin_bin_A100_job%j_%N.out  # 标准输出日志
#SBATCH --error=newbin_bin_A100_job%j_%N.err   # 错误输出日志

# ------------------------------------------------------------------------------
# 1. 环境准备与防踩坑配置
# ------------------------------------------------------------------------------
unset SSL_CERT_FILE
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
module load gcc cuda 2>/dev/null || true

cd "$SLURM_SUBMIT_DIR"

# ------------------------------------------------------------------------------
# 2. 检查显卡状态并启动分箱与评估流水线
# ------------------------------------------------------------------------------
echo "==================== 任务启动信息 ===================="
echo "运行节点: $(hostname)"
echo "分配显卡: $CUDA_VISIBLE_DEVICES"
nvidia-smi 2>/dev/null || true
echo "======================================================"

# 执行分箱与评估 (阶段 3 & 4)
bash run_pipeline.sh
