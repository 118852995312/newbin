#!/bin/bash
#SBATCH --job-name=newbin_train       # 任务名称
#SBATCH --partition=a100             # 使用 A100 显卡队列
#SBATCH --nodes=1                    # 单节点
#SBATCH --ntasks-per-node=1          # 单进程
#SBATCH --cpus-per-task=16           # 分配 16 核 CPU (加快 DataLoader 批处理)
#SBATCH --gres=gpu:1                 # 申请 1 块 A100 GPU
#SBATCH --mem=32G                    # 分配 32GB 内存 (防止爆内存)
#SBATCH --output=newbin_train_A100_job%j_%N.out  # 标准输出 (格式: 项目名_硬件_JobID_运行节点.out)
#SBATCH --error=newbin_train_A100_job%j_%N.err   # 错误输出 (格式: 项目名_硬件_JobID_运行节点.err)

# ------------------------------------------------------------------------------
# 1. 环境准备与防踩坑配置
# ------------------------------------------------------------------------------
# 避免无效 SSL 证书阻断依赖下载
unset SSL_CERT_FILE

# 确保非交互式 Shell 能够找到 uv 和用户自装工具
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# 加载集群自带的 CUDA 和 GCC 模块
module load gcc cuda 2>/dev/null || true

# 进入任务提交的当前目录
cd "$SLURM_SUBMIT_DIR"

# ------------------------------------------------------------------------------
# 2. 检查显卡状态并启动训练
# ------------------------------------------------------------------------------
echo "==================== 任务启动信息 ===================="
echo "运行节点: $(hostname)"
echo "分配显卡: $CUDA_VISIBLE_DEVICES"
nvidia-smi 2>/dev/null || true
echo "======================================================"

# 执行训练流水线 (推荐使用 bash 执行)
bash run_pipeline.sh
