#!/bin/bash
#SBATCH --job-name=newbin_bin        # 任务名称: 分箱与评估
#SBATCH --partition=64c512g          # 使用 CPU 队列 (64核/512GB大内存节点)
#SBATCH --nodes=1                    # 单节点
#SBATCH --ntasks-per-node=1          # 单进程
#SBATCH --cpus-per-task=16           # 分配 16 核 CPU (供 Leiden 并发聚类与评估)
#SBATCH --mem=32G                    # 分配 32GB 内存
#SBATCH --output=newbin_bin_cpu_job%j_%N.out  # 标准输出日志
#SBATCH --error=newbin_bin_cpu_job%j_%N.err   # 错误输出日志

# ------------------------------------------------------------------------------
# 1. 环境准备与防踩坑配置
# ------------------------------------------------------------------------------
unset SSL_CERT_FILE
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
module load gcc 2>/dev/null || true

cd "$SLURM_SUBMIT_DIR"

# ------------------------------------------------------------------------------
# 2. 检查 CPU 节点状态并启动分箱与评估流水线
# ------------------------------------------------------------------------------
echo "==================== 任务启动信息 ===================="
echo "运行节点: $(hostname)"
echo "分配 CPU 核心数: ${SLURM_CPUS_PER_TASK:-16}"
echo "======================================================"

# 执行分箱与评估 (阶段 3 & 4)
bash run_pipeline.sh
