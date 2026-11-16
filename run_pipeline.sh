#!/usr/bin/env bash
# ==============================================================================
# NewBin 宏基因组分箱全流程执行脚本 (End-to-End Metagenomic Binning Pipeline)
#
# 支持两种运行模式：
#   1. joint (多样本联合模式，默认推荐):
#      - 并发准备所有样本的 148 维共丰度特征与多样本聚合 PE 边
#      - 汇聚所有样本数据，调用 --train-from-many 集中训练 1 个全局通用模型
#      - 并发使用该全局模型对各个样本分别进行 Leiden 分箱与基准评估
#   2. independent (独立模式):
#      - 各样本完全独立执行 特征提取 -> 模型自监督训练 -> 分箱 -> 评估
#
# 使用方法：
#   ./run_pipeline.sh                  # 运行全部 10 个样本 (多样本联合模式)
#   ./run_pipeline.sh Mar_0            # 快速运行单个样本 Mar_0
#   PIPELINE_MODE=independent ./run_pipeline.sh  # 独立训练模式
# ==============================================================================

set -euo pipefail

# 终端高亮配色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ==============================================================================
# 核心配置参数区
# ==============================================================================
DATA_DIR="数据/联调数据"
SAMPLE="all"                              # "all" (全部10个样本) 或指定单样本如 "Mar_0"
PIPELINE_MODE="${PIPELINE_MODE:-joint}"   # 运行模式: "joint" (多样本联合训练) / "independent" (独立训练)
JOBS="${JOBS:-2}"                         # 最大并发任务数 (1 为串行，推荐 2~4)
TOP_N="${TOP_N:-}"                       # 留空为全量；指定数字 (如 5) 用于极速联调测试
EPOCHS=15                                 # 神经网络训练轮数
BATCH_SIZE=2048                           # Mini-batch 大小
DEVICE="auto"                             # 计算引擎: auto / gpu / cpu
OUTPUT_BASE="newbin_run"                  # 输出结果基准目录
MINFASTA_KB=200                           # 最终保留 MAG 分箱的最低碱基阈值 (kb)
RECLUSTER="auto"                          # 标记基因二次重聚类 (auto / yes / no)

# 允许可选第 1 个外部位置参数覆盖样本
SAMPLE="${1:-$SAMPLE}"

# 执行环境托管 (由 uv 统一免激活管理环境与依赖)
RUN="uv run --project newbin"

# 样本列表展开
if [[ "$SAMPLE" == "all" || "$SAMPLE" == "ALL" ]]; then
    SAMPLE_LIST=("Mar_0" "Mar_1" "Mar_2" "Mar_3" "Mar_4" "Mar_5" "Mar_6" "Mar_7" "Mar_8" "Mar_9")
else
    SAMPLE_LIST=("$SAMPLE")
    PIPELINE_MODE="independent" # 单样本强制使用独立训练模式
fi

echo -e "${CYAN}==============================================================================${NC}"
echo -e "${BOLD}${BLUE}               >>> NewBin 宏基因组分箱流水线 <<<${NC}"
echo -e "${CYAN}==============================================================================${NC}"
echo -e "  - 目标样本:       ${GREEN}${SAMPLE} (${#SAMPLE_LIST[@]} 个样本)${NC}"
echo -e "  - 流水线模式:     ${GREEN}${PIPELINE_MODE} $([ "$PIPELINE_MODE" == "joint" ] && echo "(多样本联合训练模式)" || echo "(样本独立训练模式)")${NC}"
echo -e "  - 并发任务数:     ${GREEN}${JOBS} $([ "$JOBS" -gt 1 ] && echo "(并行模式)" || echo "(串行模式)")${NC}"
echo -e "  - 数据规模:       ${GREEN}$([ -n "$TOP_N" ] && echo "前 $TOP_N 条 (测试模式)" || echo "全量数据")${NC}"
echo -e "  - 结果输出目录:   ${OUTPUT_BASE}"
echo -e "  - 训练参数:       ${EPOCHS} epochs, batch_size=${BATCH_SIZE}, engine=${DEVICE}"
echo -e "${CYAN}------------------------------------------------------------------------------${NC}\n"

# ------------------------------------------------------------------------------
# 阶段任务定义
# ------------------------------------------------------------------------------

# 阶段 1: 数据转换与特征工程 (生成 contigs.fa, 148维 data.csv, 聚合 pe_links.csv)
prepare_sample() {
    local cur_sample="$1"
    local cur_out="${OUTPUT_BASE}/${cur_sample}"
    mkdir -p "$cur_out"

    local top_n_param=""
    if [[ -n "$TOP_N" ]]; then
        top_n_param="--top_n $TOP_N"
    fi

    $RUN python prepare_for_newbin.py \
        --data_dir "$DATA_DIR" \
        --sample "$cur_sample" \
        --output_dir "$cur_out" \
        $top_n_param
}

# 阶段 2: 独立模型训练 (仅 independent 模式使用)
train_sample() {
    local cur_sample="$1"
    local cur_out="${OUTPUT_BASE}/${cur_sample}"

    $RUN newbin train_self \
        -d "$cur_out/data.csv" \
        -o "$cur_out" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --engine "$DEVICE"
}

# 阶段 3 & 4: 分箱聚类与基准评估
bin_and_eval_sample() {
    local cur_sample="$1"
    local cur_out="${OUTPUT_BASE}/${cur_sample}"

    # 确定所用模型路径 (联合模式优先使用全局联合模型)
    local model_path="${cur_out}/model.pt"
    if [[ "$PIPELINE_MODE" == "joint" && -f "${OUTPUT_BASE}/joint_model/model.pt" ]]; then
        model_path="${OUTPUT_BASE}/joint_model/model.pt"
    fi

    local bin_extra_args=()
    if [[ -n "$TOP_N" && "$TOP_N" -le 1000 ]]; then
        bin_extra_args+=(--minfasta-kb 0 --no-recluster)
    else
        bin_extra_args+=(--minfasta-kb "$MINFASTA_KB")
        if [[ "$RECLUSTER" == "no" ]]; then
            bin_extra_args+=(--no-recluster)
        elif [[ "$RECLUSTER" == "auto" ]]; then
            if ! command -v hmmsearch &>/dev/null; then
                bin_extra_args+=(--no-recluster)
            fi
        fi
    fi

    local pe_links_arg=()
    if [[ -f "$cur_out/pe_links.csv" ]]; then
        pe_links_arg+=(--pe-links "$cur_out/pe_links.csv")
    fi

    $RUN newbin bin \
        -i "$cur_out/contigs.fa" \
        -d "$cur_out/data.csv" \
        --model-path "$model_path" \
        -o "$cur_out" \
        "${pe_links_arg[@]}" \
        "${bin_extra_args[@]}"

    if [[ -f "$cur_out/ground_truth.tsv" ]]; then
        $RUN newbin evaluate \
            -b "$cur_out/contig_bins.tsv" \
            -g "$cur_out/ground_truth.tsv" \
            -f "$cur_out/contigs.fa" \
            -o "$cur_out/eval"
    fi
}

# 单样本端到端串联 (仅 independent 模式使用)
run_full_single_sample() {
    local cur_sample="$1"
    prepare_sample "$cur_sample"
    train_sample "$cur_sample"
    bin_and_eval_sample "$cur_sample"
}

# ------------------------------------------------------------------------------
# 进程池派发器
# ------------------------------------------------------------------------------
dispatch_jobs() {
    local worker_fn="$1"
    local stage_title="$2"
    shift 2
    local items=("$@")

    echo -e "${BOLD}${YELLOW}>>> [${stage_title}] (总任务数: ${#items[@]}, 并发数: ${JOBS})${NC}"

    if [[ "$JOBS" -le 1 ]]; then
        for item in "${items[@]}"; do
            echo -e "${CYAN}  -> 处理样本 [${item}]...${NC}"
            "$worker_fn" "$item"
        done
        return 0
    fi

    local pids=()
    local names=()
    for item in "${items[@]}"; do
        while [[ "${#pids[@]}" -ge "$JOBS" ]]; do
            local next_pids=()
            local next_names=()
            for i in "${!pids[@]}"; do
                if kill -0 "${pids[$i]}" 2>/dev/null; then
                    next_pids+=("${pids[$i]}")
                    next_names+=("${names[$i]}")
                else
                    wait "${pids[$i]}"
                    echo -e "${GREEN}  ✔ 样本 [${names[$i]}] 阶段完成${NC}"
                fi
            done
            pids=("${next_pids[@]}")
            names=("${next_names[@]}")
            sleep 0.5
        done

        local item_out="${OUTPUT_BASE}/${item}"
        mkdir -p "$item_out"
        echo -e "${CYAN}  🚀 样本 [${item}] 进入并发队列...${NC}"
        "$worker_fn" "$item" >> "${item_out}/pipeline.log" 2>&1 &
        pids+=($!)
        names+=("$item")
    done

    for i in "${!pids[@]}"; do
        wait "${pids[$i]}"
        echo -e "${GREEN}  ✔ 样本 [${names[$i]}] 阶段完成${NC}"
    done
}

# ------------------------------------------------------------------------------
# 主流程编排
# ------------------------------------------------------------------------------
GLOBAL_START=$(date +%s)

if [[ "$PIPELINE_MODE" == "joint" ]]; then
    # 阶段 1: 并发特征工程
    dispatch_jobs prepare_sample "阶段 1/3: 样本特征工程与多样本PE聚合" "${SAMPLE_LIST[@]}"

    # 阶段 2: 集中联合自监督训练
    echo -e "\n${BOLD}${YELLOW}>>> [阶段 2/3: 多样本联合自监督训练] 汇聚 ${#SAMPLE_LIST[@]} 个样本联合训练全局模型...${NC}"
    ALL_DATA_CSVS=()
    for s in "${SAMPLE_LIST[@]}"; do
        ALL_DATA_CSVS+=("${OUTPUT_BASE}/${s}/data.csv")
    done
    mkdir -p "${OUTPUT_BASE}/joint_model"

    $RUN newbin train_self \
        --train-from-many \
        -d "${ALL_DATA_CSVS[@]}" \
        -o "${OUTPUT_BASE}/joint_model" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --engine "$DEVICE"

    echo -e "${GREEN}✔ 全局联合模型训练完成，权重保存至: ${OUTPUT_BASE}/joint_model/model.pt${NC}\n"

    # 阶段 3 & 4: 并发分箱与基准评估
    dispatch_jobs bin_and_eval_sample "阶段 3/3: 双轨增强图分箱与基准评估" "${SAMPLE_LIST[@]}"

else
    # 独立训练流水线
    dispatch_jobs run_full_single_sample "样本独立端到端流水线" "${SAMPLE_LIST[@]}"
fi

GLOBAL_END=$(date +%s)
echo -e "\n${CYAN}==============================================================================${NC}"
echo -e "${BOLD}${GREEN}✔ NewBin 流水线全部执行完成！总耗时: $((GLOBAL_END - GLOBAL_START)) 秒${NC}"
echo -e "  - 结果目录: ${OUTPUT_BASE}/"
if [[ "$PIPELINE_MODE" == "joint" ]]; then
    echo -e "  - 全局通用模型: ${OUTPUT_BASE}/joint_model/model.pt"
fi
echo -e "${CYAN}==============================================================================${NC}\n"
