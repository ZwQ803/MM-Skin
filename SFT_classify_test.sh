#!/bin/bash


OUTPUT_LOG="/result/SFTclassification/output.txt"
PATH_RESULTS_TRANSFERABILITY="/result/SFTclassification/"

# 定义要测试的实验列表
EXPERIMENTS=(
    "HAM10000"
    "DDI_2class"
    "Dermnet"
    "Fitzpatrick"
    "HIBA_2class"
    "ISIC"
    "PAD"
    "Patch16_2class"
)
# 定义CUDA设备
CUDA_DEVICE=6,7

OUTPATH="${PATH_RESULTS_TRANSFERABILITY}SkinVL_PubMM/"
WEIGHTSPATH=/merge/SkinVL_PubMM


# 遍历每个实验
for EXP in "${EXPERIMENTS[@]}"; do
    echo "===============================" | tee -a "$OUTPUT_LOG"
    echo "Starting experiments for: $EXP" | tee -a "$OUTPUT_LOG"
    echo "===============================" | tee -a "$OUTPUT_LOG"

#    echo "Running Experiment $EXP: shots_train=$TRAIN_SHOTS, shots_test=20% (Fine-tuning)" | tee -a "$OUTPUT_LOG"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python classification_LLava.py \
        --experiment "$EXP" \
        --result_dir "$OUTPATH" \
        --model_path "$WEIGHTSPATH" \

    echo "" | tee -a "$OUTPUT_LOG"
#  done
  echo "Completed experiments for: $EXP" | tee -a "$OUTPUT_LOG"
  echo "" | tee -a "$OUTPUT_LOG"
done

echo "All experiments completed." | tee -a "$OUTPUT_LOG"

