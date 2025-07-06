# 定义要测试的实验列表
EXPERIMENTS=(
    "HAM10000"
    "PAD"
    "Patch16_2class"
)
# 定义CUDA设备
CUDA_DEVICE=4

OUTPUT_LOG="log.txt"
OUTPATH="result/test/zeroshot_class/9pubCHOICE"
WEIGHTSPATH=/merge/SkinVL_PubMM


for EXP in "${EXPERIMENTS[@]}"; do
    echo "===============================" | tee -a "$OUTPUT_LOG"
    echo "Starting experiments for: $EXP" | tee -a "$OUTPUT_LOG"
    echo "===============================" | tee -a "$OUTPUT_LOG"

#    echo "Running Experiment $EXP: shots_train=$TRAIN_SHOTS, shots_test=20% (Fine-tuning)" | tee -a "$OUTPUT_LOG"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python zero_class.py \
        --experiment "$EXP" \
        --result-path "$OUTPATH" \
        --model_path "$WEIGHTSPATH" \

    echo "" | tee -a "$OUTPUT_LOG"
#  done
  echo "Completed experiments for: $EXP" | tee -a "$OUTPUT_LOG"
  echo "" | tee -a "$OUTPUT_LOG"
done