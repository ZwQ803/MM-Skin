import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
import seaborn as sns
import matplotlib.pyplot as plt
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from transformers import AutoTokenizer, AutoModel
import nltk
from metric.Bertscore import BERTScorer


# 计算BLEU分数
def calculate_bleu(predictions, ground_truths):
    bleu_scores = {
        "bleu-1": corpus_bleu([[truth.split()] for truth in ground_truths], [pred.split() for pred in predictions],
                              weights=(1, 0, 0, 0)),
        "bleu-2": corpus_bleu([[truth.split()] for truth in ground_truths], [pred.split() for pred in predictions],
                              weights=(0.5, 0.5, 0, 0)),
        "bleu-3": corpus_bleu([[truth.split()] for truth in ground_truths], [pred.split() for pred in predictions],
                              weights=(0.33, 0.33, 0.33, 0)),
        "bleu-4": corpus_bleu([[truth.split()] for truth in ground_truths], [pred.split() for pred in predictions],
                              weights=(0.25, 0.25, 0.25, 0.25))
    }
    return bleu_scores


# 计算ROUGE分数
def calculate_rouge(predictions, ground_truths):
    rouge = Rouge()
    scores = rouge.get_scores(predictions, ground_truths, avg=True)
    return scores


# 计算METEOR分数
def calculate_meteor(predictions, ground_truths):
    meteor_scores = [nltk.translate.meteor_score.meteor_score([truth.split()], pred.split()) for pred, truth in
                     zip(predictions, ground_truths)]
    return sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0


# 计算BERTScore
def calculate_bertscore(predictions, ground_truths):
    model_path = '/home/user6/model/bert-base-uncased'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer = BERTScorer(model_type=model_path, device=device, lang='en')
    precision, recall, f1 = scorer.score(predictions, ground_truths)
    # 计算平均值
    avg_precision = precision.mean().item()
    avg_recall = recall.mean().item()
    avg_f1 = f1.mean().item()

    return {
        "bert_precision": avg_precision,
        "bert_recall": avg_recall,
        "bert_f1": avg_f1
    }


# 计算Recall
def calculate_recall(predictions, ground_truths):
    recall_scores = []
    for pred, truth in zip(predictions, ground_truths):
        pred_tokens = set(pred.split())
        truth_tokens = set(truth.split())
        recall_scores.append(
            len(pred_tokens.intersection(truth_tokens)) / len(truth_tokens) if len(truth_tokens) > 0 else 0)
    return sum(recall_scores) / len(recall_scores) if recall_scores else 0

def get_modality_from_image(image_name, captions_file):
    df_caps = pd.read_csv(captions_file)
    vals = df_caps.loc[df_caps['image'] == image_name, 'modality'].values
    return vals[0] if len(vals) > 0 else None


# 主评估函数
def eval_model(args):
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.expanduser(args.model_path)
    model_name = args.model_name


    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
        model = model.module

    model = model.to(device)
    # 加载VQA数据集
    df = pd.read_csv(os.path.expanduser(args.dataset_file))
    df['modality'] = df['image'].apply(lambda im: get_modality_from_image(im, args.captions_file))

    predictions = []
    ground_truths = []
    modalities = []

    os.makedirs(args.result_dir, exist_ok=True)
    result_file = os.path.join(args.result_dir, "predictions.csv")
    if not os.path.exists(result_file):
        pd.DataFrame(columns=["image", "question", "predicted_answer", "ground_truth", "modality"]).to_csv(result_file, index=False)

    modality_results = {}

    # 推理与评估
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        image_file = row["image"]
        question = row["question"]
        ground_truth = row["answer"]
        modality = row["modality"]

        cur_prompt = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()

        if model.config.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
            print(question)
        else:
            question = DEFAULT_IMAGE_TOKEN + '\n' + question
            print(question)

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda().to(device)

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_size = image.size
        # print(image_size)
        image_tensor = process_images([image], image_processor, model.config)[0].to(device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode(), torch.cuda.amp.autocast():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                min_new_tokens=20,
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id,
                # image_sizes=[image_size],
                use_cache=True)

        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(output)

        # 逐行追加保存预测结果
        new_data = pd.DataFrame({
            "image": [image_file],
            "question": row["question"],
            "predicted_answer": [output],
            "ground_truth": [ground_truth],
            "modality": [modality]
        })
        new_data.to_csv(result_file, mode='a', header=False, index=False)

        if modality not in modality_results:
            modality_results[modality] = {"predictions": [], "ground_truths": []}

        modality_results[modality]["predictions"].append(output)
        modality_results[modality]["ground_truths"].append(ground_truth)

    # 计算每个模态的评估指标并保存
    eval_metrics = {}
    for modality, results in modality_results.items():
        bleu_scores = calculate_bleu(results["predictions"], results["ground_truths"])
        rouge_scores = calculate_rouge(results["predictions"], results["ground_truths"])
        meteor_score = calculate_meteor(results["predictions"], results["ground_truths"])
        bert_score = calculate_bertscore(results["predictions"], results["ground_truths"])
        recall_score = calculate_recall(results["predictions"], results["ground_truths"])

        # 将每个modality的评估结果保存到eval_metrics字典
        eval_metrics[f"{modality}_bleu-1"] = bleu_scores["bleu-1"]
        eval_metrics[f"{modality}_bleu-2"] = bleu_scores["bleu-2"]
        eval_metrics[f"{modality}_bleu-3"] = bleu_scores["bleu-3"]
        eval_metrics[f"{modality}_bleu-4"] = bleu_scores["bleu-4"]
        eval_metrics[f"{modality}_meteor"] = meteor_score
        eval_metrics[f"{modality}_rouge-1"] = rouge_scores["rouge-1"]["f"]
        eval_metrics[f"{modality}_rouge-2"] = rouge_scores["rouge-2"]["f"]
        eval_metrics[f"{modality}_rouge-l"] = rouge_scores["rouge-l"]["f"]
        eval_metrics[f"{modality}_bert_precision"] = bert_score["bert_precision"]
        eval_metrics[f"{modality}_bert_recall"] = bert_score["bert_recall"]
        eval_metrics[f"{modality}_bert_f1"] = bert_score["bert_f1"]
        eval_metrics[f"{modality}_recall"] = recall_score

    # 保存评估指标到CSV文件
    eval_file = os.path.join(args.result_dir, "evaluation_metrics.csv")
    eval_df = pd.DataFrame([eval_metrics])
    eval_df.to_csv(eval_file, index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 我训练的lora train merged model VQA test
    parser.add_argument("--model-path", type=str, default="/merge/SkinVL_PubMM",help="Path to the pretrained model")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="LlavaMistralForCausalLM")
    parser.add_argument("--result-dir", type=str, default="result/VQA/SkinVL_PubMM", help="Directory to save results")
    parser.add_argument("--conv-mode", type=str, default="mistral_instruct", help="Conversation mode for model")
    parser.add_argument("--captions-file", type=str, default='/Dataset/MM-Skin/caption.csv', help="File with modality information")


    # LLAVA其他模型 如llava-llama
    # parser.add_argument("--model-path", type=str, default="/home/user6/model/llava-v1.6-vicuna-7b")
    # parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--model-name", type=str, default="LlavaLlamaForCausalLM")
    # parser.add_argument("--result-dir", type=str, default="result/llava-v1.6-vicuna-7b_LlavaLlamaForCausalLM", help="Directory to save results")
    # parser.add_argument("--conv-mode", type=str, default="v1", help="Conversation mode for model")
    # 默认配置
    parser.add_argument("--image-folder", type=str, default="/Dataset/MM-Skin", help="Folder containing images")
    parser.add_argument("--dataset-file", type=str, default="/Dataframe/test/vqa/MM-Skin_test.csv",help="VQA dataset file (CSV)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for beam search")
    parser.add_argument("--load-8bit", type=bool, default=False, help="Load model with 8-bit precision")
    parser.add_argument("--load-4bit", type=bool, default=False, help="Load model with 4-bit precision")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (cuda or cpu)")

    args = parser.parse_args()

    eval_model(args)
