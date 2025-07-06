import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import pandas as pd

from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
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


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Load the VQA test dataset
    with open(os.path.expanduser(args.question_file), 'r') as f:
        questions = json.load(f)

    result_file = os.path.expanduser(args.result_file)

    # Ensure the CSV file exists and has the correct header
    if not os.path.exists(result_file):
        pd.DataFrame(columns=["question", "predicted_answer", "ground_truth"]).to_csv(result_file, index=False)

    predictions = []
    ground_truths = []

    # Iterate through the dataset and get predictions
    for line in tqdm(questions, total=len(questions)):
        question = line["conversations"][0]["value"]  # human's question
        ground_truth = line["conversations"][1]["value"]  # gpt's answer
        image_file = line["image"]

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=False
            )

        predicted_answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # Save the prediction and ground truth to CSV
        new_data = pd.DataFrame({
            "question": [question],
            "predicted_answer": [predicted_answer],
            "ground_truth": [ground_truth]
        })
        new_data.to_csv(result_file, mode='a', header=False, index=False)

        predictions.append(predicted_answer)
        ground_truths.append(ground_truth)

    # Calculate evaluation metrics
    bleu_scores = calculate_bleu(predictions, ground_truths)
    rouge_scores = calculate_rouge(predictions, ground_truths)
    meteor_score = calculate_meteor(predictions, ground_truths)
    bert_score = calculate_bertscore(predictions, ground_truths)
    recall_score = calculate_recall(predictions, ground_truths)

    # Save the evaluation metrics to CSV
    eval_metrics = {
        "bleu-1": bleu_scores["bleu-1"],
        "bleu-2": bleu_scores["bleu-2"],
        "bleu-3": bleu_scores["bleu-3"],
        "bleu-4": bleu_scores["bleu-4"],
        "meteor": meteor_score,
        "rouge-1": rouge_scores["rouge-1"]["f"],
        "rouge-2": rouge_scores["rouge-2"]["f"],
        "rouge-l": rouge_scores["rouge-l"]["f"],
        "bert_precision": bert_score["bert_precision"],
        "bert_recall": bert_score["bert_recall"],
        "bert_f1": bert_score["bert_f1"],
        "recall": recall_score,
    }

    eval_file = os.path.join(args.result_dir, "evaluation_metrics.csv")
    eval_df = pd.DataFrame([eval_metrics])
    eval_df.to_csv(eval_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/user6/LLaVA/checkpoints/llava-v1.5-13b-lora-old")
    parser.add_argument("--model-base", type=str, default="/home/user6/LLaVA-Med/llava-med-v1.5-mistral-7b")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="/home/user6/LLaVA/vqa_test_dataset.json")
    parser.add_argument("--result-file", type=str, default="prediction_results.csv")
    parser.add_argument("--result-dir", type=str, default="result")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    args = parser.parse_args()

    eval_model(args)
