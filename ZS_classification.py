import os
import argparse
import pandas as pd
import torch
from PIL import Image
import numpy as np
import ast
import random
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from transformers import AutoTokenizer, AutoModel
from llava.conversation import Conversation

from utils.eval_help import binary_metrics

BASE_IMAGE_PATH = "/Dataset"
PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION = '/Dataframe/test/classification/'


def get_experiment_setting(experiment):
    if experiment == "ISIC":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "ISIC_test.csv",
                   "task": "classification",
                   "targets": {"Actinic Keratosis": 0, "Basal Cell Carcinoma": 1, "Benign Keratosis-like Lesions": 2,
                               "Dermatofibroma": 3, "Melanoma": 4, "Nevus": 5, "Squamous Cell Carcinoma": 6,
                               "Vascular Lesions": 7}}

    elif experiment == "MSKCC":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "MSKCC_test.csv",
                   "task": "classification",
                   "targets": {"AIMP": 0, "acrochordon": 1, "actinic keratosis": 2, "angiokeratoma": 3,
                               "atypical melanocytic proliferation": 4, "basal cell carcinoma": 5,
                               "cafe-au-lait macule": 6, "dermatofibroma": 7, "lentigo NOS": 8, "lentigo simplex": 9,
                               "lichenoid keratosis": 10, "melanoma": 11, "neurofibroma": 12, "nevus": 13, "other": 14,
                               "scar": 15, "seborrheic keratosis": 16, "solar lentigo": 17,
                               "squamous cell carcinoma": 18, "vascular lesion": 19, "verruca": 20}}

    elif experiment == "PAD":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "PAD_test.csv",
                   "task": "classification",
                   "targets": {"Actinic Keratosis": 0, "Basal Cell Carcinoma": 1, "Melanoma": 2, "Nevus": 3,
                               "Seborrheic Keratosis": 4, "Squamous Cell Carcinoma": 5}}

    elif experiment == "HIBA":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "HIBA_test.csv",
                   "task": "classification",
                   "targets": {"actinic keratosis": 0, "basal cell carcinoma": 1, "dermatofibroma": 2,
                               "lichenoid keratosis": 3, "melanoma": 4, "nevus": 5, "seborrheic keratosis": 6,
                               "solar lentigo": 7, "squamous cell carcinoma": 8, "vascular lesion": 9}}

    elif experiment == "HIBA_2class":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "HIBA_2class_test.csv",
                   "task": "classification",
                   "targets": {"benign": 0, "malignant": 1}}

    elif experiment == "BCN20000":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "BCN20000_test.csv",
                   "task": "classification",
                   "targets": {"actinic keratosis": 0, "basal cell carcinoma": 1, "dermatofibroma": 2, "melanoma": 3,
                               "melanoma metastasis": 4, "nevus": 5, "other": 6, "scar": 7, "seborrheic keratosis": 8,
                               "solar lentigo": 9, "squamous cell carcinoma": 10, "vascular lesion": 11}}

    elif experiment == "Fitzpatrick":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "Fitzpatrick_test.csv",
                   "task": "classification",
                   "targets": {"benign": 0, "malignant": 1, "non-neoplastic": 2}}

    elif experiment == "HAM10000":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "HAM10000_test.csv",
                   "task": "classification",
                   "targets": {"Actinic Keratoses": 0, "Basal Cell Carcinoma": 1, "Benign Keratosis": 2,
                               "Dermatofibroma": 3, "Melanoma": 4, "Nevus": 5, "Vascular lesions": 6}}

    elif experiment == "Dermnet":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "Dermnet_test.csv",
                   "task": "classification",
                   "targets": {"Acne and rosacea": 0,
                               "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": 1,
                               "Atopic dermatitis": 2, "Bullous disease": 3,
                               "Cellulitis Impetigo and other Bacterial Infections": 4, "Eczema": 5,
                               "Exanthems and Drug Eruptions": 6, "Hair loss  alopecia and other hair diseases": 7,
                               "Herpes hpv and other stds": 8, "Light Diseases and Disorders of Pigmentation": 9,
                               "Lupus and other Connective Tissue diseases": 10,
                               "Melanoma Skin Cancer Nevi and Moles": 11, "Nail Fungus and other Nail Disease": 12,
                               "Poison ivy  and other contact dermatitis": 13,
                               "Psoriasis pictures Lichen Planus and related diseases": 14,
                               "Scabies Lyme Disease and other Infestations and Bites": 15,
                               "Seborrheic Keratoses and other Benign Tumors": 16, "Systemic Disease": 17,
                               "Tinea Ringworm Candidiasis and other Fungal Infections": 18, "Urticaria Hives": 19,
                               "Vascular Tumors": 20, "Vasculitis": 21,
                               "Warts Molluscum and other Viral Infections": 22}}

    elif experiment == "Patch16":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "Patch16_test.csv",
                   "task": "classification",
                   "targets": {"nontumor skin chondraltissue": 0, "nontumor skin dermis": 1,
                               "nontumor skin elastosis": 2, "nontumor skin epidermis": 3,
                               "nontumor skin hairfollicle": 4, "nontumor skin muscle skeletal": 5,
                               "nontumor skin necrosis": 6, "nontumor skin nerves": 7,
                               "nontumor skin sebaceousglands": 8, "nontumor skin subcutis": 9,
                               "nontumor skin sweatglands": 10, "nontumor skin vessel": 11,
                               "tumor skin epithelial bcc": 12, "tumor skin epithelial sqcc": 13,
                               "tumor skin melanoma": 14, "tumor skin naevus": 15}}
    elif experiment == "Patch16_2class":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "Patch16_2class_test.csv",
                   "task": "classification",
                   "targets": {"nontumor": 0, "tumor": 1}}

    elif experiment == "DDI":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "DDI_test.csv",
                   "task": "classification",
                   "targets": {"Abrasions": 0, "Abscess": 1, "Acne Cystic": 2, "Acquired Digital Fibrokeratoma": 3,
                               "Acral Melanotic Macule": 4, "Acrochordon": 5, "Actinic Keratosis": 6,
                               "Angioleiomyoma": 7, "Angioma": 8, "Arteriovenous Hemangioma": 9,
                               "Atypical Spindle Cell Nevus of Reed": 10, "Basal Cell Carcinoma": 11,
                               "Benign Keratosis": 12, "Blastic Plasmacytoid Dendritic Cell Neoplasm": 13,
                               "Blue Nevus": 14, "Cellular Neurothekeoma": 15, "Chondroid Syringoma": 16,
                               "Clear Cell Acanthoma": 17, "Coccidioidomycosis": 18, "Condyloma Acuminatum": 19,
                               "Dermatofibroma": 20, "Dermatomyositis": 21, "Dysplastic Nevus": 22,
                               "Eccrine Poroma": 23, "Eczema": 24, "Epidermal Cyst": 25, "Epidermal Nevus": 26,
                               "Fibrous Papule": 27, "Focal Acral Hyperkeratosis": 28, "Folliculitis": 29,
                               "Foreign Body Granuloma": 30, "Glomangioma": 31, "Graft vs Host Disease": 32,
                               "Hematoma": 33, "Hyperpigmentation": 34, "Inverted Follicular Keratosis": 35,
                               "Kaposi Sarcoma": 36, "Keloid": 37, "Leukemia Cutis": 38, "Lichenoid Keratosis": 39,
                               "Lipoma": 40, "Lymphocytic Infiltrations": 41, "Melanocytic Nevi": 42, "Melanoma": 43,
                               "Metastatic Carcinoma": 44, "Molluscum Contagiosum": 45, "Morphea": 46,
                               "Mycosis Fungoides": 47, "Neurofibroma": 48, "Neuroma": 49,
                               "Nevus Lipomatosus Superficialis": 50, "Onychomycosis": 51,
                               "Pigmented Spindle Cell Nevus of Reed": 52, "Prurigo Nodularis": 53,
                               "Pyogenic Granuloma": 54, "Reactive Lymphoid Hyperplasia": 55, "Scar": 56,
                               "Sebaceous Carcinoma": 57, "Seborrheic Keratosis": 58, "Solar Lentigo": 59,
                               "Squamous Cell Carcinoma": 60, "Subcutaneous T-cell Lymphoma": 61,
                               "Syringocystadenoma Papilliferum": 62, "Tinea Pedis": 63, "Trichilemmoma": 64,
                               "Trichofolliculoma": 65, "Ulcerations and Physical Injuries": 66, "Verruca Vulgaris": 67,
                               "Verruciform Xanthoma": 68, "Wart": 69, "Xanthogranuloma": 70}}

    elif experiment == "DDI_2class":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "DDI_2class_test.csv",
                   "task": "classification",
                   "targets": {"Non-Malignant": 0, "Malignant": 1}}

    else:
        setting = None
        print("Experiment not prepared...")
    return setting


def process_categories(categories):
    """
    处理每行类别数据，将其转化为分类选项，随机选择一个类别。
    """
    # 将类别字段转换为列表
    category_list = eval(categories)  # 使用eval将字符串形式的列表转为实际的列表

    if len(category_list) > 1:
        category = random.choice(category_list)
    else:
        category = category_list[0]

    return category

def eval_model(args):
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.expanduser(args.model_path)
    model_name = args.model_name
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=device)
    model.eval()

    # 从文件获取实验设置
    setting = get_experiment_setting(args.experiment)
    if not setting:
        raise ValueError(f"Experiment '{args.experiment}' settings are not found.")

    os.makedirs(args.result_path, exist_ok=True)
    df = pd.read_csv(setting["dataframe"])
    predictions = []
    ground_truths = []
    result_file = os.path.join(args.result_path, f"{args.experiment}_predictions.csv")
    # 确保结果文件存在
    if not os.path.exists(result_file):
        result_df = pd.DataFrame(columns=["image", "question", "predicted_answer", "ground_truth", "predicted_label", "ground_truth_label"])
        result_df.to_csv(result_file, index=False)

    # 获取类别名称
    possible_diseases = list(setting["targets"].keys())  # 从targets中提取类别名
    label_set = list(setting["targets"].values())
    print(f"Possible diseases: {possible_diseases}", f"Label set: {label_set}")
    # 遍历数据集
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        image_file = row["image"]
        # ground_truth = row["categories"]  # 真正的诊断标签，格式类似 'non-malignant' 或 'malignant'
        ground_truth = process_categories(row["categories"])
        print(f"Image: {image_file}, Ground truth: {ground_truth}")
        true_label = setting["targets"][ground_truth]
        question = f"This is a skin lesion image. From the following categories: {', '.join(possible_diseases)}, which one is the diagnosis?"
        print(f"Question: {question}")
        image_path = os.path.join(args.image_folder, image_file)

        ############## 使用模型进行推理，得到预测结果 ##############

        if model.config.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
            print(question)
        else:
            question = DEFAULT_IMAGE_TOKEN + '\n' + question
            # print(question)
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda().to(device)

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_size = image.size
        # print(image_size)
        image_tensor = process_images([image], image_processor, model.config)[0].to(device)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                min_new_tokens=1,
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id,
                # image_sizes=[image_size],
                use_cache=True)

        predicted_answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(predicted_answer)
        # 提取诊断类别并与目标匹配
        predicted_diagnosis = None
        # 如果模型输出actinic keratosis，但实际诊断是Actinic Keratosis，也应该被视为匹配Actinic Keratoses，先进行大小写转换
        predicted_answer = predicted_answer.replace("actinic keratosis", "actinic keratoses")
        for disease in possible_diseases:
            if disease.lower() in predicted_answer.lower():
                predicted_diagnosis = disease
                break

        if predicted_diagnosis:
            predicted_label = setting["targets"][predicted_diagnosis]
        else:
            predicted_label = -1  # 如果无法匹配诊断，则设置为-1

        # 将预测值和真实值保存
        predictions.append(predicted_label)
        ground_truths.append(setting["targets"][ground_truth])

        # 保存当前样本的结果到CSV
        new_data = pd.DataFrame({
            "image": [image_file],
            "question": [question],
            "output": [predicted_answer],
            "predicted_answer": [predicted_diagnosis],
            "ground_truth": [ground_truth],
            "predicted_label": [predicted_label],
            "ground_truth_label": [true_label]
        })
        new_data.to_csv(result_file, mode='a', header=False, index=False)

    res = binary_metrics(ground_truths, predictions, label_set)
    df = pd.DataFrame([res])
    path = os.path.join(args.result_path, f"{args.experiment}_results.csv")
    df.to_csv(path, index=False)
    print(f"Predictions saved to {result_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnosis Classification via Generation")
    parser.add_argument('--experiment', default='PAD', help="Experiment name (e.g., PAD, DDI_2class, etc.)")
    parser.add_argument('--model_path', default="/merge/SkinVL_PubMM", help="Path to model weights")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="LlavaMistralForCausalLM")
    parser.add_argument("--load-8bit", type=bool, default=False, help="Load model with 8-bit precision")
    parser.add_argument("--load-4bit", type=bool, default=False, help="Load model with 4-bit precision")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for beam search")
    parser.add_argument("--image-folder", type=str, default="Dataset", help="Folder containing images")
    # parser.add_argument("--dataset-file", type=str, default="/home/user6/LLaVA/vqa_test_dataset.csv", help="Test dataset file (CSV)")
    parser.add_argument("--conv_mode", type=str, default="mistral_instruct", help="Conversation mode for prompt templates")
    parser.add_argument('--result-path', default='result/zeroshot_class/DermMM_9pubCHOICE', type=str,
                        help="File to save predictions")
    args = parser.parse_args()
    eval_model(args)