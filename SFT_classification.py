import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import argparse
# import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from llava.model.builder import load_pretrained_model
from torchvision.transforms import ToTensor
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
import ast
from tqdm import tqdm
import random
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from utils.eval_help import binary_metrics, prob_metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb
from torch.utils.data import SubsetRandomSampler





# import cuml
# from cuml.decomposition import PCA as cuPCA
# import cupy as cp


from transferability.local_data.constants import *
# from transferability.local_data.experiments import get_experiment_setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 设定基本路径
BASE_IMAGE_PATH = "/mnc/zwq/Dataset/SkinModel_resized512"
TRAIN_PATH = "/home/user6/SkinModel/Dataframe/Pretrain"
TEST_PATH = "/home/user6/SkinModel/Dataframe/test/classification"


"""
获取迁移实验的数据集目录、迁移任务、类别
"""

from transferability.local_data.constants import *


def get_experiment_setting(experiment):
    if experiment == "ISIC":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "ISIC_train.csv",
                   "task": "classification",
                   "targets": {"Actinic Keratosis": 0, "Basal Cell Carcinoma": 1, "Benign Keratosis-like Lesions": 2,
                               "Dermatofibroma": 3, "Melanoma": 4, "Nevus": 5, "Squamous Cell Carcinoma": 6,
                               "Vascular Lesions": 7}}

    elif experiment == "MSKCC":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "MSKCC_train.csv",
                   "task": "classification",
                   "targets": {"AIMP": 0, "acrochordon": 1, "actinic keratosis": 2, "angiokeratoma": 3,
                               "atypical melanocytic proliferation": 4, "basal cell carcinoma": 5,
                               "cafe-au-lait macule": 6, "dermatofibroma": 7, "lentigo NOS": 8, "lentigo simplex": 9,
                               "lichenoid keratosis": 10, "melanoma": 11, "neurofibroma": 12, "nevus": 13, "other": 14,
                               "scar": 15, "seborrheic keratosis": 16, "solar lentigo": 17,
                               "squamous cell carcinoma": 18, "vascular lesion": 19, "verruca": 20}}

    elif experiment == "PAD":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "PAD_train.csv",
                   "task": "classification",
                   "targets": {"Actinic Keratosis": 0, "Basal Cell Carcinoma": 1, "Melanoma": 2, "Nevus": 3,
                               "Seborrheic Keratosis": 4, "Squamous Cell Carcinoma": 5}}

    elif experiment == "HIBA":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "HIBA_train.csv",
                   "task": "classification",
                   "targets": {"actinic keratosis": 0, "basal cell carcinoma": 1, "dermatofibroma": 2,
                               "lichenoid keratosis": 3, "melanoma": 4, "nevus": 5, "seborrheic keratosis": 6,
                               "solar lentigo": 7, "squamous cell carcinoma": 8, "vascular lesion": 9}}

    elif experiment == "HIBA_2class":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "HIBA_2class_train.csv",
                   "task": "classification",
                   "targets": {"benign": 0, "malignant": 1}}

    elif experiment == "BCN20000":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "BCN20000_train.csv",
                   "task": "classification",
                   "targets": {"actinic keratosis": 0, "basal cell carcinoma": 1, "dermatofibroma": 2, "melanoma": 3,
                               "melanoma metastasis": 4, "nevus": 5, "other": 6, "scar": 7, "seborrheic keratosis": 8,
                               "solar lentigo": 9, "squamous cell carcinoma": 10, "vascular lesion": 11}}

    elif experiment == "Fitzpatrick":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "Fitzpatrick_train.csv",
                   "task": "classification",
                   "targets": {"benign": 0, "malignant": 1, "non-neoplastic": 2}}

    elif experiment == "HAM10000":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "HAM10000_train.csv",
                   "task": "classification",
                   "targets": {"Actinic Keratoses": 0, "Basal Cell Carcinoma": 1, "Benign Keratosis": 2,
                               "Dermatofibroma": 3, "Melanoma": 4, "Nevus": 5, "Vascular lesions": 6}}

    elif experiment == "Dermnet":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "Dermnet_train.csv",
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
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "Patch16_train.csv",
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
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "Patch16_2class_train.csv",
                   "task": "classification",
                   "targets": {"nontumor": 0, "tumor": 1}}

    elif experiment == "DDI":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "DDI_train.csv",
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
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "DDI_2class_train.csv",
                   "task": "classification",
                   "targets": {"Non-Malignant": 0, "Malignant": 1}}

    else:
        setting = None
        print("Experiment not prepared...")
    return setting



# 自定义数据集
class SkinDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(BASE_IMAGE_PATH, self.data.iloc[idx, 0])  # 图片路径
        label = self.data.iloc[idx, 2]  # 类别

        return img_path, label

# 加载数据集
def get_dataloaders(experiment, batch_size=1, val_split=0.2, num_workers=2):
    setting = get_experiment_setting(experiment)
    if not setting:
        raise ValueError(f"Experiment {experiment} not found in settings!")
    label_set = setting['targets'].values()
    train_csv = os.path.join(TRAIN_PATH, f"{experiment}_train.csv")
    test_csv = os.path.join(TEST_PATH, f"{experiment}_test.csv")
    # 读取数据
    train_dataset = SkinDataset(train_csv)
    test_dataset = SkinDataset(test_csv)
    # 如果存在val_csv，使用val_csv，否则从训练集划分
    val_csv = os.path.join(TEST_PATH, f"{experiment}_val.csv")
    if os.path.exists(val_csv):
        val_dataset = SkinDataset(val_csv)
    else:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    if experiment == "Patch16_2class":
        # 随机选取20%的训练数据和20%的验证数据
        train_size = int(len(train_dataset) * 0.1)
        val_size = int(len(val_dataset) * 0.1)
        print("train_size:", train_size, "val_size:", val_size)
        train_dataset, _ = random_split(train_dataset, [train_size, len(train_dataset) - train_size])
        val_dataset, _ = random_split(val_dataset, [val_size, len(val_dataset) - val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader,label_set


def get_representations(model, loader, device,image_processor, tokenizer, targets, mode='train'):
    ys, zs = [], []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            image_paths, labels = batch  # DataLoader 传入的是批量的路径和标签

            images = [Image.open(img_path).convert('RGB') for img_path in image_paths]  # **避免逐个读取**
            images_tensor = process_images(images, image_processor, model.config).half().to(device)  # **一次性处理整个 batch**

            batch_features = model.encode_images(images_tensor)

            batch_labels = []
            for label in labels:
                # print("label:", label)
                label_list = ast.literal_eval(label)  # 解析成列表
                label_text = [random.choice(label_list)] if len(label_list) > 1 else label_list  # 保持方括号格式
                batch_labels.append([targets[label_item] for label_item in label_text])  # 转换为数值

            zs.append(batch_features.cpu().numpy())  # 存入特征
            ys.extend(batch_labels)  # 存入 labels

            torch.cuda.ipc_collect()  # **释放 GPU 内存**


        # print("zs:", zs.shape, "ys:", ys.shape)

    return np.concatenate(zs, axis=0),  np.concatenate(ys, axis=0)


def apply_pca(X, n_components=100):
    svd = TruncatedSVD(n_components=n_components, algorithm='randomized', random_state=42)
    return svd.fit_transform(X)
    # return pca.fit_transform(X)

def pool_features(features, mode='average'):
    # features: (n_samples, seq_len, embedding_dim)
    if mode == 'average':
        return np.mean(features, axis=1)  # 结果形状: (n_samples, embedding_dim)
    elif mode == 'max':
        return np.max(features, axis=1)
    else:
        raise ValueError("Unsupported pooling mode")

def fit_model(train_X, train_Y, val_X, val_Y, test_X, test_Y, label_set, model_type='lr'):
    label_set = list(label_set)
    if model_type == 'lr':
        pipe = Pipeline(steps=[
            ('scaler', StandardScaler()),  # 加入标准化步骤
            ('model', LogisticRegression(random_state=42, n_jobs=-1, max_iter=1000))
        ])
        param_grid = {
            'model__C': 10**np.linspace(-5, 1, 10)
        }
    elif model_type == 'rf':
        pipe = Pipeline(steps=[
            ('model', RandomForestClassifier(random_state=42, n_jobs=-1))
            # ('model', XGBClassifier(random_state=42, n_jobs=-1))
        ])
        param_grid = {
            'model__max_depth': list(range(1, 7))
        }
    else:
        raise NotImplementedError
    print("pooling features")
    train_X = pool_features(train_X, mode='average')
    val_X = pool_features(val_X, mode='average')
    test_X = pool_features(test_X, mode='average')
    print("pooled features")
    print(f"train_X.shape: {train_X.shape}, val_X.shape: {val_X.shape}")
    print(f"train_Y.shape: {train_Y.shape}, val_Y.shape: {val_Y.shape}")
    # pds = PredefinedSplit(test_fold=np.concatenate([np.ones((len(train_X),))*-1, np.zeros((len(val_X),))]))
    cv = StratifiedKFold(n_splits=5)

    # 在交叉验证之前检查每个折叠的数据
    for train_index, val_index in cv.split(train_X, train_Y):
        train_fold_Y = train_Y[train_index]
        if len(np.unique(train_fold_Y)) == 1:
            print("Skipping fold with only one class:", np.unique(train_fold_Y))
            continue

    print("start grid search")
    cv_lr = (GridSearchCV(pipe, param_grid, refit=False, cv=cv, scoring='roc_auc_ovr', verbose=10, n_jobs=5).fit(
        np.concatenate((train_X, val_X)), np.concatenate((train_Y, val_Y))))
    print("grid search done")
    pipe = clone(
        clone(pipe).set_params(**cv_lr.best_params_)
    )
    pipe = pipe.fit(train_X, train_Y)

    label_set = np.sort(np.unique(train_Y))
    print("strat predict")
    res = {}
    for sset, X, Y in zip(['va', 'te'], [val_X, test_X], [val_Y, test_Y]):
        preds = pipe.predict_proba(X)
        if len(label_set) == 2:
            preds = preds[:, 1]
            preds_rounded = preds >= 0.5
            preds_rounded = preds_rounded.astype(int)
        else:
            preds_rounded = preds.argmax(1)

        print("preds:", preds, "preds_rounded:", preds_rounded, "Y:", Y, "label_set:", label_set)
        res[sset] = binary_metrics(Y, preds_rounded, label_set=label_set, return_arrays=True)
        prob_mets = prob_metrics(Y, preds, label_set=label_set, return_arrays=True)
        prob_mets['pred_probs'] = prob_mets['preds']
        del prob_mets['targets']
        res[sset] = {
            **res[sset],
            **prob_mets
        }
    print("finished")

    return res


def eval_lin_attr_pred(train_zs, train_ys, val_zs, val_ys, test_zs, test_ys, label_set):
    res = {}
    num_classes = np.max(train_ys) + 1  # 假设类标签从0开始，且连续
    for model_type in ['lr']:
        print(f"Training {model_type} models...")
        res[f'{model_type}_uncond'] = fit_model(
            train_zs, train_ys, val_zs, val_ys, test_zs, test_ys, label_set, model_type=model_type)

        # # 处理所有类别的条件模型
        # for cls in range(num_classes):
        #     # Check if there are any samples for the current class in each dataset
        #     if np.any(train_ys == cls) and np.any(val_ys == cls) and np.any(test_ys == cls):
        #         res[f'{model_type}_cond_{cls}'] = fit_model(
        #             train_zs[train_ys == cls], train_ys[train_ys == cls],
        #             val_zs[val_ys == cls], val_ys[val_ys == cls],
        #             test_zs[test_ys == cls], test_ys[test_ys == cls], label_set,
        #             model_type=model_type)
        #     else:
        #         print(f"Skipping model {model_type} for class {cls} due to insufficient data.")

    return res

def results_to_dataframe(results):
    # 准备列表来存储数据
    data = []
    for key, value in results.items():
        for sub_key, sub_value in value.items():
            if isinstance(sub_value, dict):
                row = sub_value
                row['model'] = key
                row['subset'] = sub_key
                data.append(row)
            else:
                row = {'metric': sub_key, 'value': sub_value, 'model': key, 'subset': 'general'}
                data.append(row)
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description='Test the pretrained models and save the results.')
    # parser.add_argument('--model_files', nargs='+', help='Paths to the pretrained model files.')
    parser.add_argument('--result_dir', default='/result/classification',type=str, help='Directory to save the results.')
    parser.add_argument('--experiment', default='PAD',
                        help='02_MESSIDOR - 13_FIVES - 25_REFUGE - 08_ODIR200x3 - 05_20x3 - AMD - TAOP')            # 实验使用的数据集  data set used in the experiment
    parser.add_argument('--model_path', default="/merge/DermMM_train")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="LlavaMistralForCausalLM")
    parser.add_argument("--load-8bit", type=bool, default=False, help="Load model with 8-bit precision")
    parser.add_argument("--load-4bit", type=bool, default=False, help="Load model with 4-bit precision")

    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    # 设备配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Processing model:")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, args.model_name, args.load_8bit, args.load_4bit, device=device)
    model.eval()

    train_loader, val_loader, test_loader, label_set = get_dataloaders(args.experiment, batch_size=64)


    # 特征提取
    train_zs, train_ys = get_representations(model, train_loader, device, image_processor, tokenizer, get_experiment_setting(args.experiment)['targets'])
    test_zs, test_ys = get_representations(model, test_loader, device, image_processor, tokenizer, get_experiment_setting(args.experiment)['targets'], mode='test')
    val_zs, val_ys = get_representations(model, val_loader, device, image_processor, tokenizer, get_experiment_setting(args.experiment)['targets'], mode='val')


    # 计算指标
    lin_eval_metrics = eval_lin_attr_pred(train_zs, train_ys, val_zs, val_ys, test_zs, test_ys, label_set)

    # 结果转换为 DataFrame
    df_metrics = results_to_dataframe(lin_eval_metrics)

    # 保存到 CSV 文件
    output_path = os.path.join(args.result_dir, f'{args.experiment}_classify_evaluation.csv')
    df_metrics.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
