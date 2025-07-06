"""
获取迁移实验的数据集目录、迁移任务、类别
"""

from transferability.local_data.constants import *


def get_experiment_setting(experiment):

    # Transferability for classification
    # if experiment == "HAM10000_test":
    #     setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "HAM10000_test.csv",
    #                "task": "classification",
    #                "targets": {"Nevus": 0, "Benign Keratosis": 1, "Melanoma": 2, "Actinic Keratoses": 3,
    #                            "Basal Cell Carcinoma": 4, "Vascular lesions": 5, "Dermatofibroma": 6}}
    # elif experiment == "BCN20000_test":
    #     setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "BCN20000_test.csv",
    #                "task": "classification",
    #                "targets": {"normal": 0, "pathologic myopia": 1, "cataract": 2}}
    if experiment == "Dermnet_test":
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

    elif experiment == "Fitzpatrick_test":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "Fitzpatrick_test.csv",
                   "task": "classification",
                   "targets": {"benign": 0, "malignant": 1, "non-neoplastic": 2}}

    elif experiment == "DDI_test":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "DDI_test.csv",
                   "task": "classification",
                   "targets": {"Abrasions": 0, "Acral Melanotic Macule": 1, "Acrochordon": 2, "Angioma": 3,
                               "Arteriovenous Hemangioma": 4, "Atypical Spindle Cell Nevus of Reed": 5,
                               "Basal Cell Carcinoma": 6, "Blue Nevus": 7, "Coccidioidomycosis": 8, "Dermatofibroma": 9,
                               "Dysplastic Nevus": 10, "Epidermal Cyst": 11, "Graft vs Host Disease": 12,
                               "Inverted Follicular Keratosis": 13, "Lipoma": 14, "Melanocytic Nevi": 15,
                               "Melanoma": 16, "Metastatic Carcinoma": 17, "Molluscum Contagiosum": 18,
                               "Mycosis Fungoides": 19, "Neurofibroma": 20, "Nevus Lipomatosus Superficialis": 21,
                               "Pyogenic Granuloma": 22, "Scar": 23, "Sebaceous Carcinoma": 24,
                               "Seborrheic Keratosis": 25, "Squamous Cell Carcinoma": 26, "Trichofolliculoma": 27,
                               "Ulcerations and Physical Injuries": 28, "Verruca Vulgaris": 29, "Wart": 30,
                               "Xanthogranuloma": 31}}

    elif experiment == "Patch16_test":
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

    elif experiment == "HIBA_test":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "HIBA_test.csv",
                   "task": "classification",
                   "targets": {"actinic keratosis": 0, "basal cell carcinoma": 1, "dermatofibroma": 2, "melanoma": 3,
                               "nevus": 4, "seborrheic keratosis": 5, "solar lentigo": 6, "squamous cell carcinoma": 7,
                               "vascular lesion": 8}}

    elif experiment == "HAM10000_test":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "HAM10000_test.csv",
                   "task": "classification",
                   "targets": {"Actinic Keratoses": 0, "Basal Cell Carcinoma": 1, "Benign Keratosis": 2,
                               "Dermatofibroma": 3, "Melanoma": 4, "Nevus": 5, "Vascular lesions": 6}}

    elif experiment == "MSKCC_test":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "MSKCC_test.csv",
                   "task": "classification",
                   "targets": {"AIMP": 0, "acrochordon": 1, "actinic keratosis": 2, "angiokeratoma": 3,
                               "atypical melanocytic proliferation": 4, "basal cell carcinoma": 5, "dermatofibroma": 6,
                               "lentigo NOS": 7, "lentigo simplex": 8, "lichenoid keratosis": 9, "melanoma": 10,
                               "neurofibroma": 11, "nevus": 12, "seborrheic keratosis": 13, "solar lentigo": 14,
                               "squamous cell carcinoma": 15, "verruca": 16}}

    elif experiment == "BCN20000_test":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "BCN20000_test.csv",
                   "task": "classification",
                   "targets": {"actinic keratosis": 0, "basal cell carcinoma": 1, "dermatofibroma": 2, "melanoma": 3,
                               "melanoma metastasis": 4, "nevus": 5, "other": 6, "scar": 7, "seborrheic keratosis": 8,
                               "solar lentigo": 9, "squamous cell carcinoma": 10, "vascular lesion": 11}}

    elif experiment == "ISIC_test":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "ISIC_test.csv",
                   "task": "classification",
                   "targets": {"Actinic Keratosis": 0, "Basal Cell Carcinoma": 1, "Benign Keratosis-like Lesions": 2,
                               "Dermatofibroma": 3, "Melanoma": 4, "Nevus": 5, "Squamous Cell Carcinoma": 6,
                               "Vascular Lesions": 7}}

    elif experiment == "PAD_test":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "PAD_test.csv",
                   "task": "classification",
                   "targets": {"Actinic Keratosis": 0, "Basal Cell Carcinoma": 1, "Melanoma": 2, "Nevus": 3,
                               "Seborrheic Keratosis": 4, "Squamous Cell Carcinoma": 5}}


    # elif experiment == "02_MESSIDOR":
    #     setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "02_MESSIDOR.csv",
    #                "task": "classification",
    #                "targets": {"no diabetic retinopathy": 0, "mild diabetic retinopathy": 1,
    #                            "moderate diabetic retinopathy": 2, "severe diabetic retinopathy": 3,
    #                            "proliferative diabetic retinopathy": 4}}
    # elif experiment == "AMD":
    #     setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "AMD.csv",
    #                "task": "classification",
    #                "targets": {"age related macular degeneration": 0, "normal": 1}}
    # elif experiment == "TAOP":
    #     setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "TAOP_train.csv",
    #                "task": "classification",
    #                "targets": {"0c":0, "1c":1, "2c":2, "3c":3, "4c":4}}
    # elif experiment == "25_REFUGE":
    #     setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "25_REFUGE.csv",
    #                "task": "classification",
    #                "targets": {"no glaucoma": 0, "glaucoma": 1}}
    # elif experiment == "13_FIVES":
    #     setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "13_FIVES.csv",
    #                "task": "classification",
    #                "targets": {"normal": 0, "age related macular degeneration": 1, "diabetic retinopathy": 2,
    #                            "glaucoma": 3}}
    # elif experiment == "08_ODIR200x3":
    #     setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "08_ODIR200x3.csv",
    #                "task": "classification",
    #                "targets": {"normal": 0, "pathologic myopia": 1, "cataract": 2}}
    # elif experiment == "36_ACRIMA":
    #     setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "36_ACRIMA.csv",
    #                "task": "classification",
    #                "targets": {"no glaucoma": 0, "glaucoma": 1}}
    # elif experiment == "CAT_MYA_2":
    #     setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "CAT_MYA_2.csv",
    #                "task": "classification",
    #                "targets": {"normal": 0, "pathologic myopia": 1, "cataract": 2}}
    # elif experiment == "05_20x3":
    #     setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "05_20x3.csv",
    #                "task": "classification",
    #                "targets": {"normal": 0, "retinitis pigmentosa": 1, "macular hole": 2}}
    # elif experiment == "MHL_RP_2":
    #     setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "MHL_RP_2.csv",
    #                "task": "classification",
    #                "targets": {"normal": 0, "retinitis pigmentosa": 1, "macular hole": 2}}
    # elif experiment == "37_DeepDRiD_train_eval":
    #     setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "37_DeepDRiD_train_eval.csv",
    #                "task": "classification",
    #                "targets": {"no diabetic retinopathy": 0, "mild diabetic retinopathy": 1,
    #                            "moderate diabetic retinopathy": 2, "severe diabetic retinopathy": 3,
    #                            "proliferative diabetic retinopathy": 4}}
    # elif experiment == "37_DeepDRiD_test":
    #     setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "37_DeepDRiD_test.csv",
    #                "task": "classification",
    #                "targets": {"no diabetic retinopathy": 0, "mild diabetic retinopathy": 1,
    #                            "moderate diabetic retinopathy": 2, "severe diabetic retinopathy": 3,
    #                            "proliferative diabetic retinopathy": 4}}
    #
    # elif experiment == "CGI_HRDC_Task1":
    #     setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "CGI_HRDC_Task1.csv",
    #                "task": "classification",
    #                "targets": {"no hypertensive": 0, "hypertensive": 1}}
    # elif experiment == "CGI_HRDC_Task2":
    #     setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "CGI_HRDC_Task2.csv",
    #                "task": "classification",
    #                "targets": {"no hypertensive retinopathy": 0, "hypertensive retinopathy": 1}}

    else:
        setting = None
        print("Experiment not prepared...")

    return setting
