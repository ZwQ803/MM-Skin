# MM-Skin: Enhancing Dermatology Vision-Language Model with an Image-Text Dataset Derived from Textbooks

Paper[PDF] Dataset[[Google Drive](https://drive.google.com/drive/folders/1gAQOpJjzldpqegIcZcKX5_2Wau54taJ4?usp=sharing)] Code[[Github](https://github.com/ZwQ803/MM-Skin/tree/main)]

we propose MM-Skin, a large-scale multimodal dermatology dataset that encompasses 3 imaging modalities, including clinical, dermoscopic, and pathological and nearly 10k high-quality image-text pairs collected from professional textbooks and over 27k vision question answering (VQA) samples.

In addition, we developed SkinVL, a dermatology-specific VLM, and conducted comprehensive benchmark evaluations of SkinVL on VQA, supervised fine-tuning (SFT), and zero-shot classification tasks.

Code and model weights are coming soon.


# MM-Skin: Enhancing Dermatology Vision-Language Model with an Image-Text Dataset Derived from Textbooks

Paper[PDF] Dataset[[Google Drive](https://drive.google.com/drive/folders/1gAQOpJjzldpqegIcZcKX5_2Wau54taJ4?usp=sharing)] Code[[Github](https://github.com/ZwQ803/MM-Skin/tree/main)]

we propose MM-Skin, a large-scale multimodal dermatology dataset that encompasses 3 imaging modalities, including clinical, dermoscopic, and pathological and nearly 10k high-quality image-text pairs collected from professional textbooks and over 27k vision question answering (VQA) samples.

In addition, we developed SkinVL, a dermatology-specific VLM, and conducted comprehensive benchmark evaluations of SkinVL on VQA, supervised fine-tuning (SFT), and zero-shot classification tasks.



## Quick Start

1、Environment

First, clone the repo and cd into the directory:

```
git clone https://github.com/ZwQ803/MM-Skin.git
cd MM-Skin
```

Then create a conda env and install the dependencies:

```
conda create -n mmskin python=3.10 -y
conda activate mmskin
pip install -r requirements.txt
```

2、Download MM-SkinVL Pre-trained Weights

| Model Name   | Link |
| ------------ | ---- |
| SkinVL-MM    | Link |
| SkinVL-Pub   | Link |
| SkinVL-PubMM | Link |



## Download Pre-training Datasets

| Dataset        | Modality                        | Link                                                         |
| -------------- | ------------------------------- | ------------------------------------------------------------ |
| SCIN           | Clinical                        | [Link](https://console.cloud.google.com/storage/browser/dx-scin-public-data?inv=1&invt=Abw9Eg) |
| DDI            | Clinical                        | [Link](https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965) |
| Fitzpatrick17k | Clinical                        | [Link](https://github.com/mattgroh/fitzpatrick17k)           |
| PAD            | Clinical                        | [Link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7479321/) |
| Dermnet        | Clinical                        | [Link](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) |
| HAM10000       | Dermoscopy                      | [Link](https://challenge.isic-archive.com/data/#2018)        |
| ISIC2019       | Dermoscopy                      | [Link](https://api.isic-archive.com/collections/65/)         |
| BCN20000       | Dermoscopy                      | [Link](https://api.isic-archive.com/collections/249/)        |
| HIBA           | Dermoscopy                      | [Link](https://api.isic-archive.com/collections/175/)        |
| MSKCC          | Dermoscopy                      | [Link](https://api.isic-archive.com/collections/163/)        |
| Patch16        | Pathology                       | [Link](https://heidata.uni-heidelberg.de/dataset.xhtml?persistentId=doi:10.11588/data/7QCR8S) |
| MM-Skin        | Clinical, Dermoscopy, Pathology | [Link](https://drive.google.com/drive/folders/1gAQOpJjzldpqegIcZcKX5_2Wau54taJ4?usp=sharing) |



## Training

To train the model using LoRA, run `finetune_lora.sh` with pre-trained **LLaVA-Med** weights (available [here](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b)).  
Update `LLAVA_MED_WEIGHT_PATH` in the script to your local path, and replace `PRETRAIN_DATAFRAME` with the processed JSON training file.  
We provide training JSONs for **SkinVL-MM**, **SkinVL-Pub**, and **SkinVL-PubMM** at: `/Dataframe/Pretrain`.

After training, merge the LoRA weights with the base model:

```
python merge_lora_weights.py \
    --model-path /path/to/lora_model \
    --model-base /path/to/base_model/llava-med-v1.5-mistral-7b \
    --save-model-path /path/to/merge_model
```

You can also directly use our provided merged models by placing them in the `/merge` directory.



## Evaluation

**1、 VQA Evaluation。**

To evaluate **SkinVL-MM**, **SkinVL-Pub**, and **SkinVL-PubMM**, run:

```
python VQA_test.py --model-path MERGED_SKINVL_MODEL
```

Replace `caption file` and `image folder` in the script with your dataset paths.

**2. Supervised Fine-Tuning (SFT) Classification**

Run `SFT_classify_test.sh` for supervised classification. Replace all paths with your local files. Preprocessed data for reproducing our results can be found in `/Dataframe/test/classification`.

**3. Zero-Shot Classification**

Run `ZS_classify_test.sh` to perform zero-shot classification.



## Data Collection and Statistics

The 15 professional dermatology textbooks are:

- [Diagnostic Dermoscopy: The Illustrated Guide, Second Edition](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118932063)
- [Skin Lymphoma: The Illustrated](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118492505)
- [Skin Disease: Diagnosis and Treatment, Fourth Edition](https://www.clinicalkey.com/#!/browse/book/3-s2.0-C20130186114)
- [Imported Skin Diseases](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118472620)
- [Shimizu's Dermatology](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119099086)
- [Skin Cancer: Recognition and Management](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470696347)
- [Clinical Dermatology](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118938164)
- [Andrews' Diseases of the Skin, Fourteenth Edition](https://www.clinicalkey.com/#!/browse/book/3-s2.0-C20210009858)
- [Diseases of the Liver and Biliary System in Children](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119046936)
- [McKee's Pathology of the Skin, Fifth Edition](https://www.clinicalkey.com/#!/browse/book/3-s2.0-C20151017471)
- [Harper's Textbook of Pediatric Dermatology](https://onlinelibrary.wiley.com/doi/book/10.1002/9781444345384)
- [Skin Infections](https://www.cambridge.org/core/books/skin-infections/288086E3FEE42641212A8A2820280B37#)
- [Advances in Integrative Dermatology](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119476009)
- [Cancer of the Skin, 2nd Edition](https://www.asia.elsevierhealth.com/cancer-of-the-skin-e-book-9781437736144.html)
- [Rook's Textbook of Dermatology](https://onlinelibrary.wiley.com/doi/book/10.1002/9781444317633)



MM-Skin contains **11,039 dermatology images** with expert descriptions across three modalities. It provides three subsets:  

\- **MM-Skin-C** (Captions)  

\- **MM-Skin-O** (Open-ended VQA)  

\- **MM-Skin-D** (Demographics)



Data Collection Process

1. **Image-Text Extraction**: From 15 dermatology textbooks using OCR and Adobe API.  

2. **Alignment**: Match images with captions.  

3. **Modality Classification**: Feature-based classification (color, texture) with manual verification.  

4. **Text Cleaning**: Extract age and gender info.  

5. **Filtering**: Remove sensitive or annotated images.

 [Fig1.pdf](../1project/2MultiSkinModel/fig/Fig1.pdf) 
