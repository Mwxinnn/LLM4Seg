# Pre-Trained LLM is a Semantic-Aware and Generalizable Segmentation Booster

Official pytorch code base for "Pre-Trained LLM is a Semantic-Aware and Generalizable Segmentation Booster"



## Introduction
With the advancement of Large Language Model (LLM) for natural language processing, this paper presents an intriguing finding: a frozen pre-trained LLM layer can process visual tokens for medical image segmentation tasks. Specifically, we propose a simple hybrid  structure (LLM4Seg) that integrates a pre-trained, frozen LLM layer within the CNN encoder-decoder framework. Surprisingly, this design improves segmentation performance with a minimal increase in trainable parameters across various modalities, including ultrasound, dermoscopy, polypscopy, and CT scans. Our in-depth analysis reveals the potential of transferring LLM's semantic awareness to enhance segmentation tasks, offering both improved global understanding and better local modeling capabilities. The improvement proves robust across different LLMs, validated using LLaMA and DeepSeek.



## Get Start



### Datasets

Please put the [BUSI](https://www.kaggle.com/aryashah2k/breast-ultrasound-images-dataset) dataset or your own dataset as the following architecture. 
```
└── LLM4Seg
    ├── data
        ├── busi
            ├── images
            |   ├── benign (10).png
            │   ├── malignant (17).png
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── benign (10).png
                |   ├── malignant (17).png
                |   ├── ...
        ├── your dataset
            ├── images
            |   ├── 0a7e06.png
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── ...
    ├── dataloader
    ├── network
    ├── utils
    ├── main.py
    └── split.py
```


## Environment

- Pytorch: 2.5 cuda 12.4
- Python: 3.9
- transformer: 4.46.3
- albumentations: 1.2.0
- 

## Training and Validation

You can first split your dataset:

```python
python split.py --dataset_name busi --dataset_root ./data
```



Then, train and validate your dataset:

```python
# + DeepSeeK 28-th layer:
python main.py --mode deepseek --layer 27 --base_dir ./data/busi --train_file_dir busi_train.txt --val_file_dir busi_val.txt
# + DeepSeeK (T) 18-th layer:
python main.py --mode deepseek --layer 17 --unfreeze --base_dir./data/busi --train_file_dir busi_train.txt --val_file_dir busi_val.txt
# + DS Transformer 18-th layer:
python main.py --mode deepseek --layer 17 --need_init --base_dir./data/busi --train_file_dir busi_train.txt --val_file_dir busi_val.txt


# + LLaMA 18-th layer:
python main.py --mode llama --layer 17 --base_dir ./data/busi --train_file_dir busi_train.txt --val_file_dir busi_val.txt
# + LLaMA (T) 8-th layer:
python main.py --mode deepseek --layer 7 --unfreeze --base_dir./data/busi --train_file_dir busi_train.txt --val_file_dir busi_val.txt

```



