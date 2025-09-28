# Skin Cancer Classification using Deep Learning

This project applies **deep learning models** to classify skin lesion images (HAM10000 or ISIC dataset).  
We implemented **five supervised models** and compared their performance:

- EfficientNetB0 (transfer learning + fine-tuning)
- ResNet50 (transfer learning + label smoothing)
- MobileNetV2 (transfer learning + strong augmentation)
- DenseNet121 (transfer learning + class weights)
- Custom CNN (from scratch with BN & Dropout, baseline)

---

## 📂 Project Structure

        skin-cancer/
        ├── data/ # dataset (train/val/test folders)
        ├── notebooks/ # optional: per-member exploration notebooks
        ├── results/ # metrics, confusion matrices, logs
        ├── src/
        │ ├── config.py # constants (image size, batch size, seed)
        │ ├── data_loader.py # dataset loading + augmentation
        │ ├── utils.py # training helpers + evaluation
        │ └── models/ # each model in a separate file
        │ ├── effnet_b0.py
        │ ├── resnet50.py
        │ ├── mobilenet_v2.py
        │ ├── densenet121.py
        │ └── custom_cnn.py
        ├── train.py # main training script
        ├── requirements.txt # dependencies
        └── README.md # this file

## 📊 Dataset

We used the **HAM10000** dataset (7 skin lesion classes)  
👉 https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000  

Alternatively, this code also works with **ISIC binary melanoma** datasets by switching to binary mode in `src/config.py`.

# 1. Clone your GitHub repo
!git clone https://github.com/your-username/skin-cancer.git
%cd skin-cancer

# 2. Install requirements
!pip install -r requirements.txt

# 3. Train a model (pick one)
!python train.py --model effnet --data data --epochs 20 --unfreeze 20


# EfficientNetB0
!python train.py --model effnet --data data --epochs 20 --unfreeze 20

        Uses EfficientNetB0 (pretrained on ImageNet).
        Stage 1: Trains only the classifier head (frozen base).
        Stage 2: “Unfreezes” the last 20 layers of EfficientNet → adjusts those layers to fit skin cancer images.
        Runs for 20 epochs (with early stopping, so may stop earlier).
        Efficient, accurate, and compact.
        
# ResNet50
!python train.py --model resnet50 --data data --epochs 20 --unfreeze 30

        Uses ResNet50 with residual connections.
        Label smoothing is applied in train.py → prevents overconfidence.
        Fine-tunes the last 30 layers after warmup.
        Runs for 20 epochs.
        More computationally heavy than EfficientNet, but reliable.

# MobileNetV2
!python train.py --model mobilenetv2 --data data --epochs 20 --unfreeze 20

        Uses MobileNetV2, a lightweight CNN (depthwise separable convolutions).
        Good for speed and efficiency → trains faster.
        Fine-tunes the last 20 layers.
        Stronger data augmentation is usually used with MobileNet to improve generalization.

# DenseNet121
!python train.py --model densenet --data data --epochs 20 --unfreeze 30

        Uses DenseNet121, which connects each layer to all following layers (dense connections).
        Excellent for feature reuse → strong results in medical imaging.
        Fine-tunes the last 30 layers after warmup.
        Runs for 20 epochs.
        Heavier than MobileNet, but very accurate.


# Custom CNN
!python train.py --model custom --data data --epochs 20

        Uses your own CNN architecture (with Conv → BN → ReLU → Dropout).
        No pretrained weights → trains from scratch.
        No “unfreeze” option (since nothing is pretrained).
        Runs for 20 epochs.
        Trains fastest, but accuracy usually lower than transfer learning models.

