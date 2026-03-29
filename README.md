# Image Caption Generator

A deep learning model that generates natural language captions for images, built from scratch using TensorFlow. Based on Google's [Show and Tell](https://arxiv.org/abs/1411.4555) architecture.

## How it works

The model uses two neural networks working together:

- **CNN (InceptionV3)** — pre-trained on ImageNet, extracts a 2048-dimensional feature vector from each image that represents what the model "sees"
- **LSTM** — takes that feature vector plus the words generated so far, and predicts the next word one at a time until it produces a complete sentence

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/anusha167/Image_Caption_Generator.git
cd Image_Caption_Generator
```

**2. Create environment and install dependencies**
```bash
conda create -n caption_env python=3.10 -y
conda activate caption_env
pip install -r requirements.txt
```

**3. Download the dataset**

Download the following from [cocodataset.org](https://cocodataset.org/#download) and place them in the `data/` folder:
- 2017 Val images [5K/1GB] → `data/val2017/`
- 2017 Train/Val annotations [241MB] → `data/annotations/`

**4. Train the model**
```bash
python src/train.py
```

**5. Generate a caption**
```bash
python src/inference.py
```

## Example Output
```
Image: 000000435081.jpg
Generated caption: a man is standing next to a building
```

## Built With

- TensorFlow / Keras
- InceptionV3 (pretrained on ImageNet)
- MS-COCO Dataset
