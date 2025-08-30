#  Wafer Defect Detection using CNN

This project implements a deep learning model to classify wafer map defect patterns into multiple categories (e.g., Center, Donut, Edge-Loc, Scratch, etc.).  
It uses a Convolutional Neural Network (CNN) trained on a wafer defect dataset.

---

## 📂 Repository Structure
```

wafer-defect-detection/
├── data                  # dataset (empty, download separately)
│   └── .gitkeep
├── models                # trained models
│   └── wafer_defect_model_generator.h5
├── results               # evaluation results (confusion matrix)
├── src                   # source code
│   ├── train.py           # training script
│   ├── evaluate.py        # evaluation script
│   └── model_summary.py   # model summary
├── scripts/               # helper scripts (dataset download)
│   └── download_data.sh
├── requirements.txt       # dependencies
└── README.md              # project documentation

````

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/sarthak30102003/Wafer-Detection.git
cd wafer-defect-detection
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Download Dataset

The dataset (\~1.5 GB) is too large for GitHub. Download it from:

* [📦 Kaggle Dataset](https://www.kaggle.com/datasets/sarthakaggarwal3010/wafer-detection)
* [🤗 Hugging Face Dataset](https://huggingface.co/datasets/Sarthak123Agg/wafer_detection)

Place it inside the `data/` folder:

```
data/
├── train/
│   ├── Center/
│   ├── Donut/
│   └── ...
└── test/
    ├── Center/
    ├── Donut/
    └── ...
```

---

## 🏋️‍♂️ Training

Train the model using the dataset:

```bash
python src/train.py --dataset_dir data/train --epochs 10 --batch_size 32 --img_size 128
```

The trained model will be saved to:

```
models/wafer_defect_model_generator.h5
```

---

## 📊 Evaluation

Evaluate the trained model on the test set:

```bash
python src/evaluate.py
```

* Shows per-image predictions ✅/❌
* Prints overall accuracy
* Saves confusion matrix at:

```
results/confusion_matrix.png
```

---

## 🏗 Model Summary

To view the architecture of the trained model:

```bash
python src/model_summary.py
```

---

## ⚙️ Requirements

All dependencies are listed in `requirements.txt`. Main packages:

* TensorFlow / Keras
* NumPy, Pandas, Scikit-learn
* OpenCV
* Matplotlib, Seaborn

---

## 📜 License

This project is licensed under the MIT License.
Feel free to use, modify, and share!

---

## 🤝 Contributing

Pull requests are welcome.
If you’d like to improve the dataset handling, model architecture, or evaluation pipeline, feel free to fork and submit a PR.
