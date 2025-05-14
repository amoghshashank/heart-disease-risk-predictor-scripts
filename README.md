# Heart Disease Risk Prediction – Script-Based Version

This repository implements a heart disease risk prediction model using Python scripts and scikit-learn. It uses the UCI Cleveland Heart Disease dataset (via Kaggle) and includes full preprocessing, model training, evaluation, and comparison of classifiers.

---

## 🗂️ Structure

```
heart-disease-risk-predictor-scripts/
├── data/                    # Contains original and processed data files
├── notebooks/               # Initial data exploration
│   └── data_exploration.py
├── scripts/                 # Core processing pipeline
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── model_comparison.py
├── requirements.txt         # Project dependencies
├── LICENSE                  # MIT License
└── README.md
```

---

## 🚀 How to Run

### 1. Clone and Set Up

```bash
git clone https://github.com/amoghshashank/heart-disease-risk-predictor-scripts.git
cd heart-disease-risk-predictor-scripts
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Run Scripts Step-by-Step

#### 🧪 Data Exploration

```bash
python notebooks/data_exploration.py
```

#### 🧹 Preprocessing

```bash
python scripts/data_preprocessing.py
```

* Encodes categorical features
* Scales numeric features
* Splits data into train/test
* Outputs `X_train.csv`, `y_train.csv`, etc.

#### 🤖 Train Model (Logistic Regression)

```bash
python scripts/train_model.py
```

* Trains and evaluates a logistic regression model
* Outputs classification report
* Saves model to `logistic_model.joblib`

#### ⚖️ Compare Multiple Models

```bash
python scripts/model_comparison.py
```

* Compares Logistic Regression, Random Forest, SVM, KNN
* Plots F1 Score and Accuracy
* Outputs `model_comparison.png`

---

## 📦 Requirements

See `requirements.txt` for full list. To generate/update:

```bash
pip freeze > requirements.txt
```

---

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 👥 Credits

Developed by Amogh Shashank as a part of a two-track ML + Cloud architecture showcase.
