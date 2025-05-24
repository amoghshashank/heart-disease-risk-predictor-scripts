🫀 Heart Disease Risk Predictor – Script-Based Version
This repository implements a heart disease risk prediction model using Python scripts and scikit-learn. It leverages the UCI Cleveland Heart Disease dataset and encompasses comprehensive data preprocessing, model training, evaluation, and comparison of classifiers.

🗂️ Project Structure
bash
Copy
Edit
heart-disease-risk-predictor-scripts/
├── data/                    # Contains original and processed data files
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
├── notebooks/               # Initial data exploration
│   └── data_exploration.py
├── scripts/                 # Core processing pipeline
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── model_comparison.py
│   └── best_model.py        # Final tuned Random Forest model
├── best_model.joblib        # Saved best-performing model
├── requirements.txt         # Project dependencies
├── LICENSE                  # MIT License
└── README.md                # Project documentation
🚀 How to Run
1. Clone and Set Up
bash
Copy
Edit
git clone https://github.com/amoghshashank/heart-disease-risk-predictor-scripts.git
cd heart-disease-risk-predictor-scripts
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
2. Run Scripts Step-by-Step
🧪 Data Exploration
bash
Copy
Edit
python notebooks/data_exploration.py
🧹 Preprocessing
bash
Copy
Edit
python scripts/data_preprocessing.py
Encodes categorical features

Scales numeric features

Splits data into train/test sets

Outputs X_train.csv, y_train.csv, X_test.csv, y_test.csv

🤖 Train Model (Logistic Regression)
bash
Copy
Edit
python scripts/train_model.py
Trains and evaluates a logistic regression model

Outputs classification report

Saves model to logistic_model.joblib

⚖️ Compare Multiple Models
bash
Copy
Edit
python scripts/model_comparison.py
Compares Logistic Regression, Random Forest, SVM, KNN

Plots F1 Score and Accuracy

Outputs model_comparison.png

🌲 Final Model: Tuned Random Forest
bash
Copy
Edit
python scripts/best_model.py
Performs hyperparameter tuning using GridSearchCV

Saves the best-performing model to best_model.joblib

Outputs evaluation metrics

📊 Model Performance
The tuned Random Forest model achieved the following performance metrics on the test dataset:

Accuracy: 89%

Precision: 91%

Recall: 89%

F1-Score: 90%

These results indicate a robust model capable of effectively predicting heart disease risk based on the provided features.

📦 Requirements
See requirements.txt for the full list of dependencies. To generate or update this file:

bash
Copy
Edit
pip freeze > requirements.txt
📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

👥 Credits
Developed by Amogh Shashank as part of a two-track ML + Cloud architecture showcase.

📌 Notes
The dataset used is the UCI Cleveland Heart Disease dataset.

For more details on the dataset and its attributes, refer to the UCI Machine Learning Repository.