# Credit Card Fraud Detection

## 🚀 Overview
This project aims to **detect fraudulent credit card transactions** using **machine learning** techniques. Given the highly imbalanced nature of fraud detection, we apply various resampling techniques and model evaluation metrics to ensure robust predictions.

## 📂 Dataset
The dataset used for this project is the **Credit Card Fraud Detection dataset** available on [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud). It consists of:
- **284,807** transactions
- **492** fraudulent transactions (**0.172% of total data**)
- **Features**: 30 columns (**V1-V28 are PCA-transformed**), `Time`, `Amount`, and `Class` (target variable: **0 = Legitimate, 1 = Fraudulent**)

## 🛠️ Approach
### 1️⃣ Data Preprocessing
- **Handling class imbalance** using **SMOTE (Synthetic Minority Over-sampling Technique)** and **undersampling**.
- **Scaling numerical features** using `StandardScaler` or `MinMaxScaler`.
- **Feature engineering** to extract meaningful transaction patterns.

### 2️⃣ Exploratory Data Analysis (EDA)
- Distribution of transaction amounts and time.
- Fraudulent vs. legitimate transaction patterns.
- Correlation analysis of PCA-transformed features.

### 3️⃣ Model Selection & Training
We experimented with various machine learning models:
✅ **Random Forest**
✅ **AdaBoost**
✅ **XGBoost**
✅ **Lightbgm**
✅ **Neural Networks (Deep Learning Approach)**

### 4️⃣ Evaluation Metrics
Since fraud detection is an **imbalanced classification problem**, we focus on:
- **Precision, Recall, F1-score** (to minimize false negatives)
- **AUC-ROC Curve** (to evaluate the model’s discriminatory power)
- **Confusion Matrix** (to analyze misclassification rates)

## ☁️ Deployment with AWS SageMaker
This project utilizes **AWS SageMaker** for deploying the trained fraud detection model. The deployment steps include:

### **1️⃣ Model Training on SageMaker**
- Using **built-in SageMaker algorithms** or **custom scripts**.
- Training the model with **SageMaker’s managed Jupyter notebooks**.

### **2️⃣ Model Deployment**
- Deploying the trained model as a **real-time endpoint**.
- Using **SageMaker Inference** for making predictions on new transactions.


## 📊 Results
- The **best model achieved:**
  - **88.1% roc auc score** 
  - **85.2% F1-score**
- **Random Forest** and **XGBoost** were the most balanced models in terms of performance.
- **Neural Networks** performed exceptionally well in high-computational environments with **AUC close to 0.999**.

## 🔮 Future Improvements
- Implementing **deep learning architectures** for better accuracy.
- Deploying the model using **Flask or FastAPI** for API-based fraud detection.
- Enhancing **real-time fraud detection** using **streaming data** (e.g., Apache Kafka).

## 🤝 Contributing
Contributions are **welcome**! Feel free to submit **pull requests** or raise **issues**.

## 📝 License
This project is licensed under the **MIT License**.

## 📢 Acknowledgments
- Open-source **Machine Learning & Data Science Communities**

---
### ⭐ **Feel free to star this repository if you find it useful!** 🌟

