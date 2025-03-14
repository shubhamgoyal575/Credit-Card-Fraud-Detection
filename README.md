# Credit Card Fraud Detection

## Overview
This project aims to detect fraudulent credit card transactions using machine learning techniques. The dataset contains transactions labeled as fraudulent or legitimate, and the goal is to build a model that can accurately identify fraudulent activities.

## Dataset
The dataset used for this project is the **Credit Card Fraud Detection dataset** available on [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud). It consists of:
- **284,807** transactions
- **492** fraudulent transactions (0.172% of total data)
- Features: 30 columns (V1-V28 are PCA-transformed, Time, Amount, and Class (target variable))

## Approach
1. **Data Preprocessing**
   - Handling class imbalance using techniques like SMOTE or under-sampling.
   - Scaling numerical features using StandardScaler or MinMaxScaler.
   
2. **Exploratory Data Analysis (EDA)**
   - Understanding the distribution of transactions.
   - Identifying correlations and patterns.

3. **Model Selection & Training**
   - Logistic Regression
   - Random Forest
   - Support Vector Machine (SVM)
   - XGBoost
   - Neural Networks

4. **Evaluation Metrics**
   - Precision, Recall, F1-score
   - AUC-ROC Curve
   - Confusion Matrix

## Deployment with AWS SageMaker
This project utilizes **AWS SageMaker** for deploying the trained fraud detection model. The deployment steps include:
1. **Model Training on SageMaker**
   - Using built-in SageMaker algorithms or custom scripts.
   - Training the model with SageMaker’s managed Jupyter notebooks.

2. **Model Deployment**
   - Deploying the trained model as a real-time endpoint.
   - Using SageMaker Inference for making predictions on incoming transactions.

3. **Monitoring & Optimization**
   - Implementing CloudWatch for monitoring performance.
   - Fine-tuning the model based on incoming data.


## Results
- The best model achieved **X% accuracy, Y% recall, and Z% F1-score**.
- The recall score was prioritized to reduce false negatives (missing fraudulent transactions).

## Future Improvements
- Implementing deep learning models for better accuracy.
- Deploying the model using Flask or FastAPI.
- Real-time fraud detection using streaming data.

## Contributing
Contributions are welcome! Feel free to submit pull requests or raise issues.

## License
This project is licensed under the MIT License.

## Acknowledgments
- [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Machine Learning and Data Science Communities

---

⭐ Feel free to star this repository if you find it useful!

