# Bank Account Fraud Detection using ML & Deep Learning

### 📌 Project Overview
This project investigates the application of Machine Learning and Deep Learning to identify fraudulent activities in digital banking. Using a high-dimensional synthetic dataset, the research addresses the challenge of imbalanced data (where fraud cases are rare) to improve the precision of security protocols.

### 🛠️ Technical Implementation
Data Engineering: Developed a Python pipeline using Pandas and Scikit-learn. Implemented SMOTE (Synthetic Minority Over-sampling Technique) and random undersampling to balance the dataset.

Supervised Learning: Built and fine-tuned SVM and XGBoost models, establishing strong baselines for fraud classification.

Deep Learning: Engineered a Multilayer Perceptron (MLP) and a custom Multilayer Neural Network using TensorFlow/Keras to capture complex, non-linear patterns.

Evaluation Metrics: Focused on Precision-Recall (PR) Curves, F1-Score, and AUC-ROC to ensure the model effectively catches fraud while minimizing false alarms for legitimate customers.

### 📊 Analytical Outcome
The study identified XGBoost and the Multilayer Neural Network as the most robust models for this use case. The final framework successfully reduced false-positive rates, offering a scalable solution for real-time anomaly detection in financial institutions.

### 🚀 Tech Stack
Python | TensorFlow | Scikit-learn | XGBoost | Matplotlib | Seaborn
