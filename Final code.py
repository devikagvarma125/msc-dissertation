# %% [markdown]
# Library Imports

# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
# from sklearn import metrics
import xgboost as xgb
import seaborn as sns
from sklearn.svm import SVC
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neural_network import MLPClassifier
# imports for neural network
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from aequitas.group import Group   # Aequitas is a package for Fairness evaluation

# %% [markdown]
# Reading the dataset

# %%
# Load Base.csv
df = pd.read_csv('Base.csv')

# %% [markdown]
# Data Preprocessing

# %%
print(df['device_fraud_count'].value_counts()) # It's 0 for all rows
df = df.drop(['device_fraud_count'], axis=1, errors='ignore') 
print(df)

# to remove null values

df.dropna(inplace=True) 
print(df)

# %% [markdown]
# Summary statistics

# %%
summary_stats = df.describe().transpose()
print(summary_stats)
# Get a summary of statistical information for each non-numerical column in the DataFrame
df.describe(include=["object", "bool"]).transpose()

# %%
# Count the number non-frauds and frauds - class distribution
fraud_values=df['fraud_bool'].value_counts()
# Reset the index of the DataFrame and rename the columns
fraud_values = df['fraud_bool'].value_counts().reset_index()
fraud_values.columns = ['fraud_bool', 'count']
print(fraud_values)

# %% [markdown]
# Define custom color palette

# %%
my_palette = sns.color_palette("husl", 2)
sns.set_style("whitegrid")

# Set up plot
plt.figure(figsize=(8, 6))

# Create bar plot
sns.barplot(data=fraud_values, x='fraud_bool', y='count', palette=my_palette, alpha=0.6)

# Customize labels and legend
plt.xlabel("Fraud", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.title("Number of Transactions by Fraud Status", fontsize=14)

# Display plot
plt.show()

# %%
# unique value in payment_type column
value_counts = df['payment_type'].value_counts()
print(value_counts)

# average income

mean_income = df['income'].mean()
print(mean_income)

# customer age
median_age = df['customer_age'].median()
print(median_age)
# Plot a histogram of the customer age
plt.hist(df['customer_age'], bins=20)
plt.xlabel('Customer Age')
plt.ylabel('Frequency')
plt.show()

# monthtly counts
df['month'].value_counts()

# %% [markdown]
# Univariate Analysis

# %%
plt.figure(figsize=(10, 6))
sns.histplot(df['income'], kde=True)
plt.xlabel('Income')
plt.ylabel('Count')
plt.title('Distribution of Income')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='payment_type')
plt.xlabel('Payment Type')
plt.ylabel('Count')
plt.title('Distribution of Payment Types')
plt.show()

# %% [markdown]
# Bivariate Analysis

# %%
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='fraud_bool', y='income')
plt.xlabel('Fraud')
plt.ylabel('Income')
plt.title('Income Distribution for Fraud vs Non-Fraud')
plt.show()

# %% [markdown]
# Heatmap and Correlation Analysis

# %%
numeric_columns = ['income', 'customer_age', 'intended_balcon_amount', 'credit_risk_score']
numeric_data = df[numeric_columns]
corr_matrix = numeric_data.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# %% [markdown]
# Time-Series Analysis

# %%
monthly_fraud_counts = df.groupby('month')['fraud_bool'].sum()
plt.figure(figsize=(10, 6))
monthly_fraud_counts.plot(kind='line')
plt.xlabel('Month')
plt.ylabel('Fraud Count')
plt.title('Monthly Fraud Count')
plt.show()

# %% [markdown]
# Data Imbalance Analysis

# %%
fraud_counts = df['fraud_bool'].value_counts()
plt.figure(figsize=(6, 4))
sns.barplot(x=fraud_counts.index, y=fraud_counts.values)
plt.xlabel('Fraud')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()

# %% [markdown]
# Cross-Tabulations and Pivot Tables

# %%
pivot_table = pd.pivot_table(df, values='income', index='housing_status', columns='payment_type', aggfunc=np.mean)
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, cmap='Blues', annot=True, fmt='.2f', cbar=False)
plt.title('Average Income by Housing Status and Payment Type')
plt.show()

# %% [markdown]
# Fraud transactions based on payment types

# %%
payment_fraud_data = df[['payment_type', 'fraud_bool']]

# Group the data by "payment_type" and calculate the count of fraud transactions for each type
fraud_counts = payment_fraud_data.groupby('payment_type')['fraud_bool'].sum()

# Plot the data as a bar plot
plt.figure(figsize=(10, 6))
fraud_counts.plot(kind='bar')
plt.title('Number of Fraud Transactions for Each Payment Type')
plt.xlabel('Payment Type')
plt.ylabel('Number of Fraud Transactions')
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# Data preprocessing for model training

# %%
# Define the undersampling strategy
undersample = RandomUnderSampler(sampling_strategy=0.5)

# Define the column names and target variable
cols = df.columns.tolist()
target = "fraud_bool"

# Exclude the target variable from the column list
cols = [c for c in cols if c != target]

# Define X (features) and Y (target)
X = df[cols]
Y = df[target]

# Undersample the dataset
X_under, Y_under = undersample.fit_resample(X, Y)

# Create DataFrames for visualization
df_before = pd.DataFrame({target: Y})
df_after = pd.DataFrame({target: Y_under})

# Visualize the class distribution before and after undersampling
fig, axs = plt.subplots(ncols=2, figsize=(13, 4.5))
sns.countplot(x=target, data=df_before, ax=axs[0])
sns.countplot(x=target, data=df_after, ax=axs[1])

# Set titles and labels for the subplots
fig.suptitle("Class Repartition Before and After Undersampling")
axs[0].set_title("Before Undersampling")
axs[1].set_title("After Undersampling")
axs[0].set_xlabel(target)
axs[1].set_xlabel(target)
axs[0].set_ylabel("Count")
axs[1].set_ylabel("Count")

# Show the plot
plt.show()

X = X_under
Y = Y_under

# %%
# Define the column names and target variable
# cols = df.columns.tolist()
# target = "fraud_bool"

# # Exclude the target variable from the column list
# cols = [c for c in cols if c != target]

# # Define X (features) and Y (target)
# X = df[cols]
# Y = df[target]

# %% [markdown]
# One hot encoding and data splitting

# %%
# Split the data into training and testing sets
# X: input features
# Y: target variable
# test_size: the proportion of the dataset to include in the test set
# random_state: the seed used by the random number generator for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# OneHotEncode on all the categorical features
# Find all the columns containing categorical features

s = (X_train.dtypes == 'object') # list of column-names and whether they contain categorical features
object_cols = list(s[s].index) # All the columns containing these features

# Create an instance of the OneHotEncoder
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

# Get one-hot-encoded columns for the training data
ohe_cols_train = pd.DataFrame(ohe.fit_transform(X_train[object_cols]))

# Get one-hot-encoded columns for the test data
ohe_cols_test = pd.DataFrame(ohe.transform(X_test[object_cols]))

# Set the index of the transformed data to match the original data
ohe_cols_train.index = X_train.index
ohe_cols_test.index = X_test.index

# Remove the original categorical columns from the training and test data
num_X_train = X_train.drop(object_cols, axis=1)
num_X_test = X_test.drop(object_cols, axis=1)

# Concatenate the numerical data with the transformed categorical data for the training data
X_train = pd.concat([num_X_train, ohe_cols_train], axis=1)

# Concatenate the numerical data with the transformed categorical data for the test data
X_test = pd.concat([num_X_test, ohe_cols_test], axis=1)

# Convert column names to strings (required in newer versions of scikit-learn)
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

# Verify the transformed training data
X_train.head(1)
print(X_train.shape)
print(y_train.shape)

# Print the shape of the encoded training data to get the number of features
print("Number of features after one-hot encoding:", X_train.shape[1])
# With this code, you split your dataset into training and testing sets using train_test_split, and then you perform one-hot encoding on the categorical features for both the training and test data. The code doesn't perform any random undersampling and maintains the original distribution of the data.

# %%
df

# %% [markdown]
# Model 1: Support Vector Machine (SVM)

# %%
# Create an instance of the SVC model
model = SVC(probability=True, random_state=2)

# Train the model using the training data
svm_model = model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_svm = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred_svm)
precision = metrics.precision_score(y_test, y_pred_svm)
recall = metrics.recall_score(y_test, y_pred_svm)
f1_score = metrics.f1_score(y_test, y_pred_svm)

# Print evaluation scores
print("Accuracy SVM:", accuracy)
print("Precision SVM:", precision)
print("Recall SVM:", recall)
print("F1 Score SVM:", f1_score)

# Create a confusion matrix
cm_svm = metrics.confusion_matrix(y_test, y_pred_svm)
cm_svm_df = pd.DataFrame(cm_svm, index=['not_fraud', 'fraud'], columns=['not_fraud', 'fraud'])

# Plot the confusion matrix
sns.heatmap(cm_svm_df, annot=True, cbar=None, cmap="Blues", fmt='g')
plt.title("Confusion Matrix SVM")
plt.tight_layout()
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.show()

# Calculate the AUC score
y_pred_svm_proba = model.predict_proba(X_test)[:, 1]
fpr_svm, tpr_svm, _ = metrics.roc_curve(y_test, y_pred_svm_proba)
auc_svm = metrics.roc_auc_score(y_test, y_pred_svm_proba)
print("AUC SVM:", auc_svm)

# Plot the ROC curve
plt.plot(fpr_svm, tpr_svm, label="SVM, auc={:.3f}".format(auc_svm))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('SVM ROC curve') 
plt.legend(loc=4)
plt.show()

# Plot the Precision-Recall curve
svm_precision, svm_recall, _ = metrics.precision_recall_curve(y_test, y_pred_svm_proba)
no_skill = len(y_test[y_test == 1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='black', label='No Skill')
plt.plot(svm_recall, svm_precision, color='orange', label='SVM')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend()
plt.show()

# %% [markdown]
# Model 2: XGboost Classifier

# %%
# model = xgb.XGBClassifier(
#     tree_method='gpu_hist', gpu_id=0, 
#     scale_pos_weight=89.67005
# )
# model.fit(X_train, y_train)

# predictions = model.predict_proba(X_test)[:,1]
# evaluate(predictions)

# Create an instance of the XGBClassifier
model = xgb.XGBClassifier(scale_pos_weight=89.67005, random_state=2)

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1_score = metrics.f1_score(y_test, y_pred)

# Print evaluation scores
print("Accuracy XGBoost:", accuracy)
print("Precision XGBoost:", precision)
print("Recall XGBoost:", recall)
print("F1 Score XGBoost:", f1_score)

# Create a confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['not_fraud', 'fraud'], columns=['not_fraud', 'fraud'])

# Plot the confusion matrix
sns.heatmap(cm_df, annot=True, cbar=None, cmap="Blues", fmt='g')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.show()

# Calculate the AUC score
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
print("AUC:", auc)

# Plot the ROC curve
plt.plot(fpr, tpr, label="XGBoost, auc={:.3f}".format(auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc=4)
plt.show()

# Plot the Precision-Recall curve
precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred_proba)
no_skill = len(y_test[y_test == 1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='black', label='No Skill')
plt.plot(recall, precision, color='orange', label='XGBoost')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend()
plt.show()

# %% [markdown]
# Model 3: MLP classifier

# %%
# Create an instance of the MLPClassifier
model = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(100, 100), random_state=2)

# Train the model using the training data
mlp_model = model.fit(X_train, y_train)

# Get the model parameters
model.get_params(deep=True)

# Make predictions on the test data
y_pred_mlp = model.predict(X_test)

# Calculate evaluation metrics
accuracy = metrics.accuracy_score(y_test, y_pred_mlp)
precision = metrics.precision_score(y_test, y_pred_mlp)
recall = metrics.recall_score(y_test, y_pred_mlp)
f1_score = metrics.f1_score(y_test, y_pred_mlp)

# Print evaluation scores
print("Accuracy MLP:", accuracy)
print("Precision MLP:", precision)
print("Recall MLP:", recall)
print("F1 Score MLP:", f1_score)

# Create a confusion matrix
matrix_mlp = confusion_matrix(y_test, y_pred_mlp)
cm_mlp = pd.DataFrame(matrix_mlp, index=['not_fraud', 'fraud'], columns=['not_fraud', 'fraud'])

# Plot the confusion matrix
sns.heatmap(cm_mlp, annot=True, cbar=None, cmap="Blues", fmt='g')
plt.title("Confusion Matrix MLP")
plt.tight_layout()
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.show()

# Calculate the AUC score
y_pred_mlp_proba = model.predict_proba(X_test)[:, 1]
fpr_mlp, tpr_mlp, _ = metrics.roc_curve(y_test, y_pred_mlp_proba)
auc_mlp = metrics.roc_auc_score(y_test, y_pred_mlp_proba)
print("AUC MLP:", auc_mlp)

# Plot the ROC curve
plt.plot(fpr_mlp, tpr_mlp, label="MLPC, auc={:.3f}".format(auc_mlp))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Multilayer Perceptron ROC curve')
plt.legend(loc=4)
plt.show()

# Plot the Precision-Recall curve
mlp_precision, mlp_recall, _ = metrics.precision_recall_curve(y_test, y_pred_mlp_proba)
no_skill = len(y_test[y_test == 1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='black', label='No Skill')
plt.plot(mlp_recall, mlp_precision, color='orange', label='MLP')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend()
plt.show()

# %% [markdown]
# Model 4 : Multilayer Neural Network with Tensorflow/Keras

# %%
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)

# %%
X_train

# %%
y_train

# %%
# Keras model using dropout and batch normalization
model = keras.Sequential([
    keras.layers.BatchNormalization(input_shape=[X_train.shape[1]]),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')
])

metrics = [
    'accuracy',tf.metrics.AUC(),tf.metrics.F1Score(),tf.metrics.Precision(),tf.metrics.Recall()
]

model.compile(
    optimizer=keras.optimizers.Adam(0.0001),
    loss="binary_crossentropy",
    metrics=metrics
)

# Use EarlyStopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
    mode='max'
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=5,
    callbacks=[early_stopping],
    verbose=1,
    validation_split=0.15
)

history_dict = history.history

# %%
# Calculate the AUC score
y_pred_keras_proba = model.predict(X_test)
fpr_keras, tpr_keras, _ = roc_curve(y_test, y_pred_keras_proba)
auc_keras = roc_auc_score(y_test, y_pred_keras_proba)

# %%
# Plot the ROC curve
plt.plot(fpr_keras, tpr_keras, label="Neural Network, auc={:.3f}".format(auc_keras))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Multilayer Neural Network with Tensorflow/Keras ROC curve')
plt.legend(loc=4)
plt.show()

# Plot the Precision-Recall curve
keras_precision, keras_recall, _ = precision_recall_curve(y_test, y_pred_keras_proba)
no_skill = len(y_test[y_test == 1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='black', label='No Skill')
plt.plot(keras_recall, keras_precision, color='orange', label='Neural Network')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend()
plt.show()


