import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('loan_data.csv')
df.head()

# Basic info and description
df.info()
df.describe()

# Visualizing the distribution of Loan Status
temp = df['Loan_Status'].value_counts()
plt.pie(temp.values, labels=temp.index, autopct='%1.1f%%')
plt.show()

# Visualizing categorical features
plt.subplots(figsize=(15, 5))
for i, col in enumerate(['Gender', 'Married']):
    plt.subplot(1, 2, i+1)
    sb.countplot(data=df, x=col, hue='Loan_Status')
plt.tight_layout()
plt.show()

# Visualizing continuous features
plt.subplots(figsize=(15, 5))
for i, col in enumerate(['ApplicantIncome', 'LoanAmount']):
    plt.subplot(1, 2, i+1)
    sb.histplot(df[col], kde=True)
plt.tight_layout()
plt.show()

# Removing extreme outliers
df = df[df['ApplicantIncome'] < 25000]
df = df[df['LoanAmount'] < 400000]

# Encode categorical variables
def encode_labels(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
    return data

df = encode_labels(df)

# Correlation Heatmap
sb.heatmap(df.corr() > 0.8, annot=True, cbar=False)
plt.show()

# Splitting features and target
features = df.drop('Loan_Status', axis=1)
target = df['Loan_Status'].values

# Train-test split
X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, random_state=10)

# Apply SMOTE for balancing the data
smote = SMOTE(sampling_strategy='minority', random_state=0)
X_train, Y_train = smote.fit_resample(X_train, Y_train)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# SVC Model with GridSearchCV for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['rbf', 'linear']
}
svc = SVC()
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, Y_train)

best_svc = grid_search.best_estimator_

# Evaluate the best SVC model
train_pred_svc = best_svc.predict(X_train)
val_pred_svc = best_svc.predict(X_val)

print('Best SVC Parameters:', grid_search.best_params_)
print('Training ROC AUC Score:', roc_auc_score(Y_train, train_pred_svc))
print('Validation ROC AUC Score:', roc_auc_score(Y_val, val_pred_svc))

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=10)
rf_model.fit(X_train, Y_train)

# Evaluate Random Forest Model
train_pred_rf = rf_model.predict(X_train)
val_pred_rf = rf_model.predict(X_val)

print('Random Forest Training ROC AUC Score:', roc_auc_score(Y_train, train_pred_rf))
print('Random Forest Validation ROC AUC Score:', roc_auc_score(Y_val, val_pred_rf))

# Cross-validation for Random Forest
cv_scores_rf = cross_val_score(rf_model, X_train, Y_train, cv=5, scoring='roc_auc')
print('Random Forest Cross-Validation ROC AUC Scores:', cv_scores_rf)
print('Mean ROC AUC Score (CV):', cv_scores_rf.mean())

# Confusion matrix for SVC
cm_svc = confusion_matrix(Y_val, val_pred_svc)
plt.figure(figsize=(6, 6))
sb.heatmap(cm_svc, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('SVC Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Classification Report for SVC
print("SVC Classification Report:")
print(classification_report(Y_val, val_pred_svc))

# Confusion matrix for Random Forest
cm_rf = confusion_matrix(Y_val, val_pred_rf)
plt.figure(figsize=(6, 6))
sb.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Classification Report for Random Forest
print("Random Forest Classification Report:")
print(classification_report(Y_val, val_pred_rf))
