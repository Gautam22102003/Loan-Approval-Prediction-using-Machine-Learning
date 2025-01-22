Loan-Approval-Prediction-using-Machine-Learning

Key Steps:
1. Data Preprocessing:
   - The dataset is loaded and its shape and basic info are examined.
   - Missing or outlier values in the `ApplicantIncome` and `LoanAmount` columns are filtered to remove extreme values (e.g., `ApplicantIncome` greater than 25000, `LoanAmount` greater than 400000).
   
2. Exploratory Data Analysis (EDA):
   - A pie chart visualizes the distribution of loan statuses (`Loan_Status`).
   - Count plots, histograms, and boxplots are used to analyze the relationship between categorical and numerical variables (`Gender`, `Married`, `ApplicantIncome`, `LoanAmount`).

3. Label Encoding:
   - Categorical features like `Gender` and `Married` are encoded into numerical values using `LabelEncoder`.

4. Correlation Analysis:
   - A heatmap is used to analyze the correlation between features.

5. Splitting Data:
   - The dataset is split into training and validation sets, using an 80-20 split.
   - The training data is balanced using `RandomOverSampler` to address class imbalance.

6. Normalization:
   - The features are normalized using `StandardScaler` to improve the stability and speed of training.

7. Model Training (SVC):
   - A Support Vector Classifier (SVC) with an RBF kernel is used to train the model.

8. Model Evaluation:
   - The model's performance is evaluated using the ROC-AUC score and confusion matrix on both training and validation sets.
   - The classification report gives detailed metrics like precision, recall, F1-score, and support.



Expected Output:
Best hyperparameters for the SVC model.
ROC AUC scores for both the SVC and Random Forest models on both training and validation data.
Cross-validation ROC AUC scores for Random Forest.
Confusion matrix and classification report for both models, showing precision, recall, and F1-score.

