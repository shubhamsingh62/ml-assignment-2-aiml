# Credit Card Default Prediction

**Live Streamlit app:** _add link after deploying_

## a. Problem Statement

Banks that issue credit cards want to know which customers are likely to miss their next payment, so they can act early (reminders, lower limits, restructuring). The goal of this project is to predict whether a credit card customer will **default on their payment next month**, using their profile and their last six months of billing and repayment history.

This is a **binary classification** problem. Five classification models are trained on the same data, compared using six evaluation metrics, and served through an interactive Streamlit app.

## b. Dataset Description

| | |
|---|---|
| **Name** | Default of Credit Card Clients |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (Yeh & Lien, 2009) |
| **Context** | Credit card customers of a bank in Taiwan, April to September 2005 |
| **Instances** | 30,000 |
| **Features** | 23 (all numeric) |
| **Target** | `default_next_month`: 1 = default, 0 = no default |
| **Class balance** | 77.9% no default, 22.1% default (imbalanced) |
| **Missing values** | None |

**Features**

| Feature(s) | Description |
|---|---|
| `LIMIT_BAL` | Credit limit in NT dollars (individual and family/supplementary credit) |
| `SEX` | 1 = male, 2 = female |
| `EDUCATION` | 1 = graduate school, 2 = university, 3 = high school, 4 = others |
| `MARRIAGE` | 1 = married, 2 = single, 3 = others |
| `AGE` | Age in years |
| `PAY_1` to `PAY_6` | Repayment status from September (`PAY_1`) back to April 2005 (`PAY_6`). -1 = paid on time, 1 to 9 = payment delayed by that many months |
| `BILL_AMT1` to `BILL_AMT6` | Bill statement amount, September back to April 2005 |
| `PAY_AMT1` to `PAY_AMT6` | Amount paid, September back to April 2005 |

Because of the class imbalance, a model that always predicts "no default" would already reach 77.9% accuracy. F1, MCC and AUC are therefore more informative than accuracy for this dataset.

**Preprocessing**

- Dropped the `ID` column (row number only).
- Renamed `PAY_0` to `PAY_1` so the repayment columns are numbered 1 to 6 like the bill and payment columns.
- `EDUCATION` codes 0, 5 and 6 (345 rows) and `MARRIAGE` code 0 (54 rows) are not in the dataset documentation, so they were merged into the existing "others" categories.
- Stratified 80/20 train/test split (`random_state=2026`): 24,000 training rows and 6,000 test rows, both with a 22.1% default rate. The test set is saved as `test_data.csv`.
- `StandardScaler` is applied inside the pipeline for Logistic Regression, KNN and SVM, so every saved model works directly on raw test data.

## c. GitHub Repository Link

https://github.com/shubhamsingh62/ml-assignment-2-aiml

```
ml-assignment-2-aiml/
├── app.py                  # Streamlit app
├── train_models.py         # cleaning, train/test split, training, evaluation, saving models
├── explore.py              # initial data exploration
├── requirements.txt
├── README.md
├── test_data.csv           # held-out test set (23 features + target)
├── data/
│   └── default of credit card clients.xls
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── metrics.csv         # evaluation results of all five models
```

## d. Models Used

| Model | Configuration |
|---|---|
| Logistic Regression | `StandardScaler` + `LogisticRegression(max_iter=1000)` |
| Decision Tree | `DecisionTreeClassifier(max_depth=6)` |
| KNN | `StandardScaler` + `KNeighborsClassifier(n_neighbors=15)` |
| Naive Bayes | `GaussianNB` |
| SVM | `StandardScaler` + RBF-kernel `SVC`, wrapped in `CalibratedClassifierCV` to get probabilities for AUC |

### Comparison Table

All metrics are calculated on the 6,000-row test set. For Precision, Recall and F1 the positive class is "default".

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8083 | 0.7371 | 0.7016 | 0.2321 | 0.3488 | 0.3252 |
| Decision Tree | 0.8178 | 0.7598 | 0.6630 | 0.3587 | 0.4655 | 0.3925 |
| KNN | 0.8125 | 0.7496 | 0.6435 | 0.3414 | 0.4461 | 0.3710 |
| Naive Bayes | 0.3698 | 0.6816 | 0.2482 | 0.9111 | 0.3901 | 0.1351 |
| Support Vector Machine | 0.8158 | 0.7267 | 0.6869 | 0.3075 | 0.4248 | 0.3719 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | The most precise model (0.70): when it flags a customer as a defaulter, it is right 70% of the time. But it is also the most cautious, catching only 308 of the 1,327 actual defaulters (recall 0.23), which gives it the lowest F1 (0.35). As a linear model it cannot capture threshold effects such as "two or more months late". |
| Decision Tree | Best on four of the six metrics: Accuracy (0.818), AUC (0.760), F1 (0.466) and MCC (0.393). It also catches the most defaulters (476) among the models with reasonable accuracy. The most recent repayment status (`PAY_1`) accounts for 65% of the tree's feature importance, and a tree can model rules like "late by 2+ months means high risk" directly. Limiting depth to 6 prevents overfitting. |
| KNN | Very close behind the Decision Tree, with the second-best AUC (0.750) and F1 (0.446), and an MCC (0.371) practically tied with SVM. It depends on feature scaling to work well, and because it stores the whole training set it is the largest saved model (4.5 MB). |
| Naive Bayes | The clear outlier. It flags 4,872 of the 6,000 test customers as defaulters, so its recall is the highest (0.91), but its accuracy (0.37) is far below the 77.9% baseline and its MCC is the lowest (0.135). Gaussian Naive Bayes assumes features are independent and normally distributed, but here the six bill amounts are strongly correlated (0.78 to 0.95) and the money amounts are heavily skewed, so its probability estimates are unreliable. |
| Support Vector Machine | Second-highest accuracy (0.816) and precision (0.687), but recall is only 0.31 and its AUC (0.727) is the lowest of the four reasonable models. Training took about 45 seconds, compared with under one second for every other model, without any performance advantage on this dataset. |
| **Overall winner** | **Decision Tree.** It has the best MCC, F1, AUC and accuracy, trains in under a second, is small (11 KB) and its decisions are easy to interpret. All models struggle with recall because of the class imbalance, and the best MCC of 0.39 shows that predicting default from these features alone is genuinely hard. |

## Streamlit App

The app loads the saved models from `model/` and evaluates them on uploaded test data without retraining.

- Upload a test CSV (or use the bundled sample / download `test_data.csv`)
- Choose a model from the dropdown
- See all six evaluation metrics
- Confusion matrix and classification report
- "Compare all models" tab showing every model's metrics on the uploaded data

## How to Run Locally

```bash
pip install -r requirements.txt
python train_models.py     # optional: retrains and saves all models (about 1 minute)
streamlit run app.py
```
