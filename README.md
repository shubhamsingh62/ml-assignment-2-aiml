# Credit Card Default Prediction

**Live Streamlit app:** https://ml-assignment-ss.streamlit.app/

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

Note: on the deployed Streamlit app the KNN numbers are slightly different in the third decimal (for example accuracy 0.8122 instead of 0.8125). A few test rows have two neighbours at exactly the same distance for the 15th spot, and different machines (CPU and maths library) can round these distances a little differently, so a different neighbour gets picked. My PC and the BITS Virtual Lab both give 0.8125. The other four models give the same numbers everywhere.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best precision of all five models (0.70), so when it says a customer will default it is usually right. The problem is that it predicts default very rarely. It found only 308 of the 1,327 defaulters (recall 0.23), so its F1 is the lowest (0.35). Since it is a linear model, it can't easily learn rules like "more than two months late means high risk". |
| Decision Tree | Best model on accuracy (0.818), AUC (0.760), F1 (0.466) and MCC (0.393). It also found the most defaulters (476) among the models with decent accuracy. Most of its decisions come from `PAY_1`, the latest repayment status, which has 65% of the feature importance. A tree can split directly on things like "2 months late", which suits this data. I kept `max_depth` at 6 so it does not overfit. |
| KNN | Second best on AUC (0.750) and F1 (0.446), and its MCC (0.371) is almost the same as SVM. It needs scaled features, otherwise large columns like the bill amounts would dominate the distance. It also keeps the whole training set inside the model, so it is the biggest saved file (4.5 MB). |
| Naive Bayes | Worst model here. It marks 4,872 of the 6,000 test customers as defaulters. That gives the highest recall (0.91), but accuracy is only 0.37, much lower than just predicting "no default" for everyone (0.779). MCC is also the lowest (0.135). Gaussian NB assumes the features are independent and normally distributed, which is not true for this data: the six bill amounts are highly correlated with each other (0.78 to 0.95) and the money columns are very skewed. |
| Support Vector Machine | Second highest accuracy (0.816) and precision (0.687), but recall is only 0.31 and its AUC (0.727) is lower than Logistic Regression, Decision Tree and KNN. It was also the slowest to train, about 45 seconds when every other model took under a second, and it did not give better results for that extra time. |
| **Overall winner** | **Decision Tree.** It has the best accuracy, AUC, F1 and MCC, it trains in under a second, the saved model is only 11 KB, and it is easy to see why it makes a decision. I also tested the app on a random 2,000-row part of the test set and got the same picture: Decision Tree best by MCC, Logistic Regression best precision, Naive Bayes best recall. Recall is low for every model because only 22% of customers default, and the best MCC is only 0.39, so predicting default from this data is hard. |

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
