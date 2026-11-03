import os
import time

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "data/default of credit card clients.xls"
TARGET = "default_next_month"
RANDOM_STATE = 2026
MODEL_DIR = "model"

df = pd.read_excel(DATA_PATH, header=1)  # first row is just X1..X23 labels
df = df.drop(columns=["ID"])
df = df.rename(columns={"default payment next month": TARGET, "PAY_0": "PAY_1"})

# undocumented codes -> "others"
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

X = df.drop(columns=[TARGET])
y = df[TARGET]

# stratify since only ~22% are defaults
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# test set for the streamlit app
test_df = X_test.copy()
test_df[TARGET] = y_test
test_df.to_csv("test_data.csv", index=False)

print("Train rows:", len(X_train), "| Test rows:", len(X_test))


def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred),
    }


# scaler goes inside the pipeline so the saved model works on raw test data
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
    "Naive Bayes": GaussianNB(),
    # SVC has no predict_proba of its own, needed for AUC
    "SVM": make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(), ensemble=False)),
}

os.makedirs(MODEL_DIR, exist_ok=True)
results = {}

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    results[name] = evaluate(model, X_test, y_test)

    file_name = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, os.path.join(MODEL_DIR, file_name))
    print(f"{name}: trained and saved in {time.time() - start:.1f}s")

results_df = pd.DataFrame(results).T.round(4)
results_df.to_csv(os.path.join(MODEL_DIR, "metrics.csv"), index_label="Model")
print()
print(results_df.to_string())
