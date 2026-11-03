from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
TARGET = "default_next_month"
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "SVM"]
CLASS_LABELS = ["No default", "Default"]

st.set_page_config(page_title="Credit Card Default Classifier", layout="wide")


@st.cache_resource
def load_model(name):
    return joblib.load(MODEL_DIR / (name.lower().replace(" ", "_") + ".joblib"))


@st.cache_data(show_spinner="Running model...")
def run_model(name, df):
    model = load_model(name)
    X = df[list(model.feature_names_in_)]
    y = df[TARGET]
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        # AUC is undefined if the file only has one class
        "AUC": roc_auc_score(y, y_prob) if y.nunique() > 1 else float("nan"),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return metrics, y_pred


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


st.title("Credit Card Default Prediction")
st.caption(
    "Five classifiers trained on the UCI *Default of Credit Card Clients* dataset "
    "(Taiwan, 2005). Upload held-out test data to see how each saved model performs."
)

with st.sidebar:
    st.header("1. Test data")
    uploaded = st.file_uploader("Upload a test CSV", type="csv")
    use_sample = st.checkbox("Use the sample test data instead")
    st.download_button(
        "Download sample test_data.csv",
        data=(BASE_DIR / "test_data.csv").read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )

    st.header("2. Model")
    model_name = st.selectbox("Choose a classifier", MODEL_NAMES)

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    df = pd.read_csv(BASE_DIR / "test_data.csv")
else:
    st.info(
        "Upload a test CSV from the sidebar to get started. It needs the 23 feature "
        f"columns plus the `{TARGET}` target column (1 = default, 0 = no default)."
    )
    st.stop()

required = list(load_model(model_name).feature_names_in_) + [TARGET]
missing = [col for col in required if col not in df.columns]
if missing:
    st.error("The uploaded file is missing these columns: " + ", ".join(missing))
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Test rows", f"{len(df):,}")
c2.metric("Actual defaults", f"{int(df[TARGET].sum()):,}")
c3.metric("Default rate", f"{df[TARGET].mean():.1%}")

with st.expander("Preview test data"):
    st.dataframe(df.head(20))

tab_model, tab_compare = st.tabs([model_name, "Compare all models"])

with tab_model:
    metrics, y_pred = run_model(model_name, df)

    for col, (metric, value) in zip(st.columns(6), metrics.items()):
        col.metric(metric, f"{value:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        fig = plot_confusion_matrix(df[TARGET], y_pred)
        st.pyplot(fig, width=420)
        plt.close(fig)
    with right:
        st.subheader("Classification report")
        report = classification_report(
            df[TARGET], y_pred, labels=[0, 1], target_names=CLASS_LABELS,
            output_dict=True, zero_division=0,
        )
        report.pop("accuracy", None)
        st.dataframe(pd.DataFrame(report).T.round(4))

with tab_compare:
    results = pd.DataFrame({name: run_model(name, df)[0] for name in MODEL_NAMES}).T
    st.dataframe(results.round(4))

    best = results["MCC"].idxmax()
    st.success(f"Best model on this data by MCC: **{best}** ({results.loc[best, 'MCC']:.4f})")
    st.bar_chart(results[["F1", "MCC"]], stack=False)
