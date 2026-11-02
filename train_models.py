import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/default of credit card clients.xls"
TARGET = "default_next_month"
RANDOM_STATE = 2026

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

print("Features:", X.shape[1])
print("Train rows:", len(X_train), "| Test rows:", len(X_test))
print("Default rate - train:", round(y_train.mean(), 4), "| test:", round(y_test.mean(), 4))
print("EDUCATION codes:", sorted(df["EDUCATION"].unique().tolist()))
print("MARRIAGE codes:", sorted(df["MARRIAGE"].unique().tolist()))
