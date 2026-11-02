import pandas as pd

df = pd.read_excel("data/default of credit card clients.xls", header=1)  # first row is just X1..X23 labels

print("Shape:", df.shape)
print(df.head())
print(df.dtypes)
print("Missing values:", df.isnull().sum().sum())

print(df["default payment next month"].value_counts(normalize=True))

print(df["EDUCATION"].value_counts().sort_index())
print(df["MARRIAGE"].value_counts().sort_index())
