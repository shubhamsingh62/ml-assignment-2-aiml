import pandas as pd

# The Excel file has an extra row of labels (X1, X2, ...) above the real
# column names, so header=1 tells pandas to use the second row as headers.
df = pd.read_excel("data/default of credit card clients.xls", header=1)

print("Shape:", df.shape)
print(df.head())
print(df.dtypes)
print("Missing values:", df.isnull().sum().sum())

print(df["default payment next month"].value_counts(normalize=True))

print(df["EDUCATION"].value_counts().sort_index())
print(df["MARRIAGE"].value_counts().sort_index())
