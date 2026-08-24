# Core data handling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

RANDOM_STATE = 42

#DATA PREPARATION

# UCI Credit Approval attribute names (anonymized by data provider — no header row in raw file)
col_names = [f"A{i}" for i in range(1, 16)] + ["class"]

# Load raw data: no header, '?' marks missing values -> convert to NaN
demo = pd.read_csv("crx.data", header=None, names=col_names, na_values="?")
print("Shape:", demo.shape)
demo.head()

numeric_cols = ["A2", "A3", "A8", "A11", "A14", "A15"]
categorical_cols = ["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"]

# A2/A14 read as object due to embedded '?' — force numeric
for c in numeric_cols:
    demo[c] = pd.to_numeric(demo[c], errors="coerce")

# Encode target as binary: '+' (approved) -> 1, '-' (denied) -> 0
demo["class"] = demo["class"].map({"+": 1, "-": 0})

# Write prepared DataFrame to CSV as required by task
demo.to_csv("credit_approval_prepared.csv", index=False)
print("Saved credit_approval_prepared.csv with shape", demo.shape)

#2. Exploratory Data Analysis

# --- Missing values ---
print(demo.isnull().sum())  # count of missing values per column

# --- Visualization 1: class balance ---
demo["class"].value_counts().plot(kind="bar", title="Class Distribution")
plt.show()

# --- Visualization 2: numeric feature distributions ---
demo[numeric_cols].hist(bins=30, figsize=(12, 8))
plt.suptitle("Distribution of Numeric Features")
plt.show()

# --- Visualization 3: boxplots to VISUALLY identify outliers (was missing before) ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flatten(), numeric_cols):
    ax.boxplot(demo[col].dropna())
    ax.set_title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()

# --- Quantify outliers using IQR rule (supports the boxplots above) ---
for col in numeric_cols:
    q1, q3 = demo[col].quantile(0.25), demo[col].quantile(0.75)
    iqr = q3 - q1
    n_outliers = ((demo[col] < q1 - 1.5*iqr) | (demo[col] > q3 + 1.5*iqr)).sum()
    print(col, "outliers:", n_outliers)

# --- Correlation heatmap (numeric features vs target) ---
print(demo[numeric_cols + ["class"]].corr())

#3. Preprocessing, Feature Selection & Engineering

demo_proc = demo.copy()

# a) Handle missing values: median for numeric (robust to skew), mode for categorical
for col in numeric_cols:
    demo_proc[col] = demo_proc[col].fillna(demo_proc[col].median())
for col in categorical_cols:
    demo_proc[col] = demo_proc[col].fillna(demo_proc[col].mode()[0])

# a) Handle outliers: cap at IQR fences (winsorize) rather than delete rows
def cap_outliers_iqr(series, k=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return series.clip(lower=q1 - k*iqr, upper=q3 + k*iqr)

for col in numeric_cols:
    demo_proc[col] = cap_outliers_iqr(demo_proc[col])

# Feature engineering: log-transform skewed feature + engineered ratio
demo_proc["A15_log"] = np.log1p(demo_proc["A15"])
demo_proc["A8_A2_ratio"] = (demo_proc["A8"] / demo_proc["A2"].replace(0, np.nan)).fillna(0)

# b) Label encode categorical attributes
for col in categorical_cols:
    demo_proc[col] = LabelEncoder().fit_transform(demo_proc[col])

# Train/test split
feature_cols = numeric_cols + categorical_cols + ["A15_log", "A8_A2_ratio"]
X, y = demo_proc[feature_cols], demo_proc["class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# b) Scale numeric/encoded features — important since Logistic Regression is scale-sensitive
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Classification Model — Logistic Regression

logreg_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
logreg_model.fit(X_train_scaled, y_train)
logreg_pred = logreg_model.predict(X_test_scaled)

logreg_accuracy = accuracy_score(y_test, logreg_pred)
logreg_f1 = f1_score(y_test, logreg_pred)

print("Logistic Regression Accuracy:", logreg_accuracy)
print("Logistic Regression F1-score:", logreg_f1)

# Accuracy: gives an overall, easily interpretable measure of correctness.
# F1-score: chosen because the class distribution is only mildly balanced —
# F1 balances precision and recall, which matters here since both false
# approvals (credit risk) and false denials (lost business) are costly.