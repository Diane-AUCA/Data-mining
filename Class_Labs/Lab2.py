import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import ModelSpec as MS, summarize
from ISLP import confusion_table
from ISLP.models import contrast
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# The Stock Market Data
Smarket = load_data("Smarket")
print(Smarket)

print(Smarket.columns)

print(Smarket.select_dtypes(include="number").corr())
Smarket.plot(y="Volume")
plt.show()

# Logistic Regression

allvars = Smarket.columns.drop(["Today", "Direction", "Year"])
design = MS(allvars)
X = design.fit_transform(Smarket)
y = Smarket.Direction == "Up"
glm = sm.GLM(y, X, family=sm.families.Binomial())
results = glm.fit()
print(summarize(results))

print(results.params)
print(results.pvalues)
probs = results.predict()
print(probs[:10])

labels = np.array(["Down"] * 1250)
labels[probs > 0.5] = "Up"

print(confusion_table(labels, Smarket.Direction))
print((507 + 145) / 1250, np.mean(labels == Smarket.Direction))

train = Smarket.Year < 2005
Smarket_train = Smarket.loc[train]
Smarket_test = Smarket.loc[~train]
print(Smarket_test.shape)

X_train, X_test = X.loc[train], X.loc[~train]
y_train, y_test = y.loc[train], y.loc[~train]
glm_train = sm.GLM(y_train, X_train, family=sm.families.Binomial())
results = glm_train.fit()
probs = results.predict(exog=X_test)
D = Smarket.Direction
L_train, L_test = D.loc[train], D.loc[~train]
labels = np.array(["Down"] * 252)
labels[probs > 0.5] = "Up"

print(confusion_table(labels, L_test))
print(np.mean(labels == L_test), np.mean(labels != L_test))

model = MS(["Lag1", "Lag2"]).fit(Smarket)
X = model.transform(Smarket)
X_train, X_test = X.loc[train], X.loc[~train]
glm_train = sm.GLM(y_train, X_train, family=sm.families.Binomial())
results = glm_train.fit()
probs = results.predict(exog=X_test)
labels = np.array(["Down"] * 252)
labels[probs > 0.5] = "Up"

print(confusion_table(labels, L_test))
print((35 + 106) / 252, 106 / (106 + 76))

newdata = pd.DataFrame({"Lag1": [1.2, 1.5], "Lag2": [1.1, -0.8]})
newX = model.transform(newdata)
print(results.predict(newX))

# Linear Discriminant Analysis
lda = LDA(store_covariance=True)
X_train, X_test = [M.drop(columns=["intercept"]) for M in [X_train, X_test]]

lda.fit(X_train, L_train)

print("LDA Means:")
print(lda.means_)

print("LDA Classes:")
print(lda.classes_)

print("LDA Priors:")
print(lda.priors_)

print("LDA Scalings:")
print(lda.scalings_)

lda_pred = lda.predict(X_test)

print("LDA Confusion Table:")
print(confusion_table(lda_pred, L_test))

lda_prob = lda.predict_proba(X_test)

print("LDA Probability Predictions:")
print(lda_prob[:10])

print(
    "LDA probability predictions match predictions:", np.all(np.where(lda_prob[:, 1] >= 0.5, "Up", "Down") == lda_pred)
)

print(
    "LDA argmax predictions match predictions:", np.all([lda.classes_[i] for i in np.argmax(lda_prob, 1)] == lda_pred)
)

print("Number of predictions with probability > 0.9:")
print(np.sum(lda_prob[:, 0] > 0.9))

# Quadratic Discriminant Analysis
qda = QDA(store_covariance=True)
qda.fit(X_train, L_train)

print("QDA Means and Priors:")
print(qda.means_, qda.priors_)

print("QDA Covariance:")
print(qda.covariance_[0])

qda_pred = qda.predict(X_test)

print("QDA Confusion Table:")
print(confusion_table(qda_pred, L_test))

print("QDA Accuracy:")
print(np.mean(qda_pred == L_test))

# Naive Bayes
NB = GaussianNB()
NB.fit(X_train, L_train)

print("Naive Bayes Classes:")
print(NB.classes_)

print("Naive Bayes Class Prior:")
print(NB.class_prior_)

print("Naive Bayes Means:")
print(NB.theta_)

print("Naive Bayes Variances:")
print(NB.var_)

print("Down Mean:")
print(X_train[L_train == "Down"].mean())

print("Down Variance:")
print(X_train[L_train == "Down"].var(ddof=0))

nb_labels = NB.predict(X_test)

print("Naive Bayes Confusion Table:")
print(confusion_table(nb_labels, L_test))

print("Naive Bayes Probability Predictions:")
print(NB.predict_proba(X_test)[:5])

# K-Nearest Neighbors
knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_train, L_train)
knn1_pred = knn1.predict(X_test)

print("KNN K=1 Confusion Table:")
print(confusion_table(knn1_pred, L_test))

print("KNN K=1 Accuracy:")
print((83 + 43) / 252, np.mean(knn1_pred == L_test))

knn3 = KNeighborsClassifier(n_neighbors=3)
knn3_pred = knn3.fit(X_train, L_train).predict(X_test)

print("KNN K=3 Accuracy:")
print(np.mean(knn3_pred == L_test))

# Caravan Data
Caravan = load_data("Caravan")
Purchase = Caravan.Purchase

print("Purchase Value Counts:")
print(Purchase.value_counts())

print("Purchase Yes Proportion:")
print(348 / 5822)

feature_df = Caravan.drop(columns=["Purchase"])
scaler = StandardScaler(with_mean=True, with_std=True, copy=True)
scaler.fit(feature_df)

X_std = scaler.transform(feature_df)
feature_std = pd.DataFrame(X_std, columns=feature_df.columns)

print("Standardized Feature Standard Deviations:")
print(feature_std.std())

(X_train, X_test, y_train, y_test) = train_test_split(feature_std, Purchase, test_size=1000, random_state=0)

knn1 = KNeighborsClassifier(n_neighbors=1)
knn1_pred = knn1.fit(X_train, y_train).predict(X_test)

print("KNN Caravan Error Rate:")
print(np.mean(y_test != knn1_pred), np.mean(y_test != "No"))

print("KNN Caravan Confusion Table:")
print(confusion_table(knn1_pred, y_test))

print("Precision:")
print(9 / (53 + 9))

for K in range(1, 6):
    knn = KNeighborsClassifier(n_neighbors=K)
    knn_pred = knn.fit(X_train, y_train).predict(X_test)
    C = confusion_table(knn_pred, y_test)

    templ = "K={0:d}: # predicted to rent: {1: >2} ," + " # who did rent {2:d}, accuracy {3:.1%}"

    pred = C.loc["Yes"].sum()
    did_rent = C.loc["Yes", "Yes"]

    print(templ.format(K, pred, did_rent, did_rent / pred))

logit = LogisticRegression(C=1e10, solver="liblinear")
logit.fit(X_train, y_train)

logit_pred = logit.predict_proba(X_test)

logit_labels = np.where(logit_pred[:, 1] > 5, "Yes", "No")

print("Logistic Regression Confusion Table - Threshold 5:")
print(confusion_table(logit_labels, y_test))

logit_labels = np.where(logit_pred[:, 1] > 0.25, "Yes", "No")

print("Logistic Regression Confusion Table - Threshold 0.25:")
print(confusion_table(logit_labels, y_test))

print("Precision:")
print(9 / (20 + 9))

# Linear and Poisson Regression on the Bikeshare Data
Bike = load_data("Bikeshare")

print("Bike Dataset Shape and Columns:")
print(Bike.shape, Bike.columns)

X = MS(["mnth", "hr", "workingday", "temp", "weathersit"]).fit_transform(Bike)
Y = Bike["bikers"]

M_lm = sm.OLS(Y, X).fit()

print("Linear Regression Summary:")
print(summarize(M_lm))

hr_encode = contrast("hr", "sum")
mnth_encode = contrast("mnth", "sum")

X2 = MS([mnth_encode, hr_encode, "workingday", "temp", "weathersit"]).fit_transform(Bike)

M2_lm = sm.OLS(Y, X2).fit()

S2 = summarize(M2_lm)

print("Second Linear Regression Summary:")
print(S2)

print("Sum of Squared Differences:")
print(np.sum((M_lm.fittedvalues - M2_lm.fittedvalues) ** 2))

print("Are the fitted values equal?")
print(np.allclose(M_lm.fittedvalues, M2_lm.fittedvalues))

coef_month = S2[S2.index.str.contains("mnth")]["coef"]

months = Bike["mnth"].dtype.categories

coef_month = pd.concat([coef_month, pd.Series([-coef_month.sum()], index=["mnth[Dec]"])])

print("Monthly Coefficients:")
print(coef_month)

fig_month, ax_month = subplots(figsize=(8, 8))

x_month = np.arange(coef_month.shape[0])

ax_month.plot(x_month, coef_month, marker="o", ms=10)
ax_month.set_xticks(x_month)
ax_month.set_xticklabels([l[5] for l in coef_month.index], fontsize=20)
ax_month.set_xlabel("Month", fontsize=20)
ax_month.set_ylabel("Coefficient", fontsize=20)

plt.show()

coef_hr = S2[S2.index.str.contains("hr")]["coef"]

coef_hr = coef_hr.reindex(["hr [{0}]".format(h) for h in range(23)])

coef_hr = pd.concat([coef_hr, pd.Series([-coef_hr.sum()], index=["hr [23]"])])

print("Hourly Coefficients:")
print(coef_hr)

fig_hr, ax_hr = subplots(figsize=(8, 8))

x_hr = np.arange(coef_hr.shape[0])

ax_hr.plot(x_hr, coef_hr, marker="o", ms=10)
ax_hr.set_xticks(x_hr[::2])
ax_hr.set_xticklabels(range(24)[::2], fontsize=20)
ax_hr.set_xlabel("Hour", fontsize=20)
ax_hr.set_ylabel("Coefficient", fontsize=20)

plt.show()

M_pois = sm.GLM(Y, X2, family=sm.families.Poisson()).fit()

S_pois = summarize(M_pois)

print("Poisson Regression Summary:")
print(S_pois)

coef_month = S_pois[S_pois.index.str.contains("mnth")]["coef"]

coef_month = pd.concat([coef_month, pd.Series([-coef_month.sum()], index=["mnth[Dec]"])])

coef_hr = S_pois[S_pois.index.str.contains("hr")]["coef"]

coef_hr = pd.concat([coef_hr, pd.Series([-coef_hr.sum()], index=["hr [23]"])])

print("Poisson Monthly Coefficients:")
print(coef_month)

print("Poisson Hourly Coefficients:")
print(coef_hr)

fig_pois, (ax_month, ax_hr) = subplots(1, 2, figsize=(16, 8))

ax_month.plot(x_month, coef_month, marker="o", ms=10)

ax_month.set_xticks(x_month)

ax_month.set_xticklabels([l[5] for l in coef_month.index], fontsize=20)

ax_month.set_xlabel("Month", fontsize=20)
ax_month.set_ylabel("Coefficient", fontsize=20)

ax_hr.plot(x_hr, coef_hr, marker="o", ms=10)

ax_hr.set_xticklabels(range(24)[::2], fontsize=20)

ax_hr.set_xlabel("Hour", fontsize=20)
ax_hr.set_ylabel("Coefficient", fontsize=20)

plt.show()

fig, ax = subplots(figsize=(8, 8))

ax.scatter(M2_lm.fittedvalues, M_pois.fittedvalues, s=20)

ax.set_xlabel("Linear Regression Fit", fontsize=20)

ax.set_ylabel("Poisson Regression Fit", fontsize=20)

ax.axline([0, 0], c="black", linewidth=3, linestyle="--", slope=1)

plt.show()