import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from ISLP import load_data
from ISLP.models import ModelSpec as MS, summarize, poly

dir()
# print(dir())
A = np.array([3, 5, 11])
dir(A)
print("Sum of A:", A.sum())
# Simple Linear Regression
Boston = load_data("Boston")
# print(Boston.columns)
X = pd.DataFrame({"intercept": np.ones(Boston.shape[0]), "lstat": Boston["lstat"]})
print("\nFirst 4 rows of X:")
print(X[:4])
y = Boston["medv"]
model = sm.OLS(y, X)
results = model.fit()
print("\nSimple Linear Regression Results:")
print(summarize(results))
design = MS(["lstat"])
design = design.fit(Boston)
X = design.transform(Boston)
print("\nFirst 4 rows of transformed X:")
print(X[:4])
design = MS(["lstat"])
X = design.fit_transform(Boston)
print("\nFirst 4 rows of X using fit_transform:")
print(X[:4])
print("\nRegression Summary:")
print(results.summary())
print("\nRegression Parameters:")
print(results.params)
new_df = pd.DataFrame({"lstat": [5, 10, 15]})
newX = design.transform(new_df)
print("\nNew X:")
print(newX)
new_predictions = results.get_prediction(newX)
print("\nPredicted Mean:")
print(new_predictions.predicted_mean)
print("\nConfidence Intervals:")
print(new_predictions.conf_int(alpha=0.05))
print("\nPrediction Intervals:")
print(new_predictions.conf_int(obs=True, alpha=0.05))


def abline(ax, b, m):
    "Add a line with slope m and intercept b to ax"
    xlim = ax.get_xlim()
    ylim = [m * xlim[0] + b, m * xlim[1] + b]
    ax.plot(xlim, ylim)


def abline(ax, b, m, *args, **kwargs):
    "Add a line with slope m and intercept b to ax"
    xlim = ax.get_xlim()
    ylim = [m * xlim[0] + b, m * xlim[1] + b]
    ax.plot(xlim, ylim, *args, **kwargs)


# Regression plot
ax = Boston.plot.scatter("lstat", "medv")
abline(ax, results.params.iloc[0], results.params.iloc[1], "r--", linewidth=3)
plt.show()

# Residual plot
ax = subplots(figsize=(8, 8))[1]
ax.scatter(results.fittedvalues, results.resid)
ax.set_xlabel("Fitted value")
ax.set_ylabel("Residual")
ax.axhline(0, c="k", ls="--")
plt.show()

infl = results.get_influence()
ax = subplots(figsize=(8, 8))[1]
ax.scatter(np.arange(X.shape[0]), infl.hat_matrix_diag)
ax.set_xlabel("Index")
ax.set_ylabel("Leverage")
plt.show()
print("\nObservation with highest leverage:")
print(np.argmax(infl.hat_matrix_diag))

# Multiple Linear Regression
X = MS(["lstat", "age"]).fit_transform(Boston)
model1 = sm.OLS(y, X)
results1 = model1.fit()
print("\nMultiple Linear Regression Results:")
print(summarize(results1))
terms = Boston.columns.drop("medv")
print("\nTerms:")
print(terms)
X = MS(terms).fit_transform(Boston)
model = sm.OLS(y, X)
results = model.fit()
print("\nFull Multiple Linear Regression Results:")
print(summarize(results))
minus_age = Boston.columns.drop(["medv", "age"])
Xma = MS(minus_age).fit_transform(Boston)
model1 = sm.OLS(y, Xma)
print("\nModel Without Age:")
print(summarize(model1.fit()))

# Multivariate Goodness of Fit

vals = [VIF(X, i) for i in range(1, X.shape[1])]
vif = pd.DataFrame({"vif": vals}, index=X.columns[1:])
print("\nVariance Inflation Factors:")
print(vif)
vals = []
for i in range(1, X.values.shape[1]):
    vals.append(VIF(X.values, i))

print("\nVIF values:")
print(vals)
# Interaction Terms
X = MS(["lstat", "age", ("lstat", "age")]).fit_transform(Boston)
model2 = sm.OLS(y, X)
print("\nInteraction Terms Results:")
print(summarize(model2.fit()))

# Non-linear Transformations of the Predictors
X = MS([poly("lstat", degree=2), "age"]).fit_transform(Boston)
model3 = sm.OLS(y, X)
results3 = model3.fit()
print("\nNon-linear Transformation Results:")
print(summarize(results3))

print("\nANOVA Results:")
print(anova_lm(results1, results3))
ax = subplots(figsize=(8, 8))[1]
ax.scatter(results3.fittedvalues, results3.resid)
ax.set_xlabel("Fitted value")
ax.set_ylabel("Residual")
ax.axhline(0, c="k", ls="--")
plt.show()

# Qualitative Predictors
Carseats = load_data("Carseats")
print("\nCarseats Columns:")
print(Carseats.columns)
allvars = list(Carseats.columns.drop("Sales"))
y = Carseats["Sales"]
final = allvars + [("Income", "Advertising"), ("Price", "Age")]
X = MS(final).fit_transform(Carseats)
model = sm.OLS(y, X)
print("\nCarseats Regression Results:")
print(summarize(model.fit()))