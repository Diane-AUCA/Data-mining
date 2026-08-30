import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots, show
from statsmodels.api import OLS
import sklearn.model_selection as skm
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler
from ISLP import load_data
from ISLP.models import ModelSpec as MS
from functools import partial
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from ISLP.models import (Stepwise,sklearn_selected,sklearn_selection_path)
from l0bnb import fit_path

Hitters = load_data('Hitters')
print("Number of missing Salary values:", np.isnan(Hitters['Salary']).sum())

Hitters = Hitters.dropna();
print("Hitters shape after dropna:", Hitters.shape)

def nCp(sigma2, estimator, X, Y):
    "Negative Cp statistic"
    n, p = X.shape
    Yhat = estimator.predict(X)
    RSS = np.sum((Y - Yhat)**2)
    return -(RSS + 2 * p * sigma2) / n

design = MS(Hitters.columns.drop('Salary')).fit(Hitters)
Y = np.array(Hitters['Salary'])
X = design.transform(Hitters)
sigma2 = OLS(Y,X).fit().scale
print("sigma2:", sigma2)

neg_Cp = partial(nCp, sigma2)

strategy = Stepwise.first_peak(design,
direction='forward',
max_terms=len(design.terms))

hitters_MSE = sklearn_selected(OLS,
strategy)
hitters_MSE.fit(Hitters, Y)
print("hitters_MSE.selected_state_:", hitters_MSE.selected_state_)

strategy = Stepwise.fixed_steps(design,
len(design.terms),
direction='forward')
full_path = sklearn_selection_path(OLS, strategy)

full_path.fit(Hitters, Y)
Yhat_in = full_path.predict(Hitters)
print("Yhat_in shape:", Yhat_in.shape)

mse_fig, ax = subplots(figsize=(8,8))
insample_mse = ((Yhat_in - Y[:,None])**2).mean(0)
n_steps = insample_mse.shape[0]
ax.plot(np.arange(n_steps),
insample_mse,
'k', # color black
label='In-sample')
ax.set_ylabel('MSE',
fontsize=20)
ax.set_xlabel('# steps of forward stepwise',
fontsize=20)
ax.set_xticks(np.arange(n_steps)[::2])
ax.legend()
ax.set_ylim([50000,250000]);


K=5
kfold = skm.KFold(K,
random_state=0,
shuffle=True)
Yhat_cv = skm.cross_val_predict(full_path,
Hitters,
Y,
cv=kfold)
print("Yhat_cv shape:", Yhat_cv.shape)



cv_mse = []
for train_idx, test_idx in kfold.split(Y):
    errors=(Yhat_cv[test_idx] - Y[test_idx,None])**2
    cv_mse.append(errors.mean(0)) # column means
cv_mse = np.array(cv_mse).T
print("cv_mse shape:", cv_mse.shape)

ax.errorbar(np.arange(n_steps),
cv_mse.mean(1),
cv_mse.std(1) / np.sqrt(K),
label='Cross-validated',
c='r') # color red
ax.set_ylim([50000,250000])
ax.legend()
mse_fig
print("Displaying mse_fig (in-sample vs cross-validated MSE)...")
show()

validation = skm.ShuffleSplit(n_splits=1,
test_size=0.2,
random_state=0)
for train_idx, test_idx in validation.split(Y):
    full_path.fit(Hitters.iloc[train_idx],
    Y[train_idx])
    Yhat_val = full_path.predict(Hitters.iloc[test_idx])
    errors = (Yhat_val - Y[test_idx,None])**2
    validation_mse = errors.mean(0)

ax.plot(np.arange(n_steps),
validation_mse,
'b--', # color blue, broken line
label='Validation')
ax.set_xticks(np.arange(n_steps)[::2])
ax.set_ylim([50000,250000])
ax.legend()
mse_fig
print("Displaying mse_fig with validation curve added...")
show()

D = design.fit_transform(Hitters)
D = D.drop('intercept', axis=1)
X = np.asarray(D)

path = fit_path(X,Y,max_nonzeros=X.shape[1])
print("path[3]:", path[3])

#6.5.2 Ridge Regression and the Lasso

#Ridge Regression

Xs = X - X.mean(0)[None,:]
X_scale = X.std(0)
Xs = Xs / X_scale[None,:]
lambdas = 10**np.linspace(8, -2, 100) / Y.std()
soln_array = skl.ElasticNet.path(Xs,
Y,
l1_ratio=0.,
alphas=lambdas)[1]
print("soln_array shape:", soln_array.shape)

soln_path = pd.DataFrame(soln_array.T,
columns=D.columns,
index=-np.log(lambdas))
soln_path.index.name = 'negative log(lambda)'
print("soln_path head:\n", soln_path.head())

path_fig, ax = subplots(figsize=(8,8))
soln_path.plot(ax=ax, legend=False)
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Standardized coefficients', fontsize=20)
ax.legend(loc='upper left');
print("Displaying path_fig (ridge coefficient paths)...")
show()

beta_hat = soln_path.loc[soln_path.index[39]]
print("lambdas[39]:", lambdas[39])
print("beta_hat (index 39):\n", beta_hat)

print("Norm of beta_hat (index 39):", np.linalg.norm(beta_hat))

beta_hat = soln_path.loc[soln_path.index[59]]
print("lambdas[59]:", lambdas[59])
print("Norm of beta_hat (index 59):", np.linalg.norm(beta_hat))

ridge = skl.ElasticNet(alpha=lambdas[59], l1_ratio=0)
scaler = StandardScaler(with_mean=True, with_std=True)
pipe = Pipeline(steps=[('scaler', scaler), ('ridge', ridge)])
pipe.fit(X, Y)

print("Norm of ridge.coef_:", np.linalg.norm(ridge.coef_))

validation = skm.ShuffleSplit(n_splits=1,
test_size=0.5,
random_state=0)
ridge.alpha = 0.01
results = skm.cross_validate(ridge,
X,
Y,
scoring='neg_mean_squared_error',
cv=validation)
print("Test MSE with alpha=0.01:", -results['test_score'])

ridge.alpha = 1e10
results = skm.cross_validate(ridge,
X,
Y,
scoring='neg_mean_squared_error',
cv=validation)
print("Test MSE with alpha=1e10:", -results['test_score'])

param_grid = {'ridge__alpha': lambdas}
grid = skm.GridSearchCV(pipe,
param_grid,
cv=validation,
scoring='neg_mean_squared_error')
grid.fit(X, Y)
print("Best ridge alpha (single validation split):", grid.best_params_['ridge__alpha'])
print("Best estimator (single validation split):", grid.best_estimator_)

grid = skm.GridSearchCV(pipe,
param_grid,
cv=kfold,
scoring='neg_mean_squared_error')
grid.fit(X, Y)
print("Best ridge alpha (K-fold CV):", grid.best_params_['ridge__alpha'])
print("Best estimator (K-fold CV):", grid.best_estimator_)

ridge_fig, ax = subplots(figsize=(8,8))
ax.errorbar(-np.log(lambdas),
-grid.cv_results_['mean_test_score'],
yerr=grid.cv_results_['std_test_score'] / np.sqrt(K))
ax.set_ylim([50000,250000])
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Cross-validated MSE', fontsize=20);
print("Displaying ridge_fig (cross-validated MSE vs lambda)...")
show()

grid_r2 = skm.GridSearchCV(pipe,
param_grid,
cv=kfold)
grid_r2.fit(X, Y)

r2_fig, ax = subplots(figsize=(8,8))
ax.errorbar(-np.log(lambdas),
grid_r2.cv_results_['mean_test_score'],
yerr=grid_r2.cv_results_['std_test_score'] / np.sqrt(K)
)
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Cross-validated $R^2$', fontsize=20);
print("Displaying r2_fig (cross-validated R^2 vs lambda)...")
show()

ridgeCV = skl.ElasticNetCV(alphas=lambdas,
l1_ratio=0,
cv=kfold)
pipeCV = Pipeline(steps=[('scaler', scaler),('ridge', ridgeCV)])
pipeCV.fit(X, Y)

tuned_ridge = pipeCV.named_steps['ridge']
ridgeCV_fig, ax = subplots(figsize=(8,8))
ax.errorbar(-np.log(lambdas),tuned_ridge.mse_path_.mean(1),yerr=tuned_ridge.mse_path_.std(1) / np.sqrt(K))
ax.axvline(-np.log(tuned_ridge.alpha_), c='k', ls='--')
ax.set_ylim([50000,250000])
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Cross-validated MSE', fontsize=20);
print("Displaying ridgeCV_fig (ElasticNetCV ridge MSE path)...")
show()

print("Minimum mean CV MSE (tuned ridge):", np.min(tuned_ridge.mse_path_.mean(1)))

print("tuned_ridge.coef_:", tuned_ridge.coef_)

#Evaluating Test Error of Cross-Validated Ridge     

outer_valid = skm.ShuffleSplit(n_splits=1,
test_size=0.25,
random_state=1)
inner_cv = skm.KFold(n_splits=5,
shuffle=True,
random_state=2)
ridgeCV = skl.ElasticNetCV(alphas=lambdas,
l1_ratio=0,
cv=inner_cv)
pipeCV = Pipeline(steps=[('scaler', scaler),
('ridge', ridgeCV)]);

results = skm.cross_validate(pipeCV,
X,
Y,
cv=outer_valid,
scoring='neg_mean_squared_error')
print("Nested CV test MSE (tuned ridge):", -results['test_score'])

#THE LASSO

lassoCV = skl.ElasticNetCV(n_alphas=100,
l1_ratio=1,
cv=kfold)
pipeCV = Pipeline(steps=[('scaler', scaler),
('lasso', lassoCV)])
pipeCV.fit(X, Y)
tuned_lasso = pipeCV.named_steps['lasso']
print("tuned_lasso.alpha_:", tuned_lasso.alpha_)

lambdas , soln_array = skl.Lasso.path(Xs,
Y,
l1_ratio=1,
n_alphas=100)[:2]
soln_path = pd.DataFrame(soln_array.T,
columns=D.columns,
index=-np.log(lambdas))

path_fig, ax = subplots(figsize=(8,8))
soln_path.plot(ax=ax, legend=False)
ax.legend(loc='upper left')
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Standardized coefficiients', fontsize=20);
print("Displaying path_fig (lasso coefficient paths)...")
show()

print("Minimum mean CV MSE (tuned lasso):", np.min(tuned_lasso.mse_path_.mean(1)))

lassoCV_fig, ax = subplots(figsize=(8,8))
ax.errorbar(-np.log(tuned_lasso.alphas_),
tuned_lasso.mse_path_.mean(1),
yerr=tuned_lasso.mse_path_.std(1) / np.sqrt(K))
ax.axvline(-np.log(tuned_lasso.alpha_), c='k', ls='--')
ax.set_ylim([50000,250000])
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Cross-validated MSE', fontsize=20);
print("Displaying lassoCV_fig (ElasticNetCV lasso MSE path)...")
show()

print("tuned_lasso.coef_:", tuned_lasso.coef_)

#PCR and PLS Regression

pca = PCA(n_components=2)
linreg = skl.LinearRegression()
pipe = Pipeline([('pca', pca),
('linreg', linreg)])
pipe.fit(X, Y)
print("linreg.coef_ (PCA, unscaled):", pipe.named_steps['linreg'].coef_)

pipe = Pipeline([('scaler', scaler),
('pca', pca),
('linreg', linreg)])
pipe.fit(X, Y)
print("linreg.coef_ (PCA, scaled):", pipe.named_steps['linreg'].coef_)

param_grid = {'pca__n_components': range(1, 20)}
grid = skm.GridSearchCV(pipe,
param_grid,
cv=kfold,
scoring='neg_mean_squared_error')
grid.fit(X, Y)

pcr_fig, ax = subplots(figsize=(8,8))
n_comp = param_grid['pca__n_components']
ax.errorbar(n_comp,
-grid.cv_results_['mean_test_score'],
grid.cv_results_['std_test_score'] / np.sqrt(K))
ax.set_ylabel('Cross-validated MSE', fontsize=20)
ax.set_xlabel('# principal components', fontsize=20)
ax.set_xticks(n_comp[::2])
ax.set_ylim([50000,250000]);
print("Displaying pcr_fig (PCR cross-validated MSE vs # components)...")
show()

Xn = np.zeros((X.shape[0], 1))
cv_null = skm.cross_validate(linreg,Xn,Y,cv=kfold,scoring='neg_mean_squared_error')
print("Null model mean test MSE:", -cv_null['test_score'].mean())

print("PCA explained variance ratio:", pipe.named_steps['pca'].explained_variance_ratio_)

pls = PLSRegression(n_components=2,
scale=True)
pls.fit(X, Y)

param_grid = {'n_components':range(1, 20)}
grid = skm.GridSearchCV(pls,
param_grid,
cv=kfold,
scoring='neg_mean_squared_error')
grid.fit(X, Y)

pls_fig, ax = subplots(figsize=(8,8))
n_comp = param_grid['n_components']
ax.errorbar(n_comp,
-grid.cv_results_['mean_test_score'],
grid.cv_results_['std_test_score'] / np.sqrt(K))
ax.set_ylabel('Cross-validated MSE', fontsize=20)
ax.set_xlabel('# principal components', fontsize=20)
ax.set_xticks(n_comp[::2])
ax.set_ylim([50000,250000]);
print("Displaying pls_fig (PLS cross-validated MSE vs # components)...")
show()