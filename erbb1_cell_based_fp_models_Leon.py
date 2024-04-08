import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_val_score, train_test_split, RepeatedKFold
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score
from pycytominer.cyto_utils.util import get_pairwise_correlation
from sklearn.model_selection import learning_curve

def create_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(10, 3))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

fp_data = pd.read_csv("cell-based_fingerprints.csv")
ic50_data = pd.read_csv("erbb1_cellbased_neglog10_ic50.csv")
X = fp_data.values
y = ic50_data.iloc[:, -1].values

vt = VarianceThreshold(threshold=0.01).set_output(transform="pandas")
X = vt.fit_transform(X)

corr_list = get_pairwise_correlation(X)[1]

corr_list = corr_list[abs(corr_list['correlation']) >= 0.90]
values_in_both_columns = corr_list['pair_a'].isin(corr_list['pair_b'])
features_remove = corr_list['pair_b'].values
X = X.drop(columns=features_remove)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=23)

models_dict = {'Random Forest': RandomForestRegressor(random_state=23, n_estimators=100),
               'K-Nearest Neighbour': KNeighborsRegressor(n_neighbors=5),
               'Multilayer Perceptron': MLPRegressor(random_state=23, max_iter=500, hidden_layer_sizes=(30, 30, 30)),
               'AdaBoost': AdaBoostRegressor(random_state=23, n_estimators=100),
               'Decision Tree': DecisionTreeRegressor(random_state=23),
               'Gradient Boosting': GradientBoostingRegressor(n_estimators=100),
               'Support Vector Machine': SVR(C=1.0, epsilon=0.2, gamma='auto')
}

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=23)

scoring = {'R2': 'r2',
           'MSE': 'neg_mean_squared_error',
           'RMSE': 'neg_root_mean_squared_error',
           'MAE': 'neg_mean_absolute_error'}

for m_name, model in models_dict.items():
    if model == KNeighborsRegressor(n_neighbors=5):
        n_jobs = 1
    else:
        n_jobs = 4
    print("Running: ", str(m_name))
    for s_name, score in scoring.items():
        results = cross_val_score(model, X_train, y_train, cv=cv, scoring=score)
        print(s_name, ":\t%.3f (%.3f)" % (np.absolute(results.mean()), np.absolute(results.std())))
    title = ("Learning Curves " + str(m_name))
    create_learning_curve(model, title, X_train, y_train, cv=cv, n_jobs=n_jobs)
    plt.show()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    print('R-squared on test set:', test_r2)
