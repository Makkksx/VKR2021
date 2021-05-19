import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.experimental import enable_halving_search_cv  # noqa
import time
import datawig as dw
import matplotlib.pyplot as plt
import os
import utils


def experiment_mean(df, df_full, score):
    start_time = time.time()
    for strategy in ('mean', 'median'):
        imp = SimpleImputer(strategy=strategy)
        imp.fit(df)
        df_filled = pd.DataFrame(imp.transform(df))
        score.loc[strategy, 'r2_score'] = r2_score(df_full, df_filled)
        score.loc[strategy, 'time'] = time.time() - start_time


def experiment_LinearRegression(df, df_full, score):
    start_time = time.time()
    imp = IterativeImputer(estimator=LinearRegression(), random_state=0, max_iter=10)
    imp.fit(df)
    df_filled = pd.DataFrame(imp.transform(df))
    score.loc['Linear Regression', 'r2_score'] = r2_score(df_full, df_filled)
    score.loc['Linear Regression', 'time'] = time.time() - start_time


def experiment_grid_search(df, df_full, score, impute_estimator, class_name, parameters):
    start_time = time.time()
    imp = GridSearchCV(impute_estimator, parameters, cv=3, return_train_score=True, n_jobs=-1)
    imp.fit(df, df_full)
    imp.transform(df)
    score.loc[class_name, 'r2_score'] = imp.best_score_
    score.loc[class_name, 'time'] = time.time() - start_time
    score.loc[class_name, 'parameters'] = list(imp.best_params_.values())[0]
    res = pd.DataFrame(imp.cv_results_).loc[:, ['params', 'mean_test_score']]
    res.params = res.params.apply(lambda x: x.get(list(parameters)[0]))
    return res


def experiment_KNN(df, df_full, score, parameters=None):
    if parameters is None:
        parameters = {'n_neighbors': range(2, 50, 2)}
    return experiment_grid_search(df, df_full, score, utils.KNN(), "K-nearest neighbors", parameters)


def experiment_EM(df, df_full, score, parameters=None):
    if parameters is None:
        parameters = {'num_iters': range(1, 11)}
    return experiment_grid_search(df, df_full, score, utils.EM(), 'EM algorithm', parameters)


def experiment_Kmeans(df, df_full, score, parameters=None):
    if parameters is None:
        parameters = {'n_clusters': range(2, 50, 2)}
    return experiment_grid_search(df, df_full, score, utils.Kmeans(), 'K-means', parameters)


def experiment_DecisionTree(df, df_full, score, parameters=None):
    if parameters is None:
        parameters = {'estimator__max_depth': range(5, 101, 5)}
    return experiment_grid_search(df, df_full, score, utils.CustomImputer(DecisionTreeRegressor()), 'Decision Tree',
                                  parameters)


def experiment_RandomForest(df, df_full, score, parameters=None):
    if parameters is None:
        parameters = {'estimator__n_estimators': range(10, 101, 10)}
    max_depth = score.parameters['Decision Tree']
    return experiment_grid_search(df, df_full, score, utils.CustomImputer(RandomForestRegressor(max_depth=max_depth)),
                                  'Random Forest', parameters)


def experiment_ExtraTrees(df, df_full, score, parameters=None):
    if parameters is None:
        parameters = {'estimator__n_estimators': range(10, 101, 10)}
    max_depth = score.parameters['Decision Tree']
    return experiment_grid_search(df, df_full, score, utils.CustomImputer(ExtraTreesRegressor(max_depth=max_depth)),
                                  'Extra Trees', parameters)


def experiment_ANN(df, df_full, score):
    start_time = time.time()
    df_filled = df.copy()
    df_filled.columns = df_filled.columns.astype(str)
    dw.SimpleImputer.complete(df_filled,
                              inplace=True,
                              verbose=0,
                              num_epochs=100,
                              output_path='./dw',
                              iterations=2
                              )
    score.loc['ANN', 'r2_score'] = r2_score(df_full, df_filled)
    score.loc['ANN', 'time'] = time.time() - start_time


def plot_results(res, method_name, dataset_name, feature, percent):
    res.plot(x='params', label=None)
    plt.xlabel(feature)
    plt.ylabel("r2-score")
    plt.title(method_name + ' {}%'.format(percent))
    plt.legend().remove()
    path = os.path.join('.\\results', dataset_name, method_name)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    path = os.path.join(path, 'results_{}.png'.format(percent))
    plt.savefig(path)


def all_experiments(df, df_full, score, dataset_name, percent):
    print("Среднее, медиана и мода")
    experiment_mean(df, df_full, score)

    print("Дерево решений")
    plot_results(experiment_DecisionTree(df, df_full, score), 'Decision Tree', dataset_name, 'Max depth', percent)

    print("Линейная регрессия")
    experiment_LinearRegression(df, df_full, score)

    print("Метод k-ближайших соседей")
    plot_results(experiment_KNN(df, df_full, score), 'K-nearest neighbors', dataset_name, 'K neighbors', percent)

    print("EM-алгоритм")
    plot_results(experiment_EM(df, df_full, score), 'EM algorithm', dataset_name, 'Iterations', percent)

    print("K-means")
    plot_results(experiment_Kmeans(df, df_full, score), 'K-means', dataset_name, 'K clusters', percent)

    print("Случайный лес")
    plot_results(experiment_RandomForest(df, df_full, score), 'Random Forest', dataset_name, 'N estimators', percent)

    print("Дополнительные деревья")
    plot_results(experiment_ExtraTrees(df, df_full, score), 'Extra Trees', dataset_name, 'N estimators', percent)

    print("Нейронные сети")
    experiment_ANN(df, df_full, score)
