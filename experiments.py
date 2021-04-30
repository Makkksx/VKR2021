import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from scipy.stats import randint
from sklearn.metrics import mean_squared_error
import time
import datawig as dw

import utils


def experiment_mean(df, df_full, score):
    start_time = time.time()
    for strategy in ('mean', 'median'):
        imp = SimpleImputer(strategy=strategy)
        imp.fit(df)
        df_filled = pd.DataFrame(imp.transform(df))
        score.loc[strategy, 'r2_score'] = r2_score(df_full, df_filled)
        score.loc[strategy, 'RMSE'] = mean_squared_error(df_full, df_filled, squared=False)
        score.loc[strategy, 'time'] = time.time() - start_time


def experiment_DecisionTree(df, df_full, score, n_iter=50):
    tree_parameters = {'estimator__max_depth': range(3, 100, 3), 'estimator__min_samples_split': range(2, 20, 3),
                       'estimator__min_samples_leaf': range(1, 20, 3)}
    start_time = time.time()
    imp = RandomizedSearchCV(utils.CustomImputer(DecisionTreeRegressor()), tree_parameters, n_iter=n_iter,
                             cv=3, return_train_score=False, n_jobs=-1, random_state=0)
    imp.fit(df, df_full)
    df_filled = pd.DataFrame(imp.transform(df))
    score.loc['DecisionTree', 'r2_score'] = r2_score(df_full, df_filled)
    score.loc['DecisionTree', 'RMSE'] = mean_squared_error(df_full, df_filled, squared=False)
    score.loc['DecisionTree', 'time'] = time.time() - start_time
    score.loc['DecisionTree', 'parameters'] = str(imp.best_params_)


def experiment_LinearRegression(df, df_full, score):
    start_time = time.time()
    imp = IterativeImputer(estimator=LinearRegression(), random_state=0, max_iter=10)
    imp.fit(df)
    df_filled = pd.DataFrame(imp.transform(df))
    score.loc['LinearRegression', 'r2_score'] = r2_score(df_full, df_filled)
    score.loc['LinearRegression', 'RMSE'] = mean_squared_error(df_full, df_filled, squared=False)
    score.loc['LinearRegression', 'time'] = time.time() - start_time


def experiment_randomized_search(df, df_full, score, impute_estimator, parameters, n_iter=5):
    start_time = time.time()
    imp = RandomizedSearchCV(impute_estimator, parameters, n_iter=n_iter,
                             cv=3, return_train_score=False, n_jobs=-1, random_state=0)  # n_candidates=20
    imp.fit(df, df_full)
    df_filled = pd.DataFrame(imp.transform(df))
    score.loc[impute_estimator.__class__.__name__, 'r2_score'] = r2_score(df_full, df_filled)
    score.loc[impute_estimator.__class__.__name__, 'RMSE'] = mean_squared_error(df_full, df_filled, squared=False)
    score.loc[impute_estimator.__class__.__name__, 'time'] = time.time() - start_time
    score.loc[impute_estimator.__class__.__name__, 'parameters'] = str(imp.best_params_)


def experiment_KNN(df, df_full, score, parameters=None):
    if parameters is None:
        parameters = {'n_neighbors': range(5, 50, 2), 'weights': ['uniform', 'distance']}
    experiment_randomized_search(df, df_full, score, utils.KNN(), parameters, n_iter=20)


def experiment_EM(df, df_full, score, parameters=None):
    if parameters is None:
        parameters = {'num_iters': range(1, 11)}
    experiment_randomized_search(df, df_full, score, utils.EM(), parameters, n_iter=10)


def experiment_Kmeans(df, df_full, score, parameters=None):
    if parameters is None:
        parameters = {'n_clusters': range(1, 50, 2)}
    experiment_randomized_search(df, df_full, score, utils.Kmeans(), parameters, n_iter=20)


def experiment_halving_random_search(df, df_full, score, impute_estimator, parameters):
    start_time = time.time()
    imp = HalvingRandomSearchCV(utils.CustomImputer(impute_estimator), parameters,
                                resource='estimator__n_estimators', max_resources=200,
                                cv=3, min_resources=1, return_train_score=False, n_jobs=-1, random_state=0,
                                factor=3)
    imp.fit(df, df_full)
    df_filled = pd.DataFrame(imp.transform(df))
    score.loc[impute_estimator.__class__.__name__, 'r2_score'] = r2_score(df_full, df_filled)
    score.loc[impute_estimator.__class__.__name__, 'RMSE'] = mean_squared_error(df_full, df_filled, squared=False)
    score.loc[impute_estimator.__class__.__name__, 'time'] = time.time() - start_time
    score.loc[impute_estimator.__class__.__name__, 'parameters'] = str(imp.best_params_)


def experiment_RandomForest(df, df_full, score, parameters=None):
    if parameters is None:
        parameters = {'estimator__max_depth': randint(3, 50), 'estimator__min_samples_split': randint(2, 20),
                      'estimator__min_samples_leaf': randint(1, 20)}
    experiment_halving_random_search(df, df_full, score, RandomForestRegressor(), parameters)


def experiment_ExtraTrees(df, df_full, score, parameters=None):
    if parameters is None:
        parameters = {'estimator__max_depth': randint(3, 50), 'estimator__min_samples_split': randint(2, 20),
                      'estimator__min_samples_leaf': randint(1, 20)}
    experiment_halving_random_search(df, df_full, score, ExtraTreesRegressor(), parameters)


def experiment_ANN(df, df_full, score):
    start_time = time.time()
    df_filled = df.copy()
    df_filled.columns = df_filled.columns.astype(str)
    dw.SimpleImputer.complete(df_filled,
                              inplace=True,
                              verbose=0,
                              num_epochs=100,
                              output_path='./dw'
                              )
    score.loc['ANN', 'r2_score'] = r2_score(df_full, df_filled)
    score.loc['ANN', 'RMSE'] = mean_squared_error(df_full, df_filled, squared=False)
    score.loc['ANN', 'time'] = time.time() - start_time


def all_experiments(df, df_full, score):
    print("Среднее, медиана и мода")
    experiment_mean(df, df_full, score)

    print("Дерево решений")
    experiment_DecisionTree(df, df_full, score)

    print("Линейная регрессия")
    experiment_LinearRegression(df, df_full, score)

    print("Метод k-ближайших соседей")
    experiment_KNN(df, df_full, score)

    print("EM-алгоритм")
    experiment_EM(df, df_full, score)

    print("Kmeans")
    experiment_Kmeans(df, df_full, score)

    print("Случайный лес")
    experiment_RandomForest(df, df_full, score)

    print("Дополнительные деревья")
    experiment_ExtraTrees(df, df_full, score)

    print("Нейронные сети")
    experiment_ANN(df, df_full, score)
