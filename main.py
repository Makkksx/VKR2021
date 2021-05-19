import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder

import experiments
import os


def miss_the_data(df, percentage_of_missing=10):
    gen = np.random.RandomState(0)
    missing = gen.rand(*df.shape) < (percentage_of_missing / 100)
    return pd.DataFrame(np.where(missing, np.nan, df))


def main_func(dataset, dataset_name):
    # Преобразование в числовые данные
    cols = dataset.select_dtypes(include='O').columns.tolist()
    le = LabelEncoder()
    le.fit(dataset[cols].values.flatten())
    dataset[cols] = dataset[cols].apply(le.fit_transform)
    # Создание датасетов с разным количеством пропусков
    dataset10 = miss_the_data(dataset, 10)
    dataset30 = miss_the_data(dataset, 30)
    dataset50 = miss_the_data(dataset, 50)
    score10 = pd.DataFrame()
    score10['name'] = ['mean', 'median', 'Decision Tree', 'K-nearest neighbors', 'Linear Regression', 'K-means',
                       'EM algorithm', 'Extra Trees', 'Random Forest', 'ANN'
                       ]
    score10 = score10.set_index(score10.name)
    score10 = score10.drop(columns='name')
    score10['r2_score'] = 0
    score10['time'] = 0
    score10['parameters'] = ''
    score30 = score10.copy()
    score50 = score10.copy()
    print("10")
    experiments.all_experiments(dataset10, dataset, score10, dataset_name, 10)
    print("30")
    experiments.all_experiments(dataset30, dataset, score30, dataset_name, 30)
    print("50")
    experiments.all_experiments(dataset50, dataset, score50, dataset_name, 50)

    if not os.path.exists('.\\results'):
        os.mkdir('.\\results')

    score10.to_csv('.\\results\\score10_{}.csv'.format(dataset_name), sep=";")
    score30.to_csv('.\\results\\score30_{}.csv'.format(dataset_name), sep=";")
    score50.to_csv('.\\results\\score50_{}.csv'.format(dataset_name), sep=";")
    x = [10, 30, 50]
    plt.figure(figsize=(8, 5))
    cmap = plt.get_cmap('jet_r')
    color = [cmap(float(i) / score10.shape[0]) for i in range(score10.shape[0])]
    for method in score10.index:
        plt.plot(x, [score10.loc[method, 'r2_score'], score30.loc[method, 'r2_score'], score50.loc[method, 'r2_score']],
                 label=method, lw=1.5, c=color.pop())
    plt.title(dataset_name)
    plt.xlabel("Missing value fraction")
    plt.xlim([10, 50])
    plt.ylabel("r2-score")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig('.\\results\\results_{}.png'.format(dataset_name))
    # plt.show()
    # print(score10)
    # print(score30)
    # print(score50)


# Чтение данных
abalone = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data', header=None)
abalone = abalone.iloc[:, :-1]
boston, _ = datasets.load_boston(return_X_y=True)
boston = pd.DataFrame(boston)
california, _ = datasets.fetch_california_housing(return_X_y=True)
california = pd.DataFrame(california)
cancer, _ = datasets.load_breast_cancer(return_X_y=True)
cancer = pd.DataFrame(cancer)

main_func(abalone, "Abalone")  # shape=(4177, 8), n=33416
main_func(boston, "Boston")  # shape=(506, 13), n=6578
main_func(california, "California")  # shape=(20640, 8), n=165120
main_func(cancer, "Cancer")  # shape=(569, 30), n=17070
os.system('shutdown -s')
