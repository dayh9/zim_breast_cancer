from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit,\
    RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import GridSearchCV


def find_n(dataset):
    # ranked_features = [6, 2, 3, 8, 4, 7, 1, 5, 9]
    ranked_features = [6, 3, 8, 4, 7, 1]

    # ranked_features = [6, 2, 3, 8]
    # ranked_features = [6, 2, 4]
    X = dataset[:, ranked_features]  # [0:i]
    y = dataset[:, 10]
    max_n = 20
    # create new a knn model
    knn = KNeighborsClassifier()

    # create a dictionary of all values we want to test for n_neighbors
    param_grid = {'n_neighbors': np.arange(1, max_n+1)}

    # use gridsearch to test all values for n_neighbors
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=19042021)
    knn_gscv = GridSearchCV(knn, param_grid, cv=rskf)

    # fit model to data
    knn_gscv.fit(X, y)

    # check top performing n_neighbors value
    print(knn_gscv.best_params_)
    print(knn_gscv.best_score_)
    rank = knn_gscv.cv_results_['rank_test_score']

    for i, r in enumerate(rank):
        print(i+1, 'NN: rank', r)
