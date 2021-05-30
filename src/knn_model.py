from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit,\
    RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np


def create_model(dataset, ranked_features, n_neighbors):
    # separate the data from the target attributes

    # ranked_features = [6, 2, 3, 8, 4, 7, 1, 5, 9]
    # ranked_features = [1, 5, 9]
    # strongly correlated features: [2, 3]

    # features to build and test model on (column 0 contains ids!)
    FEATURES = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # TODO: add feature/s selection

    # neighbors in knn model to build and test
    # n_neighbors = [4, 6, 18]

    # X = dataset[:, test_features]
    y = dataset[:, 10]

    all_scores = np.zeros((6, 10))
    best_score = {'n': 0, 'dis': 0, 'score': 0}
    s = 0

    print('Means:')
    for n in n_neighbors:
        for p in [1, 2]:
            # Dataset with chosen features
            X = dataset[:, ranked_features]

            # Create KNN Classifier (manhattan_distance for p=1, euclidean_distance for p=2)
            knn = KNeighborsClassifier(n_neighbors=n, p=p)

            # 5 times repeated 2 fold cross-validation
            rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=19042021)
            scores = cross_val_score(knn, X, y, cv=rskf, scoring='balanced_accuracy')

            all_scores[s] = scores
            s += 1
            print(n, 'NN p=', p, 'score: ', scores.mean())
            if scores.mean() > best_score['score']:
                best_score['n'] = n
                if p == 1:
                    best_score['dis'] = 'man'
                else:
                    best_score['dis'] = 'euc'
                best_score['score'] = scores.mean()

    print('\nBest mean score ', best_score, '\n')
    return all_scores

