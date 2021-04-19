from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit,\
    RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score


def create_model(dataset):
    # separate the data from the target attributes

    RANKED_FEATURES = [6, 2, 3, 8, 4, 7, 1, 5, 9]
    # strongly correlated features: [2, 3]

    # features to build and test model on (column 0 contains ids!)
    FEATURES = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # TODO: add feature/s selection


    # neighbors in knn model to build and test
    n_neighbors = [3, 5, 7]

    # X = dataset[:, test_features]
    y = dataset[:, 10]

    for n in n_neighbors:
        for p in [1, 2]:
            for i in range(1, 9):
                X = dataset[:, RANKED_FEATURES[0:i]]
                # Create KNN Classifier (manhattan_distance for p=1, euclidean_distance for p=2)
                knn = KNeighborsClassifier(n_neighbors=n, p=p)

                rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=19042021)
                scores = cross_val_score(knn, X, y, cv=rskf, scoring='accuracy')

                # print("Accuracy:", scores)
                if p == 1:
                    print("%0.3f accuracy for manhattan_distance, %d neighbors and %d best features"
                          % (scores.mean(), n, i))
                else:
                    print("%0.4f accuracy for euclidean_distance, %d neighbors and %d best features"
                          % (scores.mean(), n, i))
            print('\n')
