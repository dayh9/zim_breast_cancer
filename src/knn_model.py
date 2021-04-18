from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit,\
    RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score


def create_model(dataset):
    # separate the data from the target attributes
    # X = dataset[:, 1:10]

    # features to build and test model on (column 0 contains ids!)
    features = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # TODO: add feature/s selection

    X = dataset[:, features]
    y = dataset[:, 10]

    # Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=5)  # TODO: test diff n_neighbors

    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=19042021)
    scores = cross_val_score(knn, X, y, cv=rskf, scoring='accuracy')
    print("Accuracy:", scores)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
