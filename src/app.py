from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE, chi2, SelectKBest
from matplotlib import pyplot
import numpy as np
import scipy.stats as stat
import math
import feature_selection as fs
import n_selection as ns
import knn_model as knn
import stat_test as st
import argparse as arg


def load_data():
    raw_data = open("../breast-cancer-wisconsin.data", "r")
    dataset = np.genfromtxt(raw_data, delimiter=",")
    dataset = dataset[~np.isnan(dataset).any(axis=1)]  # Removes rows with '?' - impaired data
    dataset = dataset.astype(int)
    return dataset


def main():
    parser = arg.ArgumentParser(description='knn model and tools')
    parser.add_argument('-fs', '--feature_selection', action='store_true',
                        help='Run feature ranking and multicollinearity detection')
    parser.add_argument('-ns', '--n_selection', action='store_true',
                        help='Find best neighbours amount')
    parser.add_argument('-m', '--model', action='store_true', help='Create model and test it')
    args = parser.parse_args()

    if sum([args.feature_selection, args.model, args.n_selection]) != 1:
        print('Only one argument required')
        return

    ds = load_data()

    if args.feature_selection:
        fs.chi2_ranking(ds)
        fs.multicollinearity(ds)
    elif args.n_selection:
        ns.find_n(ds)
    elif args.model:
        # ranked_features = [6, 2, 3, 8, 4, 7, 1, 5, 9]
        ranked_features = [6, 2, 8, 4, 7, 1, 5]
        n_neighbors = [5, 7, 2]
        src = knn.create_model(ds, ranked_features, n_neighbors)
        print(src)
        st.stat(src, n_neighbors)


if __name__ == '__main__':
    main()
