from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE, chi2, SelectKBest
from matplotlib import pyplot
import numpy as np
import scipy.stats as stat
import math
import feature_selection as fs
import knn_model as knn


def load_data():
    raw_data = open("../breast-cancer-wisconsin.data", "r")
    dataset = np.genfromtxt(raw_data, delimiter=",")
    dataset = dataset[~np.isnan(dataset).any(axis=1)]  # Removes rows with '?' - impaired data
    dataset = dataset.astype(int)
    return dataset


if __name__ == '__main__':
    ds = load_data()
    # fs.chi2_ranking(ds)
    # fs.multicollinearity(ds)
    knn.create_model(ds)
