from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE, chi2, SelectKBest
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from scipy.cluster import hierarchy

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import numpy as np


def chi2_ranking(dataset):
    # separate the data from the target attributes
    X = dataset[:, 1:10]
    y = dataset[:, 10]

    ch2 = SelectKBest(chi2, k=6)
    ch2.fit_transform(X, y)
    ch2.transform(X)

    importance = ch2.scores_
    plt.bar([i + 1 for i in range(len(importance))], importance)
    plt.show()

    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))


def multicollinearity(dataset):
    # separate the data from the target attributes
    X = dataset[:, 1:10]
    y = dataset[:, 10]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    corr = spearmanr(X).correlation
    corr_linkage = hierarchy.ward(corr)
    dendro = hierarchy.dendrogram(
        corr_linkage, labels=[*range(1, 10)], ax=ax1, leaf_rotation=0
    )
    dendro_idx = np.arange(0, len(dendro['ivl']))

    ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro['ivl'], rotation='horizontal')
    ax2.set_yticklabels(dendro['ivl'])
    fig.tight_layout()
    plt.show()
