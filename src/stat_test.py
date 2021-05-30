from scipy.stats import ttest_ind
import numpy as np
from tabulate import tabulate


def stat(scores, n_neighbors):
    alfa = .05
    size = 6
    t_statistic = np.zeros((size, size))
    p_value = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            t_statistic[i, j], p_value[i, j] = ttest_ind(scores[i], scores[j])
    print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

    headers = [str(n_neighbors[0]) + "NNman", str(n_neighbors[0]) + "NNeuc",
               str(n_neighbors[1]) + "NNman", str(n_neighbors[1]) + "NNeuc",
               str(n_neighbors[2]) + "NNman", str(n_neighbors[2]) + "NNeuc"]
    names_column = np.array([[str(n_neighbors[0]) + "NNman"], [str(n_neighbors[0]) + "NNeuc"],
                             [str(n_neighbors[1]) + "NNman"], [str(n_neighbors[1]) + "NNeuc"],
                             [str(n_neighbors[2]) + "NNman"], [str(n_neighbors[2]) + "NNeuc"]])
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

    advantage = np.zeros((size, size))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    print("Advantage:\n", advantage_table)

    significance = np.zeros((size, size))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    print("Statistical significance (alpha = 0.05):\n", significance_table)

    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    print("Statistically significantly better:\n", stat_better_table)