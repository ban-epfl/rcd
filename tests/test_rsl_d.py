from __future__ import annotations

import networkx as nx
import numpy as np

from rcd import rsl_d
from rcd.utilities.ci_tests import fisher_z, get_perfect_ci_test
from rcd.utilities.data_graph_generation import gen_er_dag_adj_mat, gen_gaussian_data
from rcd.utilities.utils import compute_mb, f1_score_edges, sanitize_data


def test_with_data():
    """RSL-D should recover a single diamond-free graph with high accuracy."""

    # generate a random Erdos-Renyi DAG
    np.random.seed(2308)
    n = 10
    p = n ** (-0.85)
    adj_mat = gen_er_dag_adj_mat(n, p)

    # generate data from the DAG
    data_df = gen_gaussian_data(adj_mat, 1000)

    # run rsl-D
    ci_test = lambda x, y, z, d: fisher_z(x, y, z, d, significance_level=2 / n**2)
    learned_skeleton = rsl_d.learn_and_get_skeleton(ci_test, data_df)

    # compare the learned skeleton to the true skeleton
    true_skeleton = nx.from_numpy_array(adj_mat, create_using=nx.Graph)

    # compute F1 score
    _, _, f1_score = f1_score_edges(true_skeleton, learned_skeleton, return_only_f1=False)
    assert f1_score == 1, "F1 score should be 1!"


def test_with_perfect_ci():
    """Perfect CI oracles should make RSL-D exact across many random graphs."""
    n = 20
    p = n ** (-0.85)

    num_graphs_to_test = 100
    np.random.seed(2308)
    for i in range(num_graphs_to_test):
        # generate a random Erdos-Renyi DAG
        adj_mat = gen_er_dag_adj_mat(n, p)

        # generate data from the DAG (unused as we use a perfect CI test)
        data_df = gen_gaussian_data(adj_mat, 1)

        # run rsl-D
        ci_test = get_perfect_ci_test(adj_mat)
        find_markov_boundary_matrix = lambda d: compute_mb(sanitize_data(d), ci_test)
        learned_skeleton = rsl_d.learn_and_get_skeleton(
            ci_test,
            data_df,
            find_markov_boundary_matrix,
        )

        # compare the learned skeleton to the true skeleton
        true_skeleton = nx.from_numpy_array(adj_mat, create_using=nx.Graph)

        # compute F1 score
        _, _, f1_score = f1_score_edges(true_skeleton, learned_skeleton, return_only_f1=False)
        assert f1_score == 1, f"F1 score of {f1_score} for graph {i} should be 1!"
