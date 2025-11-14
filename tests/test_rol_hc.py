from __future__ import annotations

import networkx as nx
import numpy as np

from rcd import rol_hc, rsl_d
from rcd.utilities.ci_tests import get_perfect_ci_test
from rcd.utilities.data_graph_generation import gen_er_dag_adj_mat, gen_gaussian_data
from rcd.utilities.utils import f1_score_edges, sanitize_data


def test_with_perf_ci():
    """ROL-HC should never underperform RSL-D when CI oracle is perfect."""

    # generate a random Erdos-Renyi DAG
    np.random.seed(2308)
    n = 15
    p = 0.3
    num_repeats = 10

    for _ in range(num_repeats):
        adj_mat = gen_er_dag_adj_mat(n, p)

        # generate data from the DAG
        data_mat = sanitize_data(gen_gaussian_data(adj_mat, 1000))

        # run rsl-D
        ci_test = get_perfect_ci_test(adj_mat)
        learned_skeleton = rsl_d.learn_and_get_skeleton(ci_test, data_mat)

        # compare the learned skeleton to the true skeleton
        true_skeleton = nx.from_numpy_array(adj_mat, create_using=nx.Graph)

        # compute F1 score
        _, _, rsl_f1 = f1_score_edges(true_skeleton, learned_skeleton, return_only_f1=False)

        # run rol-hc
        learned_skeleton = rol_hc.learn_and_get_skeleton(ci_test, data_mat, 5, 5)

        # compute F1 score
        _, _, rol_f1 = f1_score_edges(true_skeleton, learned_skeleton, return_only_f1=False)

        assert rol_f1 >= rsl_f1, "ROL-HC should have an F1 score at least as good as RSL-D!"


def test_with_data():
    """ROL-HC skeletons should not introduce more edges than RSL-D."""

    # generate a random Erdos-Renyi DAG
    np.random.seed(2308)
    n = 10
    p = np.log(n) / n
    num_repeats = 5

    for _ in range(num_repeats):
        adj_mat = gen_er_dag_adj_mat(n, p)

        # generate data from the DAG
        data_mat = sanitize_data(gen_gaussian_data(adj_mat, n * 50))

        # run rsl-D
        ci_test = get_perfect_ci_test(adj_mat)
        rsl_skeleton = rsl_d.learn_and_get_skeleton(ci_test, data_mat)

        # run rol-hc
        rol_skeleton = rol_hc.learn_and_get_skeleton(ci_test, data_mat, 5, 5)

        assert len(rol_skeleton.edges()) <= len(rsl_skeleton.edges()), (
            "ROL-HC should have at most as many edges as RSL-D!"
        )
