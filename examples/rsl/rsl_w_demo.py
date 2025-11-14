"""Visualization demo for RSL-W on a bounded-clique DAG."""

from __future__ import annotations

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from rcd import rsl_w
from rcd.utilities.ci_tests import fisher_z
from rcd.utilities.data_graph_generation import gen_er_dag_adj_mat, gen_gaussian_data
from rcd.utilities.utils import f1_score_edges, get_clique_number


def main() -> None:
    np.random.seed(23429)
    n = 10
    p = np.log(n) / n
    adj_mat = gen_er_dag_adj_mat(n, p)

    graph = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph).to_undirected()
    clique_number = get_clique_number(graph)
    print(f"Clique number: {clique_number}")

    data_df = gen_gaussian_data(adj_mat, 1000)

    ci_test = lambda x, y, z, data: fisher_z(x, y, z, data, significance_level=0.01)
    learned_skeleton = rsl_w.learn_and_get_skeleton(ci_test, data_df, clique_number)

    nx.draw(learned_skeleton, with_labels=True)
    plt.title("Learned skeleton", color="red")
    plt.show(block=False)
    plt.close()

    true_skeleton = nx.from_numpy_array(adj_mat, create_using=nx.Graph)
    precision, recall, f1_score = f1_score_edges(true_skeleton, learned_skeleton, return_only_f1=False)
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 score: {f1_score:.3f}")


if __name__ == "__main__":
    main()
