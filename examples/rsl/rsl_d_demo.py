"""Command-line demo showcasing RSL-D on a random diamond-free DAG."""

from __future__ import annotations

import time

import networkx as nx
import numpy as np

from rcd import rsl_d
from rcd.utilities.ci_tests import fisher_z
from rcd.utilities.data_graph_generation import gen_er_dag_adj_mat, gen_gaussian_data
from rcd.utilities.utils import f1_score_edges


def main() -> None:
    """Run RSL-D on a synthetic diamond-free graph and print metrics."""

    np.random.seed(2308)
    n = 100
    p = np.log(n) / n
    adj_mat = gen_er_dag_adj_mat(n, p)
    data_df = gen_gaussian_data(adj_mat, 20 * n)

    cond_set_sizes: list[int] = []

    def ci_test(x_idx: int, y_idx: int, cond_set: list[int], data) -> bool:
        cond_set_sizes.append(len(cond_set))
        return fisher_z(x_idx, y_idx, cond_set, data, significance_level=0.01)

    start = time.process_time()
    learned_skeleton = rsl_d.learn_and_get_skeleton(ci_test, data_df)
    duration = time.process_time() - start

    true_skeleton = nx.from_numpy_array(adj_mat, create_using=nx.Graph)
    precision, recall, f1_score = f1_score_edges(
        true_skeleton,
        learned_skeleton,
        return_only_f1=False,
    )

    print("RSL-D Demo Results")
    print("===================")
    print(f"Time (s): {duration:.2f}")
    print(f"Total CI calls: {len(cond_set_sizes)}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 score: {f1_score:.3f}")


if __name__ == "__main__":
    main()
