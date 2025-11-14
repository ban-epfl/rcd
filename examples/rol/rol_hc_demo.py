"""Compare RSL-D and ROL-HC on a synthetic dataset."""

from __future__ import annotations

import time

import networkx as nx
import numpy as np

from rcd import rol_hc, rsl_d
from rcd.utilities.ci_tests import fisher_z
from rcd.utilities.data_graph_generation import gen_er_dag_adj_mat, gen_gaussian_data
from rcd.utilities.utils import f1_score_edges


def main() -> None:
    np.random.seed(4)
    n = 15
    p = 0.2
    adj_mat = gen_er_dag_adj_mat(n, p)
    data_df = gen_gaussian_data(adj_mat, 10 * n)

    ci_test = lambda x, y, z, data: fisher_z(x, y, z, data, significance_level=0.01)

    start = time.time()
    learned_skeleton_rsl_d = rsl_d.learn_and_get_skeleton(ci_test, data_df)
    rsl_duration = time.time() - start

    start = time.process_time()
    learned_skeleton_rol_hc = rol_hc.learn_and_get_skeleton(
        ci_test,
        data_df,
        max_iters=12,
        max_swaps=5,
    )
    rol_duration = time.process_time() - start

    true_skeleton = nx.from_numpy_array(adj_mat, create_using=nx.Graph)

    def summarize(label: str, learned: nx.Graph, runtime: float) -> None:
        precision, recall, f1_score = f1_score_edges(true_skeleton, learned, return_only_f1=False)
        print(f"{label} -> time {runtime:.2f}s, precision {precision:.3f}, recall {recall:.3f}, F1 {f1_score:.3f}")

    print("ROL Demo Results")
    print("================")
    summarize("RSL-D", learned_skeleton_rsl_d, rsl_duration)
    summarize("ROL-HC", learned_skeleton_rol_hc, rol_duration)


if __name__ == "__main__":
    main()
