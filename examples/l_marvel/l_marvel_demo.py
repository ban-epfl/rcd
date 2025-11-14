"""Example run of L-MARVEL on a synthetic graph."""

from __future__ import annotations

import networkx as nx
import numpy as np
from tqdm import tqdm

from rcd import l_marvel
from rcd.utilities.ci_tests import fisher_z
from rcd.utilities.data_graph_generation import gen_er_dag_adj_mat, gen_gaussian_data
from rcd.utilities.utils import f1_score_edges


def main() -> None:
    n = 20
    p = 0.1
    num_rep = 3

    rng = np.random.default_rng(2308)
    for _ in tqdm(range(num_rep), desc="L-MARVEL runs"):
        seed = int(rng.integers(100_000))
        np.random.seed(seed)
        adj_mat = gen_er_dag_adj_mat(n, p)
        data_df = gen_gaussian_data(adj_mat, 20 * n)

        ci_test = lambda x, y, z, data: fisher_z(x, y, z, data, significance_level=1 / n**2)
        learned_skeleton = l_marvel.learn_and_get_skeleton(ci_test, data_df)

        true_skeleton = nx.from_numpy_array(adj_mat, create_using=nx.Graph)
        precision, recall, f1_score = f1_score_edges(true_skeleton, learned_skeleton, return_only_f1=False)

        print(f"Seed {seed} -> Precision {precision:.3f}, Recall {recall:.3f}, F1 {f1_score:.3f}")


if __name__ == "__main__":
    main()
