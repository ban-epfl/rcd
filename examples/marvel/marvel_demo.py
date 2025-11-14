"""Example run of MARVEL on latent-variable graphs."""

from __future__ import annotations

import networkx as nx
import numpy as np
from tqdm import tqdm

from rcd import marvel
from rcd.utilities.ci_tests import fisher_z
from rcd.utilities.data_graph_generation import gen_er_dag_adj_mat, gen_gaussian_data
from rcd.utilities.utils import f1_score_edges


def main() -> None:
    n = 15
    p = 2 * np.log(n) / n
    rng = np.random.default_rng(2308)

    for _ in tqdm(range(5), desc="MARVEL runs"):
        seed = int(rng.integers(100_000))
        np.random.seed(seed)
        adj_mat = gen_er_dag_adj_mat(n, p)
        data_df = gen_gaussian_data(adj_mat, 10_000)

        ci_test = lambda x, y, z, data: fisher_z(x, y, z, data, significance_level=0.05)
        learned_skeleton = marvel.learn_and_get_skeleton(ci_test, data_df)

        true_skeleton = nx.from_numpy_array(adj_mat, create_using=nx.Graph)
        precision, recall, f1_score = f1_score_edges(true_skeleton, learned_skeleton, return_only_f1=False)

        print(f"Seed {seed} -> Precision {precision:.3f}, Recall {recall:.3f}, F1 {f1_score:.3f}")


if __name__ == "__main__":
    main()
