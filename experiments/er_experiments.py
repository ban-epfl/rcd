import os
import time
import numpy as np
import networkx as nx
import h5py as h5
from joblib import Parallel, delayed
from tqdm import tqdm

from rcd import RSLDiamondFree, RSLBoundedClique, ROLHillClimb, LMarvel, Marvel
from rcd.utilities.ci_tests import fisher_z
from rcd.utilities.data_graph_generation import gen_er_dag_adj_mat, gen_gaussian_data
from rcd.utilities.utils import f1_score_edges, find_markov_boundary_matrix

"""
Run simulations of all algorithms on random ER graphs, with n ranging from 10 to 100, with p= n^{-0.5} and p=2*log(n)/n
"""

num_procs = os.cpu_count()

if __name__ == '__main__':
    num_graphs_to_test = 10

    # set up h5 dataset
    f = h5.File('er_simulations.h5', 'w')
    n_arr = list(reversed(np.linspace(20, 50, 30, dtype=int)))
    f.attrs['n_arr'] = n_arr
    f.attrs['p'] = "n^{-0.83}"
    max_clique_num = 3

    ci_test = lambda x, y, z, data: fisher_z(x, y, z, data, significance_level=0.01)

    def run_alg_on_data(alg, data_df, find_markov_boundary_matrix_fun, true_skeleton):
        # alg.find_markov_boundary_matrix = find_markov_boundary_matrix_fun

        starting_time = time.time()

        # if alg is RSLBoundedClique, set the clique number to max_clique_num
        if isinstance(alg, RSLBoundedClique):
            learned_skeleton = alg.learn_and_get_skeleton(data_df, max_clique_num)
        else:
            learned_skeleton = alg.learn_and_get_skeleton(data_df)
        time_taken = time.time() - starting_time

        precision, recall, f1_score = f1_score_edges(true_skeleton, learned_skeleton, return_only_f1=False)
        return f1_score, time_taken

    algs_to_run = [RSLDiamondFree(ci_test), RSLBoundedClique(ci_test), LMarvel(ci_test), Marvel(ci_test)]
    alg_names = [type(alg).__name__ for alg in algs_to_run]
    f.attrs['algs'] = alg_names
    num_algs = len(algs_to_run)
    print(f"Using {num_procs} processes...")
    try:
        with Parallel(n_jobs=num_procs) as parallel:
            for n in tqdm(n_arr):
                grp = f.create_group(str(n))
                f1_score_dset = grp.create_dataset('f1_score', (num_graphs_to_test, num_algs), dtype=float)
                time_dset = grp.create_dataset('time', (num_graphs_to_test, num_algs), dtype=float)
                p = n ** (-0.83)

                ci_test_mk = lambda x, y, z, data: fisher_z(x, y, z, data, significance_level=2 / n ** 2)

                # generate random graphs and data for each graph
                adj_mat_list = parallel(delayed(gen_er_dag_adj_mat)(n, p) for _ in range(num_graphs_to_test))
                print("Generated adj mats")
                data_df_list = parallel(delayed(gen_gaussian_data)(adj_mat, 50 * n) for adj_mat in adj_mat_list)
                print("Generated data")
                true_skeleton_list = parallel(delayed(nx.from_numpy_array)(adj_mat, create_using=nx.Graph) for adj_mat in adj_mat_list)
                print("Generated true skeletons")
                mkbv_mat_list = parallel(delayed(find_markov_boundary_matrix)(data_df, ci_test_mk) for data_df in data_df_list)
                print("Generated markov boundary matrices")

                # create a list of functions that simply return the already computed markov boundary matrix if called on the entire data
                find_markov_boundary_matrix_list = [(lambda data: mkbv_mat_list[i] if len(data) == len(data_df_list[i]) else find_markov_boundary_matrix(data, ci_test_mk)) for i in range(num_graphs_to_test)]

                # run each algorithm on each dataset
                for i, alg in enumerate(algs_to_run):
                    print(f"Running {type(alg).__name__} on {n} nodes")
                    # results = parallel(delayed(run_alg_on_data)(alg, data_df_list[j], find_markov_boundary_matrix_list[j], true_skeleton_list[j]) for j in range(num_graphs_to_test))
                    results = [run_alg_on_data(alg, data_df_list[j], find_markov_boundary_matrix_list[j], true_skeleton_list[j]) for j in range(num_graphs_to_test)]
                    for j in range(num_graphs_to_test):
                        f1_score_dset[j, i] = results[j][0]
                        time_dset[j, i] = results[j][1]

    finally:
        f.close()
        print("ALL DONE")

