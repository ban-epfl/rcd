import time

import networkx as nx
from matplotlib import pyplot as plt

from rcd.rol.rol_hc import ROLHillClimb
from rcd.rsl.rsl_d import RSLDiamondFree
from rcd.rsl.rsl_w import RSLBoundedClique
from rcd.utilities.ci_tests import *
from rcd.utilities.data_graph_generation import *
from rcd.utilities.utils import f1_score_edges


def get_clique_number(graph: nx.Graph):
    maximum_clique = max(nx.find_cliques(graph), key=len)
    clique_number = len(maximum_clique)
    return clique_number


if __name__ == '__main__':
    """
    In this example, we first generate an Erdos-Renyi DAG with n=50 nodes and edge probability p=n^{-0.85}. 
    Notice that by setting p as such, we are guaranteeing with high probability that the generated graph is 
    diamond-free, which is a requirement for rsl-D. Then, we generate 1000 samples per variable from this DAG and run 
    rsl-D on it, comparing the learned skeleton to the true skeleton. We use the Pearson correlation coefficient as 
    the CI test.
    """

    # generate a random DAG
    np.random.seed(23429)
    n = 20
    p = np.log(n) / n * 1.5
    adj_mat = gen_er_dag_adj_mat(n, p)

    # get graph clique number
    graph = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph).to_undirected()

    # generate data from the DAG
    data_df = gen_gaussian_data(adj_mat, 2000)

    # run rsl-D
    ci_test = lambda x, y, z, data: fisher_z(x, y, z, data, significance_level=0.01)


    ci_test = get_perfect_ci_test(adj_mat)

    rsl_d = RSLDiamondFree(ci_test)
    starting_time = time.time()
    learned_skeleton_rsl_d = rsl_d.learn_and_get_skeleton(data_df)
    print("Time taken for rsl-D: ", time.time() - starting_time)

    rol_hc = ROLHillClimb(ci_test, max_iters=10, max_swaps=5)
    starting_time = time.time()
    learned_skeleton_rol_hc = rol_hc.learn_and_get_skeleton(data_df)
    print("Time taken for rol-hc: ", time.time() - starting_time)

    # compare the learned skeleton to the true skeleton
    true_skeleton = nx.from_numpy_array(adj_mat, create_using=nx.Graph)

    # compute F1 score
    precision, recall, f1_score = f1_score_edges(true_skeleton, learned_skeleton_rsl_d, return_only_f1=False)
    print("F1 score for rsl-D: ", f1_score)
    print("Precision for rsl-D: ", precision)
    print("Recall for rsl-D: ", recall)

    precision, recall, f1_score = f1_score_edges(true_skeleton, learned_skeleton_rol_hc, return_only_f1=False)
    print("\nF1 score for rol-hc: ", f1_score)
    print("Precision for rol-hc: ", precision)
    print("Recall for rol-hc: ", recall)
