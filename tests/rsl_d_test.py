# import unittest
from pgmpy.base import DAG
from tqdm import tqdm

from rcd.rsl.rsl_d import RSLDiamondFree

import networkx as nx
import pgmpy.base
from matplotlib import pyplot as plt
from pgmpy.estimators.CITests import *


def is_d_separated(x, y, z, graph: pgmpy.base.DAG):
    """
    Check whether x and y are d-separated given z in the graph
    :param x: First variable
    :param y: Second variable
    :param z: Set of variables
    :param graph: A networkx graph
    :return: True if x and y are d-separated given z in the graph, False otherwise
    """

    # use pgmpy
    return not graph.is_dconnected(x, y, z)







# class RSLDiamondFreeTest(unittest.TestCase):
#     pass
def create_dag_from_adj_matrix(adj_matrix):
    """
    Create a DAG from an adjacency matrix
    :param adj_matrix: Adjacency matrix
    :return: A pgmpy DAG
    """
    n_nodes = adj_matrix.shape[0]

    nodes = list(range(n_nodes))
    edges = nx.convert_matrix.from_numpy_array(
        np.triu(adj_matrix, k=1), create_using=nx.DiGraph
    ).edges()

    dag = DAG(edges)
    dag.add_nodes_from(nodes)
    return dag


def evaluate_rsl_d_with_perfect_ci_test(graph_mat):
    ci_test = lambda x, y, z, data: pearsonr(x, y, z, data, significance_level=0.01)

    dag = nx.from_numpy_array(graph_mat, create_using=DAG)
    perfect_ci_test = lambda x, y, z, data: is_d_separated(x, y, z, dag)

    # create an empty dataframe with as many columns as variables in the graph
    data_mat, _ = generate_data(graph_mat, 50 * graph_mat.shape[0])

    # convert data_mat to a pandas dataframe
    data_df = pd.DataFrame(data_mat)

    rsl_d = RSLDiamondFree(data_df, perfect_ci_test)
    skeleton = rsl_d.run_algorithm()

    edges = set(nx.from_numpy_array(graph_mat).edges())
    edges_est = set(skeleton.edges())

    # compute F1 score
    precision = len(edges.intersection(edges_est)) / len(edges_est)
    recall = len(edges.intersection(edges_est)) / len(edges)
    f1_score = 2 * precision * recall / (precision + recall)

    return f1_score


def gen_graphs_and_evaluate_rsl_d():
    # generate 100 random graphs
    num_graphs = 20
    n = 100
    # p = np.log(n) / n * 1.5
    p = n ** (-0.72)
    graphs = [generate_graph_erdos(n, p) for _ in range(num_graphs)]

    # evaluate rsl-D on each of the graphs
    f1_scores = [evaluate_rsl_d_with_perfect_ci_test(graph) for graph in tqdm(graphs)]

    return np.asarray(f1_scores)


if __name__ == '__main__':
    # unittest.main()
    np.random.seed(42323)
    f1_scores = gen_graphs_and_evaluate_rsl_d()
    print("Avg F1: ", np.mean(f1_scores))

    erdos_graph_adj = generate_graph_erdos(5, 0.5)
    # visualize the graph as a directed graph
    graph = nx.from_numpy_array(erdos_graph_adj, create_using=DAG)

    graph_for_plot = nx.from_numpy_array(erdos_graph_adj, create_using=nx.DiGraph)
    nx.draw(graph_for_plot, with_labels=True)
    plt.show()

    data_mat, _ = generate_data(erdos_graph_adj, 10000)
    ci_test = lambda x, y, z, data: pearsonr(x, y, z, data, significance_level=0.05)
    perfect_ci_test = lambda x, y, z, data: not graph.is_dconnected(x, y, z)

    # convert data_mat to a pandas dataframe
    data_df = pd.DataFrame(data_mat)
    rsl_d = RSLDiamondFree(data_df, perfect_ci_test)
    skeleton = rsl_d.run_algorithm()

    # convert graph to bidirected
    edges = set(nx.Graph(graph).edges())
    edges_est = set(skeleton.edges())

    # compute F1 score
    precision = len(edges.intersection(edges_est)) / len(edges_est)
    recall = len(edges.intersection(edges_est)) / len(edges)
    f1_score = 2 * precision * recall / (precision + recall)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {f1_score}")

    # visualize the skeleton
    # label the plot
    plt.figure()
    nx.draw(skeleton, with_labels=True)
    plt.show()
