import networkx as nx
from matplotlib import pyplot as plt

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
    n = 6
    p = n ** (-0.5)
    adj_mat = gen_er_dag_adj_mat(n, p)

    # draw graph
    nx.draw(nx.from_numpy_array(adj_mat, create_using=nx.DiGraph), with_labels=True)
    plt.show()

    # get graph clique number
    graph = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph).to_undirected()

    clique_number = get_clique_number(graph)
    print("Clique number: ", clique_number)

    # generate data from the DAG
    data_df = gen_gaussian_data(adj_mat, 1000)

    # run rsl-D
    ci_test = lambda x, y, z, data: fisher_z(x, y, z, data, significance_level=0.01)
    # ci_test = get_perfect_ci_test(adj_mat)
    rsl_d = RSLBoundedClique(ci_test, clique_number)
    # rsl_d = RSLDiamondFree(ci_rest)

    learned_skeleton = rsl_d.learn_and_get_skeleton(data_df)

    # draw learned skeleton
    nx.draw(learned_skeleton, with_labels=True)
    # make the title red
    plt.title("Learned skeleton", color='red')
    plt.show()




    # compare the learned skeleton to the true skeleton
    true_skeleton = nx.from_numpy_array(adj_mat, create_using=nx.Graph)

    # compute F1 score
    precision, recall, f1_score = f1_score_edges(true_skeleton, learned_skeleton, return_only_f1=False)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 score: ", f1_score)

