from tqdm import tqdm
from rcd import l_marvel
from rcd.utilities.ci_tests import *
from rcd.utilities.data_graph_generation import *
from rcd.utilities.utils import f1_score_edges

if __name__ == '__main__':
    """
    In this example, we first generate an Erdos-Renyi DAG. Then, we generate 1000 samples per variable from this 
    DAG and run L-MARVEL on it, comparing the learned skeleton to the true skeleton. We use the Pearson correlation 
    coefficient as the CI test.
    """

    # generate a random Erdos-Renyi DAG
    # np.random.seed(2308)
    n = 20
    p = 0.1
    num_rep = 1


    rng = np.random.default_rng(2308)
    for i in tqdm(range(num_rep)):
        # seed = rng.integers(100000)
        seed = 10925
        np.random.seed(seed)
        adj_mat = gen_er_dag_adj_mat(n, p)

        # plot the graph
        # nx.draw(nx.from_numpy_array(adj_mat, create_using=nx.DiGraph), with_labels=True)
        # plt.show()

        # generate data from the DAG
        data_df = gen_gaussian_data(adj_mat, 20 * n)

        ci_test = lambda x, y, z, data: fisher_z(x, y, z, data, significance_level=1 / n ** 2)
        ci_test_mk = lambda x, y, z, data: fisher_z(x, y, z, data, significance_level=2 / n ** 2)

        # ci_test = get_perfect_ci_test(adj_mat)
        # find_markov_boundary_matrix_fun = lambda data: find_markov_boundary_matrix(data, ci_test)

        # run l-marvel
        learned_skeleton = l_marvel.learn_and_get_skeleton(ci_test, data_df)

        # compare the learned skeleton to the true skeleton
        true_skeleton = nx.from_numpy_array(adj_mat, create_using=nx.Graph)

        # compute F1 score
        precision, recall, f1_score = f1_score_edges(true_skeleton, learned_skeleton, return_only_f1=False)

        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 score: ", f1_score)
        print("seed: ", seed)

