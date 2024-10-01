from rcd import rsl_d
from rcd.utilities.ci_tests import *
from rcd.utilities.data_graph_generation import *
from rcd.utilities.utils import f1_score_edges


def test_with_data():
    """
    Test RSL-D on a single diamond-free graph. We expect it to get a perfect F1 score with sufficient data.
    """

    # generate a random Erdos-Renyi DAG
    np.random.seed(2308)
    n = 10
    p = n ** (-0.85)
    adj_mat = gen_er_dag_adj_mat(n, p)

    # generate data from the DAG
    data_df = gen_gaussian_data(adj_mat, 1000)

    # run rsl-D
    ci_test = lambda x, y, z, data: fisher_z(x, y, z, data, significance_level=2 / n ** 2)
    learned_skeleton = rsl_d.learn_and_get_skeleton(ci_test, data_df)

    # compare the learned skeleton to the true skeleton
    true_skeleton = nx.from_numpy_array(adj_mat, create_using=nx.Graph)

    # compute F1 score
    precision, recall, f1_score = f1_score_edges(true_skeleton, learned_skeleton, return_only_f1=False)
    assert f1_score == 1, "F1 score should be 1!"
    print("RSL-D passed the first test!")


def test_with_perfect_ci():
    """
    Test RSL-D on 100 random diamond-free graphs. We expect it to get a perfect F1 score with perfect CI tests.
    """
    n = 20
    p = n ** (-0.85)

    num_graphs_to_test = 100
    np.random.seed(2308)
    for i in range(num_graphs_to_test):
        # generate a random Erdos-Renyi DAG
        adj_mat = gen_er_dag_adj_mat(n, p)

        # generate data from the DAG (unused as we use a perfect CI test)
        data_df = gen_gaussian_data(adj_mat, 1)

        # run rsl-D
        ci_test = get_perfect_ci_test(adj_mat)
        learned_skeleton = rsl_d.learn_and_get_skeleton(ci_test, data_df)

        # compare the learned skeleton to the true skeleton
        true_skeleton = nx.from_numpy_array(adj_mat, create_using=nx.Graph)

        # compute F1 score
        precision, recall, f1_score = f1_score_edges(true_skeleton, learned_skeleton, return_only_f1=False)
        assert f1_score == 1, "F1 score of " + str(f1_score) + " for graph " + str(i) + " should be 1!"
    print("RSL-D passed the second test!")
