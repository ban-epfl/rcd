from rcd import rsl_d
from rcd import rol_hc
from rcd.utilities.ci_tests import *
from rcd.utilities.data_graph_generation import *
from rcd.utilities.utils import f1_score_edges


def test_with_perf_ci():
    """
    ROL-HC should do at least as better as RSL-D with perfect CI tests (because it is initialized with RSL-D).
    """

    # generate a random Erdos-Renyi DAG
    np.random.seed(2308)
    n = 15
    p = 0.3
    num_repeats = 10

    for _ in range(num_repeats):
        adj_mat = gen_er_dag_adj_mat(n, p)

        # generate data from the DAG
        data_df = gen_gaussian_data(adj_mat, 1000)
        data_mat = data_df.to_numpy()

        # run rsl-D
        # ci_test = lambda x, y, z, data: fisher_z(x, y, z, data, significance_level=2 / n ** 2)
        ci_test = get_perfect_ci_test(adj_mat)
        learned_skeleton = rsl_d.learn_and_get_skeleton(ci_test, data_mat)

        # compare the learned skeleton to the true skeleton
        true_skeleton = nx.from_numpy_array(adj_mat, create_using=nx.Graph)

        # compute F1 score
        _, _, rsl_f1 = f1_score_edges(true_skeleton, learned_skeleton, return_only_f1=False)
        # print(f"RSL-D F1 Score: {rsl_f1}")

        # run rol-hc
        learned_skeleton = rol_hc.learn_and_get_skeleton(ci_test, data_mat, 5, 5)

        # compute F1 score
        _, _, rol_f1 = f1_score_edges(true_skeleton, learned_skeleton, return_only_f1=False)
        # print(f"ROL-HC F1 Score: {rol_f1}")

        assert rol_f1 >= rsl_f1, "ROL-HC should have an F1 score at least as good as RSL-D!"


def test_with_data():
    """
    ROL-HC learned skeleton should ALWAYS have at most as many edges as RSL-D learned skeleton.
    """

    # generate a random Erdos-Renyi DAG
    np.random.seed(2308)
    n = 10
    p = np.log(n) / n
    num_repeats = 5

    for _ in range(num_repeats):
        adj_mat = gen_er_dag_adj_mat(n, p)

        # generate data from the DAG
        data_df = gen_gaussian_data(adj_mat, n*50)
        data_mat = data_df.to_numpy()

        # run rsl-D
        ci_test = lambda x, y, z, data: fisher_z(x, y, z, data, significance_level=1 / n ** 2)
        rsl_skeleton = rsl_d.learn_and_get_skeleton(ci_test, data_mat)

        # run rol-hc
        rol_skeleton = rol_hc.learn_and_get_skeleton(ci_test, data_mat, 5, 5)

        assert len(rol_skeleton.edges()) <= len(rsl_skeleton.edges()), "ROL-HC should have at most as many edges as RSL-D!"
