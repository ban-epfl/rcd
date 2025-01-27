from matplotlib import pyplot as plt
from tqdm import tqdm
from rcd import marvel
from rcd.utilities.ci_tests import *
from rcd.utilities.data_graph_generation import *
from rcd.utilities.utils import f1_score_edges

if __name__ == '__main__':
    """
    In this example, we first generate an Erdos-Renyi DAG with n=50 nodes and edge probability p=n^{-0.85}. 
    Notice that by setting p as such, we are guaranteeing with high probability that the generated graph is 
    diamond-free, which is a requirement for rsl-D. Then, we generate 1000 samples per variable from this DAG and run 
    rsl-D on it, comparing the learned skeleton to the true skeleton. We use the Pearson correlation coefficient as 
    the CI test.
    """

    # generate a random Erdos-Renyi DAG
    # np.random.seed(2308)
    n = 15
    p = 2 * np.log(n) / n



    rng = np.random.default_rng(2308)
    for i in tqdm(range(1000)):
        seed = rng.integers(100000)
        # seed = 73205
        np.random.seed(seed)
        # print(seed)
        adj_mat = gen_er_dag_adj_mat(n, p)

        # plot the graph
        # nx.draw(nx.from_numpy_array(adj_mat, create_using=nx.DiGraph), with_labels=True)
        # plt.show()

        # generate data from the DAG
        data_df = gen_gaussian_data(adj_mat, 10000)

        perfect_ci_test = get_perfect_ci_test(adj_mat)
        ci_test = lambda x, y, z, data: fisher_z(x, y, z, data, significance_level=0.05)

        # run l-marvel
        learned_skeleton = marvel.learn_and_get_skeleton(ci_test, data_df)

        # compare the learned skeleton to the true skeleton
        true_skeleton = nx.from_numpy_array(adj_mat, create_using=nx.Graph)

        # compute F1 score
        precision, recall, f1_score = f1_score_edges(true_skeleton, learned_skeleton, return_only_f1=False)

        if f1_score != 1:
            print("Precision: ", precision)
            print("Recall: ", recall)
            print("F1 score: ", f1_score)
            print("seed: ", seed)

