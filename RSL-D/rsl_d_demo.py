from rsl_d import RSLDiamondFree
from utilities.ci_tests import *
from utilities.data_graph_generation import *
from utilities.utils import f1_score_edges

if __name__ == '__main__':
    """
    In this example, we first generate an Erdos-Renyi DAG with n=50 nodes and edge probability p=n^{-0.82}. 
    Notice that by setting p as such, we are guaranteeing with high probability that the generated graph is 
    diamond-free, which is a requirement for RSL-D. Then, we generate 1000 samples per variable from this DAG and run 
    RSL-D on it, comparing the learned skeleton to the true skeleton. We use the Pearson correlation coefficient as 
    the CI test.
    """

    # generate a random DAG
    np.random.seed(2308)
    n = 50
    p = n ** (-0.85)
    adj_mat = gen_er_dag_adj_mat(n, p)

    # generate data from the DAG
    data_df = gen_gaussian_data(adj_mat, 1000)

    # run RSL-D
    ci_rest = lambda x, y, z, data: fisher_z(x, y, z, data, significance_level=0.005)
    rsl_d = RSLDiamondFree(data_df, ci_rest)
    learned_skeleton = rsl_d.learn_and_get_skeleton()

    # compare the learned skeleton to the true skeleton
    true_skeleton = nx.from_numpy_array(adj_mat, create_using=nx.Graph)

    # compute F1 score
    precision, recall, f1_score = f1_score_edges(true_skeleton, learned_skeleton, return_only_f1=False)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 score: ", f1_score)

