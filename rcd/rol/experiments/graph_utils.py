import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import networkx as nx
import math
from scipy import stats
from scipy.stats import norm


def load_adjacency_matrix(path):
    mat = scipy.io.loadmat(path)
    return mat['A']


def save_numpy_to_mat_matrix(path, array):
    scipy.io.savemat(path, {"A": array})


def generate_data(path, number_of_samples, ):
    # A = load_adjacency_matrix(path)
    A = np.load(path + '/DAG' + '.npy')
    n_variables = A.shape[1]
    N = np.matmul(np.random.rand(number_of_samples, n_variables), np.diag(0.7 + 0.5 * np.random.rand(n_variables)))
    AA = (1 + 0.5 * np.random.rand(n_variables)) * ((-1) ** (np.random.rand(n_variables) > 0.5))
    AA = A * AA
    D = np.matmul(N, np.linalg.pinv(np.eye(n_variables) - AA))

    return D, A


def load_graph_data(path, sample_number, set_number, erdos=False):
    if erdos:
        A = np.load(path + '/DAG'+'_' + str(set_number) + '.npy')
    else:
        A = np.load(path + '/DAG' + '.npy')
    D = np.load(path + '/data_' + str(sample_number) + '_' + str(set_number) + '.npy')

    return D, A


def load_partial_graph_data(path, sample_number, set_number):
    mat = scipy.io.loadmat(path + '/MAG.mat')
    selected_vars = mat['rem']
    A = mat['MAG']
    D = np.squeeze(np.load(path + '/data_' + str(sample_number) + '_' + str(set_number) + '.npy')[:, selected_vars-1])
    print(A.shape)
    print(D.shape)
    return D, A


def plot_graph(graph, path):
    rows, cols = np.where(graph == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.DiGraph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=100, with_labels=True)
    plt.savefig(path)
    plt.clf()


def get_stat(A, B):
    A = (A + A.T)
    A[A > 0] = 1
    A[A <= 0] = 0
    edges_of_A = np.sum(A) / 2
    B = (B + B.T)
    B[B > 0] = 1
    B[B <= 0] = 0
    edges_of_B = np.sum(B) / 2
    skeleton_errors = B - A
    extra_edges = np.sum(skeleton_errors == 1) / 2
    missing_edges = np.sum(skeleton_errors == -1) / 2
    precision = np.sum(A * B) / np.sum(B)
    recall = np.sum(A * B) / np.sum(A)
    skeleton_F1_score = 2 * precision * recall / (precision + recall)
    shd = missing_edges + extra_edges
    print("edges_of_A: ", edges_of_A)
    print("edges_of_B: ", edges_of_B)
    print("extra_edges: ", extra_edges)
    print("missing_edges: ", missing_edges)
    print("precision: ", precision)
    print("recall: ", recall)
    print("skeleton_F1_score:", skeleton_F1_score)
    return edges_of_A, edges_of_B, extra_edges, missing_edges, \
           precision, recall, skeleton_F1_score, shd


# generate_data(10, "alarm.mat")
def real_markov_boundary(graph):
    mb = graph.copy()
    for i in range(mb.shape[0]):
        child_ids = np.where(graph[i] == 1)[0]
        for j in child_ids:
            parents_of_the_child = np.where(graph[:, j] == 1)[0]
            mb[i][parents_of_the_child] = 1
            mb[i][i] = 0
    mb = mb + graph.T
    mb[mb > 0] = 1
    mb[mb <= 0] = 0
    return mb




def CI_test1(x, y, s, alpha, data):
    n = data.shape[0]
    k = len(s)
    if k == 0:
        r = np.corrcoef(data[:, [x, y]].T)[0][1]
    else:
        sub_index = [x, y]
        sub_index.extend(s)
        sub_corr = np.corrcoef(data[:, sub_index].T)
        # inverse matrix
        try:
            PM = np.linalg.inv(sub_corr)
        except np.linalg.LinAlgError:
            PM = np.linalg.pinv(sub_corr)
        r = -1 * PM[0, 1] / math.sqrt(abs(PM[0, 0] * PM[1, 1]))
    cut_at = 0.99999
    r = min(cut_at, max(-1 * cut_at, r))  # make r between -1 and 1

    # Fisher’s z-transform
    res = math.sqrt(n - k - 3) * .5 * math.log1p((2 * r) / (1 - r))
    p_value = 2 * (1 - stats.norm.cdf(abs(res)))

    return p_value >= alpha


def CI_test2(x, y, s, alpha, data):
    # inverse of a cdf of normal distribution
    c = norm.ppf(1 - alpha / 2)
    # get the proposed columns of data
    truncated_data = data[:, np.array([x, y] + s)]
    corrcoef_matrix = np.corrcoef(truncated_data, rowvar=False)
    if truncated_data.shape[1] == 1:
        corrcoef_matrix = np.array([[corrcoef_matrix]])
    perecion_matrix = np.linalg.inv(corrcoef_matrix)
    partial_corrolation = perecion_matrix[0, 1] / (perecion_matrix[0, 0] * perecion_matrix[1, 1]) ** 0.5
    threshold = c / (truncated_data.shape[0] - len(s) - 3) ** 0.5
    return abs(partial_corrolation) <= threshold
