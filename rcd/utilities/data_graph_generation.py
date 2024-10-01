import numpy as np
import pandas as pd


def gen_er_dag_adj_mat(num_vars: int, edge_prob: float):
    """
    Generate an Erdos-Renyi DAG with a given number of variables and edge probability
    :param num_vars: Number of variables
    :param edge_prob: Probability of an edge between any two variables
    :return: Adjacency matrix of the generated DAG
    """
    # Generate a random upper triangular matrix
    arr = np.triu(np.random.rand(num_vars, num_vars), k=1)

    # Convert to adjacency matrix with probability p
    adj_mat = (arr > 1 - edge_prob).astype(int)

    # Generate a random permutation
    perm = np.random.permutation(num_vars)

    # Apply the permutation to rows and the corresponding columns
    adj_mat_perm = adj_mat[perm, :][:, perm]

    return adj_mat_perm.astype(bool)


def gen_gaussian_data(dag_adj_mat: np.ndarray, num_samples: int, return_numpy=False):
    """
    Generate random Gaussian samples for each variable from a given DAG
    :param dag_adj_mat: The adjacency matrix of the DAG
    :param num_samples: Number of samples to generate for each variable
    :return: A pandas dataframe with the generated samples
    """
    n = dag_adj_mat.shape[1]
    noise = np.random.normal(size=(num_samples, n)) @ np.diag(0.7 + 0.5 * np.random.rand(n))
    B = dag_adj_mat.T * ((1 + 0.5 * np.random.rand(n)) * ((-1) ** (np.random.rand(n) > 0.5)))
    D = noise @ np.linalg.pinv(np.eye(n) - B.T)

    if return_numpy:
        return D
    return pd.DataFrame(D)
