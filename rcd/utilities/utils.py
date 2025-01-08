from typing import List, Callable

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

REMOVABLE_NOT_FOUND = -1


def sanitize_data(data):
    """
    This function takes as input a data argument and checks if it is a pandas DataFrame or a numpy array.
    If it's a DataFrame, it converts it to a numpy matrix and returns it.
    If it's a numpy array, it confirms that it is a matrix and returns it.

    Args:
    data (pandas.DataFrame or numpy.ndarray): Input data

    Returns:
    numpy.ndarray: Matrix form of the input data
    """
    if isinstance(data, pd.DataFrame):
        # Convert DataFrame to numpy matrix
        return data.values
    elif isinstance(data, np.ndarray):
        # Check if the numpy array is 2-dimensional (matrix)
        if data.ndim == 2:
            return data
        else:
            raise ValueError("Input numpy array is not a matrix (2-dimensional).")
    else:
        raise TypeError("Input must be a pandas DataFrame or a numpy array.")

def get_clique_number(graph: nx.Graph):
    maximum_clique = max(nx.find_cliques(graph), key=len)
    clique_number = len(maximum_clique)
    return clique_number


def f1_score_edges(true_graph: nx.Graph, est_graph: nx.Graph, return_only_f1=True) -> float | tuple[
    float, float, float]:
    """
    Compute the F1 score of the estimated graph with respect to the true graph, using edges as the unit of comparison.
    :param true_graph: The true graph
    :param est_graph: The estimated/predicted graph
    :param return_only_f1: If True, return only the F1 score. If False, return precision, recall, and F1 score.
    :return:
    """
    edges = set(true_graph.edges())
    edges_est = set(est_graph.edges())

    # compute F1 score
    precision = len(edges.intersection(edges_est)) / len(edges_est) if len(edges_est) > 0 else 0
    recall = len(edges.intersection(edges_est)) / len(edges) if len(edges) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    if return_only_f1:
        return f1_score
    else:
        return precision, recall, f1_score


def sort_vars_by_mkb_size(markov_boundary_matrix: np.ndarray, var_idx_arr: np.ndarray) -> np.ndarray:
    # sort variables by the size of their Markov boundary
    mb_size = np.sum(markov_boundary_matrix, axis=1)
    sort_indices = np.argsort(mb_size)
    sorted_var_idx = np.asarray(var_idx_arr, dtype=int)[sort_indices]
    return sorted_var_idx


def compute_mb(data: np.ndarray, ci_test: Callable[[int, int, List[int], np.ndarray], bool]) -> np.ndarray:
    """
    Computes the Markov boundary matrix for all variables.

    Args:
        data (np.ndarray): Data matrix where each column is a variable.
        ci_test (Callable[[int, int, List[int], np.ndarray], bool]):
            Conditional independence test to use. It takes the indices of two variables
            and a list of variable indices as the conditioning set. It returns True if the two
            variables are independent given the conditioning set, and False otherwise.

    Returns:
        np.ndarray: A numpy array containing the Markov boundary (symmetric) matrix, where element ij indicates whether
                    variable i is in the Markov boundary of j.
    """
    num_vars = data.shape[1]
    markov_boundary_matrix = np.zeros((num_vars, num_vars), dtype=bool)

    for i in range(num_vars - 1):  # -1 because no need to check last variable
        for j in range(i + 1, num_vars):
            # Check whether variable i and j are independent of each other given the rest of the variables
            cond_set = list(set(range(num_vars)) - {i, j})
            if not ci_test(i, j, cond_set, data):
                markov_boundary_matrix[i, j] = True
                markov_boundary_matrix[j, i] = True

    return markov_boundary_matrix


def compute_mb_gaussian(data: np.ndarray, sig_level=None) -> np.ndarray:
    """
    Computes the Markov boundary matrix for all variables.
    :param data: Dataframe where each column is a variable
    :param ci_test: Conditional independence test to use
    :param significance_level: Significance level for the conditional independence test
    :return: A numpy array containing the Markov boundary (symmetric) matrix, where element ij indicates whether
    variable i is in the Markov boundary of j
    """
    num_samples, n = data.shape
    crr = np.corrcoef(data, rowvar=False)
    prec = np.linalg.pinv(crr)

    norm_vec = np.sqrt(np.diag(prec))
    mb_mat = np.abs(prec / norm_vec[:, None] / norm_vec[None, :])

    sig_level = 1 / n ** 2 if sig_level is None else sig_level

    thresh = np.tanh(stats.norm.ppf(1 - sig_level / 2) / np.sqrt(num_samples - n - 1))

    mb_mat = mb_mat > thresh

    # set diagonal to 0
    np.fill_diagonal(mb_mat, 0)

    return mb_mat


def update_markov_boundary_matrix(markov_boundary_matrix: np.ndarray,
                                  data_included_ci_test: Callable[[int, int, List[int]], bool],
                                  var: int, var_neighbors: np.ndarray,
                                  is_diamond_free: bool = False,
                                  skip_check: np.ndarray = None):
    """
    Update the Markov boundary matrix after removing a variable.

    Args:
        markov_boundary_matrix (np.ndarray): The Markov boundary matrix.
        skip_check (np.ndarray): The skip check array.
        data_included_ci_test (Callable[[int, int, List[int]], bool]): The conditional independence test to use.
        var (int): The variable to remove.
        var_neighbors (np.ndarray): 1D numpy array containing the indices of the neighbors of var.
        is_diamond_free (bool, optional): Whether the graph is diamond-free. Defaults to False.
    """
    var_markov_boundary = np.flatnonzero(markov_boundary_matrix[var])

    # Remove var from Markov boundaries
    markov_boundary_matrix[var_markov_boundary, var] = False
    markov_boundary_matrix[var, var_markov_boundary] = False

    if skip_check is not None:
        skip_check[var_markov_boundary] = False

    if is_diamond_free:
        if len(var_markov_boundary) > len(var_neighbors):
            # Sufficient condition for diamond-free graphs
            return

    # Find nodes whose co-parent status changes after removing var
    # Only remove Y from Markov boundary of Z iff var is their ONLY common child and they are NOT neighbors
    for ne_idx_y in range(len(var_neighbors) - 1): # -1 because no need to check last variable and also symmetry
        for ne_idx_z in range(ne_idx_y + 1, len(var_neighbors)):
            var_y = var_neighbors[ne_idx_y]
            var_z = var_neighbors[ne_idx_z]

            # determine whether the markov boundary of var_y or var_z is smaller, and use the smaller one as cond_set
            var_y_markov_boundary = np.flatnonzero(markov_boundary_matrix[var_y])
            var_z_markov_boundary = np.flatnonzero(markov_boundary_matrix[var_z])
            if np.sum(markov_boundary_matrix[var_y]) < np.sum(markov_boundary_matrix[var_z]):
                cond_set = list(set(var_y_markov_boundary) - {var_z})
            else:
                cond_set = list(set(var_z_markov_boundary) - {var_y})

            if data_included_ci_test(var_y, var_z, cond_set):
                # Y and Z are co-parents and thus NOT neighbors
                markov_boundary_matrix[var_y, var_z] = False
                markov_boundary_matrix[var_z, var_y] = False

                if skip_check is not None:
                    skip_check[var_y] = False
                    skip_check[var_z] = False
