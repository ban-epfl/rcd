from typing import List

import networkx as nx
import numpy as np
import pandas as pd

REMOVABLE_NOT_FOUND = -1


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
    precision = len(edges.intersection(edges_est)) / len(edges_est)
    recall = len(edges.intersection(edges_est)) / len(edges)
    f1_score = 2 * precision * recall / (precision + recall)

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


def find_markov_boundary_matrix(data: pd.DataFrame, ci_test) -> np.ndarray:
    """
    Computes the Markov boundary matrix for all variables.
    :param data: Dataframe where each column is a variable
    :param ci_test: Conditional independence test to use
    :return: A numpy array containing the Markov boundary (symmetric) matrix, where element ij indicates whether
    variable i is in the Markov boundary of j
    """

    num_vars = len(data.columns)
    var_name_set = set(data.columns)
    markov_boundary_matrix = np.zeros((num_vars, num_vars), dtype=bool)

    for i in range(num_vars - 1):  # -1 because no need to check last variable
        var_name = data.columns[i]
        for j in range(i + 1, num_vars):
            var_name2 = data.columns[j]
            # check whether var_name and var_name2 are independent of each other given the rest of the variables
            cond_set = list(var_name_set - {var_name, var_name2})
            if not ci_test(var_name, var_name2, cond_set, data):
                markov_boundary_matrix[i, j] = 1
                markov_boundary_matrix[j, i] = 1

    return markov_boundary_matrix


def update_markov_boundary_matrix(markov_boundary_matrix: np.ndarray, skip_check: np.ndarray, var_names: List[str],
                                  data_included_ci_test, var: int, var_neighbors: np.ndarray,
                                  is_diamond_free: bool = False):
    """
    Update the Markov boundary matrix after removing a variable.
    :param markov_boundary_matrix: The Markov boundary matrix.
    :param skip_check: The skip check array.
    :param var_names: List of variable names.
    :param data_included_ci_test: The conditional independence test to use.
    :param var: The variable to remove.
    :param var_neighbors: 1D numpy array containing the indices of the neighbors of var_idx.
    :param is_diamond_free: Whether the graph is diamond-free.
    """

    var_markov_boundary = np.flatnonzero(markov_boundary_matrix[var])

    # for every variable in the markov boundary of var_idx, remove it from the markov boundary and update flag
    markov_boundary_matrix[var_markov_boundary, var] = 0
    markov_boundary_matrix[var, var_markov_boundary] = 0
    skip_check[var_markov_boundary] = False

    if is_diamond_free:
        if len(var_markov_boundary) > len(var_neighbors):
            # Sufficient condition for diamond-free graphs
            return

    # find nodes whose co-parent status changes after removing var
    # we only remove Y from markov boundary of Z iff X is their ONLY common child and they are NOT neighbors
    for ne_idx_y in range(len(var_neighbors) - 1):  # -1 because no need to check last variable and also symmetry
        for ne_idx_z in range(ne_idx_y + 1, len(var_neighbors)):
            var_y = var_neighbors[ne_idx_y]
            var_z = var_neighbors[ne_idx_z]
            var_y_name = var_names[var_y]
            var_z_name = var_names[var_z]

            # determine whether the markov boundary of var_y or var_z is smaller, and use the smaller one as cond_set
            var_y_markov_boundary = np.flatnonzero(markov_boundary_matrix[var_y])
            var_z_markov_boundary = np.flatnonzero(markov_boundary_matrix[var_z])
            if np.sum(markov_boundary_matrix[var_y]) < np.sum(markov_boundary_matrix[var_z]):
                cond_set = [var_names[idx] for idx in set(var_y_markov_boundary) - {var_z}]
            else:
                cond_set = [var_names[idx] for idx in set(var_z_markov_boundary) - {var_y}]

            if data_included_ci_test(var_y_name, var_z_name, cond_set):
                # we know that Y and Z are co-parents and thus NOT neighbors
                markov_boundary_matrix[var_y, var_z] = 0
                markov_boundary_matrix[var_z, var_y] = 0
                skip_check[var_y] = False
                skip_check[var_z] = False
