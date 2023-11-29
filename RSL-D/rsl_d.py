from typing import List

import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.estimators import CITests

REMOVED_VAR = -1


def find_markov_boundary(var_name: str, data: pd.DataFrame, ci_test) -> List[str]:
    """

    :param var_name: Name of the target variable
    :param data: Dataframe where each column is a variable
    :param ci_test: Conditional independence test to use
    :return: List containing the names of the variables in the Markov boundary
    """

    markov_boundary = []
    other_vars = [col_name for col_name in data.columns if col_name != var_name]
    for var in other_vars:
        # check whether our variable (var_name) is independent of var given the rest of the variables
        cond_set = list(set(other_vars) - {var_name, var})
        if not ci_test(var_name, var, cond_set, data):
            markov_boundary.append(var)

    return markov_boundary


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
    markov_boundary_matrix = np.zeros((num_vars, num_vars))
    for i, var_name in enumerate(data.columns):
        for j, var_name2 in enumerate(data.columns[i:]):
            # check whether var_name and var_name2 are independent of each other given the rest of the variables
            cond_set = list(var_name_set - {var_name, var_name2})
            if not ci_test(var_name, var_name2, cond_set, data):
                markov_boundary_matrix[i, j] = 1
                markov_boundary_matrix[j, i] = 1

    return markov_boundary_matrix


class RSL:
    def __init__(self, data: pd.DataFrame, ci_test):
        self.num_vars = len(data.columns)
        self.data = data
        self.var_names = data.columns
        self.ci_test = ci_test
        self.var_name_set = set(data.columns)
        self.flag_arr = np.zeros(self.num_vars, dtype=bool)
        self.var_idx_set = set(range(self.num_vars))
        self.var_name_idx_dict = {var_name: i for i, var_name in enumerate(data.columns)}

        self.markov_boundary_matrix = None

    def run_algorithm(self) -> nx.Graph:
        """
        Run the RSL algorithm on the data and return the learned skeleton graph
        :return: A networkx graph representing the learned skeleton
        """

        # initialize graph
        skeleton = nx.Graph()
        skeleton.add_nodes_from(self.var_names)

        self.markov_boundary_matrix = find_markov_boundary_matrix(self.data, self.ci_test)

        var_idx_left_set = self.var_idx_set.copy()
        while var_idx_left_set:
            # find a removable variable
            removable_var_idx = self.find_removable(list(var_idx_left_set))

            # find the neighbors of the removable variable
            neighbors = self.find_neighborhood(removable_var_idx)

            # update the markov boundary matrix
            self.update_markov_boundary_matrix(removable_var_idx, neighbors)

            # add edges between the removable variable and its neighbors
            for neighbor_idx in neighbors:
                skeleton.add_edge(self.var_names[removable_var_idx], self.var_names[neighbor_idx])

            # remove the removable variable from the set of variables left
            var_idx_left_set.remove(removable_var_idx)

        return skeleton

    def find_neighborhood(self, var_idx: int) -> np.ndarray:
        """
        Find the neighborhood of a variable using Lemma 4 of the RSL paper.
        :param var_idx: Index of the variable in the data
        :return: 1D numpy array containing the indices of the variables in the neighborhood
        """

        var_mk_arr = self.markov_boundary_matrix[var_idx]
        var_mk_idxs = np.nonzero(var_mk_arr != REMOVED_VAR)

        # loop through markov boundary matrix row corresponding to var_name
        # use Lemma 4 of RSL paper: var2_idx is Y and var3_idx is Z. cond_set is Mb(X) - {Y, Z}
        # at first, assume all variables are neighbors
        neighbors = np.copy(var_mk_arr)

        for var2_idx in var_mk_idxs:
            for var3_idx in np.arange(var2_idx + 1, len(var_mk_arr)):
                var2_name = self.var_names[var2_idx]
                var3_name = self.var_names[var3_idx]
                cond_set = [self.var_names[idx] for idx in set(var_mk_idxs) - {var2_idx, var3_idx}]

                if self.ci_test(var2_name, var3_name, cond_set, self.data):
                    # we know that var2 is a co-parent and thus NOT a neighbor
                    neighbors[var2_idx] = REMOVED_VAR
                    break

        # remove all variables that are not neighbors
        neighbors = neighbors[neighbors != REMOVED_VAR]
        return neighbors

    def is_removable(self, var_idx: int) -> bool:
        """
        Check whether a variable is removable using Lemma 3 of the RSL paper.
        :param var_idx:
        :return: True if the variable is removable, False otherwise
        """

        var_mk_arr = self.markov_boundary_matrix[var_idx]
        var_mk_idxs = np.nonzero(var_mk_arr != REMOVED_VAR)

        # use Lemma 3 of RSL paper: var2_idx is Y and var3_idx is Z. cond_set is Mb(X) + {X} - {Y, Z}
        for var2_idx in var_mk_idxs:
            for var3_idx in np.arange(var2_idx + 1, len(var_mk_arr)):
                var2_name = self.var_names[var2_idx]
                var3_name = self.var_names[var3_idx]
                cond_set = [self.var_names[idx] for idx in set(var_mk_idxs) - {var2_idx, var3_idx}] + [
                    self.var_names[var_idx]]

                if self.ci_test(var2_name, var3_name, cond_set, self.data):
                    return False
        return True

    def find_removable(self, var_idx_list: List[int]) -> int:
        """
        Find a removable variable in the given list of variables.
        :param var_idx_list: List of variable indices
        :return: Index of the removable variable
        """

        # sort variables by the size of their Markov boundary
        mb_size = np.sum(self.markov_boundary_matrix[var_idx_list], axis=1)
        sorted_var_idx = np.argsort(mb_size)

        for var_idx in sorted_var_idx:
            if not self.flag_arr[var_idx] and self.is_removable(var_idx):
                self.flag_arr[var_idx] = False  # TODO ask Ehsan about this and also final return statement
                return var_idx

    def update_markov_boundary_matrix(self, var_idx: int, var_neighbors: np.ndarray):
        """
        Update the Markov boundary matrix after removing a variable.
        :param var_idx: Index of the variable to remove
        :param var_neighbors: 1D numpy array containing the indices of the neighbors of var_idx
        """

        var_markov_boundary = np.nonzero(self.markov_boundary_matrix[var_idx])

        # for every variable in the markov boundary of var_idx, remove it from the markov boundary and update flag
        for mb_var_idx in np.nonzero(self.markov_boundary_matrix[var_idx] != REMOVED_VAR):  # TODO use indexing instead
            self.markov_boundary_matrix[mb_var_idx, var_idx] = REMOVED_VAR
            self.markov_boundary_matrix[var_idx, mb_var_idx] = REMOVED_VAR
            self.flag_arr[mb_var_idx] = True

        if len(var_markov_boundary) > len(var_neighbors):
            # Sufficient condition for diamond-free graphs
            return
