from typing import List
import networkx as nx
import numpy as np
import pandas as pd

"""
This file contains the base class implementation for the rsl-D and rsl-W algorithms for learning diamond-free 
graphs and graphs with a bounded clique number from i.i.d. samples. The class is initialized with a conditional 
independence test function, which determines whether two variables are independent given another set of variables, 
using the data provided. For examples of possible conditional independence tests, see utilities/ci_tests.py. For 
details on how to write a custom CI test function, look at the constructor (init function) of the RSLBase class 
below.

The class has a learn_and_get_skeleton function that takes in a Pandas DataFrame of data, where the ith column
contains samples from the ith variable, and returns a networkx graph representing the learned skeleton.
"""


# currently unused
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
    markov_boundary_matrix = np.zeros((num_vars, num_vars), dtype=int)

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


class RSLBase:
    def __init__(self, ci_test):
        """
        Initialize the rsl algorithm with the data and conditional independence test to use.
        :param ci_test: Conditional independence test to use that takes in the names of two variables and a list of
        variable names as the conditioning set, and returns True if the two variables are independent given the
        conditioning set, and False otherwise. The signature of the function should be:
        ci_test(var_name1: str, var_name2: str, cond_set: List[str], data: pd.DataFrame) -> bool
        """
        self.num_vars = None
        self.data = None
        self.var_names = None
        self.ci_test = ci_test

        # we use a flag array to keep track of which variables need to be checked for removal (i.e., we check if true)
        self.flag_arr = None
        self.var_idx_set = None
        self.markov_boundary_matrix = None
        self.learned_skeleton = None

    def reset_fields(self, data: pd.DataFrame):
        self.num_vars = len(data.columns)
        self.data = data
        self.var_names = data.columns

        self.flag_arr = np.ones(self.num_vars, dtype=bool)
        self.var_idx_set = set(range(self.num_vars))
        self.markov_boundary_matrix = None
        self.learned_skeleton = None

    def has_alg_run(self):
        return self.learned_skeleton is not None

    def learn_and_get_skeleton(self, data: pd.DataFrame) -> nx.Graph:
        """
        Run the rsl algorithm on the data to learn and return the learned skeleton graph
        :return: A networkx graph representing the learned skeleton
        """
        self.reset_fields(data)

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

        self.learned_skeleton = skeleton
        return skeleton

    def find_neighborhood(self, var_idx: int) -> np.ndarray:
        """
        Find the neighborhood of a variable using Lemma 4 of the rsl paper.
        :param var_idx: Index of the variable in the data
        :return: 1D numpy array containing the indices of the variables in the neighborhood
        """

        raise NotImplementedError()

    def is_removable(self, var_idx: int) -> bool:
        """
        Check whether a variable is removable using Lemma 3 of the rsl paper.
        :param var_idx:
        :return: True if the variable is removable, False otherwise
        """

        raise NotImplementedError()

    def find_removable(self, var_idx_list: List[int]) -> int:
        """
        Find a removable variable in the given list of variables.
        :param var_idx_list: List of variable indices
        :return: Index of the removable variable
        """

        # sort variables by the size of their Markov boundary
        mb_size = np.sum(self.markov_boundary_matrix[var_idx_list], axis=1)
        sort_indices = np.argsort(mb_size)
        sorted_var_idx = np.asarray(var_idx_list, dtype=int)[sort_indices]

        for var_idx in sorted_var_idx:
            if self.flag_arr[var_idx]:
                self.flag_arr[var_idx] = False
                if self.is_removable(var_idx):
                    return var_idx

        # if no removable found, return the first variable
        return sorted_var_idx[0]

    def update_markov_boundary_matrix(self, var_idx: int, var_neighbors: np.ndarray):
        """
        Update the Markov boundary matrix after removing a variable.
        :param var_idx: Index of the variable to remove
        :param var_neighbors: 1D numpy array containing the indices of the neighbors of var_idx
        """

        var_markov_boundary = np.flatnonzero(self.markov_boundary_matrix[var_idx])

        # for every variable in the markov boundary of var_idx, remove it from the markov boundary and update flag
        for mb_var_idx in np.flatnonzero(self.markov_boundary_matrix[var_idx]):  # TODO use indexing instead
            self.markov_boundary_matrix[mb_var_idx, var_idx] = 0
            self.markov_boundary_matrix[var_idx, mb_var_idx] = 0
            self.flag_arr[mb_var_idx] = True

        # TODO remove for RSL W
        # if len(var_markov_boundary) > len(var_neighbors):
        #     # Sufficient condition for diamond-free graphs
        #     return

        # find nodes whose co-parent status changes
        # we only remove Y from mkvb of Z iff X is their ONLY common child and they are NOT neighbors)
        for ne_idx_y in range(len(var_neighbors) - 1):  # -1 because no need to check last variable and also symmetry
            for ne_idx_z in range(ne_idx_y + 1, len(var_neighbors)):
                var_y_idx = var_neighbors[ne_idx_y]
                var_z_idx = var_neighbors[ne_idx_z]
                var_y_name = self.var_names[var_y_idx]
                var_z_name = self.var_names[var_z_idx]

                # determine whether the mkbv of var_y_idx or var_z_idx is smaller, and use the smaller one as cond_set
                var_y_markov_boundary = np.flatnonzero(self.markov_boundary_matrix[var_y_idx])
                var_z_markov_boundary = np.flatnonzero(self.markov_boundary_matrix[var_z_idx])
                if np.sum(self.markov_boundary_matrix[var_y_idx]) < np.sum(self.markov_boundary_matrix[var_z_idx]):
                    cond_set = [self.var_names[idx] for idx in set(var_y_markov_boundary) - {var_z_idx}]
                else:
                    cond_set = [self.var_names[idx] for idx in set(var_z_markov_boundary) - {var_y_idx}]

                if self.ci_test(var_y_name, var_z_name, cond_set, self.data):
                    # we know that Y and Z are co-parents and thus NOT neighbors
                    self.markov_boundary_matrix[var_y_idx, var_z_idx] = 0
                    self.markov_boundary_matrix[var_z_idx, var_y_idx] = 0
                    self.flag_arr[var_y_idx] = True
                    self.flag_arr[var_z_idx] = True
