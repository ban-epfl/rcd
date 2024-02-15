import itertools
from itertools import combinations
from typing import List
import networkx as nx
import numpy as np
import pandas as pd

from rcd.utilities.utils import *

"""This file contains the base class implementation for the L-MARVEL for learning causal graphs with latent 
variables. The class is initialized with a conditional independence test function, which determines whether two 
variables are independent given another set of variables, using the data provided. For examples of possible 
conditional independence tests, see utilities/ci_tests.py. For details on how to write a custom CI test function, 
look at the constructor (init function) of the RSLBase class below.

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




class LMarvel:
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

        # we use a flag array to keep track of which variables need to be checked for removal (i.e., we check if False)
        self.skip_rem_check_vec = None  # SkipCheck_VEC in the paper

        # we use a set to keep track of which Y and Z pairs have been checked for a given X (see IsRemovable in the paper)
        self.skip_rem_check_set = None  # SkipCheck_MAT in the paper

        # we use a flag array to keep track of which variables' neighbors need to be learned (i.e., we learn if False)
        self.neighbor_learned_arr = None
        self.var_idx_set = None
        self.markov_boundary_matrix = None
        self.learned_skeleton = None

    def reset_fields(self, data: pd.DataFrame):
        self.num_vars = len(data.columns)
        self.data = data
        self.var_names = data.columns

        self.skip_rem_check_vec = np.zeros(self.num_vars, dtype=bool)
        self.skip_rem_check_set = set()
        self.neighbor_learned_arr = np.zeros(self.num_vars, dtype=bool)
        self.var_idx_set = set(range(self.num_vars))
        self.markov_boundary_matrix = None
        self.learned_skeleton: nx.Graph | None = None

    def has_alg_run(self):
        return self.learned_skeleton is not None

    def learn_and_get_skeleton(self, data: pd.DataFrame) -> nx.Graph:
        """
        Run the rsl algorithm on the data to learn and return the learned skeleton graph
        :return: A networkx graph representing the learned skeleton
        """
        self.reset_fields(data)

        # initialize graph
        self.learned_skeleton = nx.Graph()
        self.learned_skeleton.add_nodes_from(self.var_names)

        self.markov_boundary_matrix = find_markov_boundary_matrix(self.data, self.ci_test)

        var_idx_arr = np.arange(self.num_vars)

        var_left_bool_arr = np.ones(len(self.var_names), dtype=bool)  # if ith position is True, indicates that i is left

        for _ in range(self.num_vars - 1):
            # sort variables by decreasing Markov boundary size
            # only sort variables that are still left and whose removability has NOT been checked
            var_to_sort_bool_arr = var_left_bool_arr & ~self.skip_rem_check_vec
            var_idx_to_sort_arr = var_idx_arr[var_to_sort_bool_arr]
            sorted_var_idx = sort_vars_by_mkb_size(self.markov_boundary_matrix[var_to_sort_bool_arr], var_idx_to_sort_arr)

            for var_idx in sorted_var_idx:
                # check whether we need to learn the neighbors of var_idx
                if not self.neighbor_learned_arr[var_idx]:
                    neighbors = self.find_neighborhood(var_idx)
                    self.neighbor_learned_arr[var_idx] = True

                    # add edges between the variable and its neighbors
                    for neighbor_idx in neighbors:
                        self.learned_skeleton.add_edge(self.var_names[var_idx], self.var_names[neighbor_idx])
                else:
                    # if neighbors already learned, get them from the graph
                    neighbors = self.learned_skeleton.neighbors(var_idx)

                    # make sure to only include neighbours that are still left
                    neighbors = [neighbor for neighbor in neighbors if var_left_bool_arr[neighbor]]

                # check if variable is removable
                if self.is_removable(var_idx, neighbors):
                    # remove the removable variable from the set of variables left
                    var_left_bool_arr[var_idx] = False

                    # update the markov boundary matrix
                    self.update_markov_boundary_matrix(var_idx, neighbors)
                    break
                else:
                    self.skip_rem_check_vec[var_idx] = True

        return self.learned_skeleton

    def find_neighborhood(self, var_idx: int) -> np.ndarray:
        """
        Find the neighborhood of a variable using Lemma 27.
        :param var_idx: Index of the variable in the data
        :return: 1D numpy array containing the indices of the variables in the neighborhood
        """

        var_name = self.var_names[var_idx]
        var_mk_arr = self.markov_boundary_matrix[var_idx]
        var_mk_idxs = np.flatnonzero(var_mk_arr)

        neighbors = np.copy(var_mk_arr)

        for mb_idx_y in range(len(var_mk_idxs)):
            var_y_idx = var_mk_idxs[mb_idx_y]
            # check if Y is already neighbor of X
            if not self.learned_skeleton.has_edge(var_idx, var_y_idx):
                if not self.is_neighbor(var_name, var_y_idx, var_mk_idxs):
                    neighbors[var_y_idx] = 0

        # remove all variables that are not neighbors
        neighbors_idx_arr = np.flatnonzero(neighbors)
        return neighbors_idx_arr

    def is_neighbor(self, var_name: str, var_y_idx: int, var_x_mk_idxs: np.ndarray) -> bool:
        var_mk_left_idxs = list(set(var_x_mk_idxs) - {var_y_idx})
        # use lemma 27 and check all proper subsets of Mb(X) - {Y}
        for cond_set_size in range(len(var_mk_left_idxs)):
            for var_s_idxs in itertools.combinations(var_mk_left_idxs, cond_set_size):
                cond_set = [self.var_names[idx] for idx in var_s_idxs]
                var_y_name = self.var_names[var_y_idx]
                if self.ci_test(var_name, var_y_name, cond_set, self.data):
                    # we know that var_y_idx is a co-parent and thus NOT a neighbor
                    return False
        return True

    def is_removable(self, var_idx: int, neighbors: np.ndarray) -> bool:
        """
        Check whether a variable is removable using Lemma 3 of the rsl paper.
        :param var_idx: Index of the variable
        :param neighbors: Neighbors of the variable
        :return: True if the variable is removable, False otherwise
        """

        var_name = self.var_names[var_idx]
        var_mk_arr = self.markov_boundary_matrix[var_idx]
        var_mk_idxs = np.flatnonzero(var_mk_arr)

        def cond_1(var_y_idx, var_z_idx):
            # there exists subset W in Mb(X) - {Y, Z}, s.t. Y ind. Z | W
            var_mk_left_idxs = list(set(var_mk_idxs) - {var_y_idx, var_z_idx})
            for cond_set_size in range(len(var_mk_left_idxs) + 1):
                for var_s_idxs in itertools.combinations(var_mk_left_idxs, cond_set_size):
                    cond_set = [self.var_names[idx] for idx in var_s_idxs]
                    if self.ci_test(var_y_name, var_z_name, cond_set, self.data):
                        return True
            return False

        def cond_2(var_y_idx, var_z_idx):
            # for all subset W in Mb(X) - {Y, Z}, s.t. Y NOT ind. Z | W + {X}
            var_mk_left_idxs = list(set(var_mk_idxs) - {var_y_idx, var_z_idx})
            for cond_set_size in range(len(var_mk_left_idxs) + 1):
                for var_s_idxs in itertools.combinations(var_mk_left_idxs, cond_set_size):
                    cond_set = [self.var_names[idx] for idx in var_s_idxs] + [var_name]
                    if self.ci_test(var_y_name, var_z_name, cond_set, self.data):
                        return False
            return True

        # Use Theorem 32 to check if X is removable. Loop over Y in Mb(X) and Z in Ne(X)
        for var_y_idx in var_mk_idxs:
            var_y_name = self.var_names[var_y_idx]
            for var_z_idx in neighbors:
                var_z_name = self.var_names[var_z_idx]
                if var_y_idx == var_z_idx:
                    continue
                xyz_tuple = (var_idx, min(var_y_idx, var_z_name), max(var_y_idx, var_z_idx))
                if xyz_tuple in self.skip_rem_check_set:
                    continue
                if not (cond_1(var_y_idx, var_z_idx) or cond_2(var_y_idx, var_z_idx)):
                    return False
                self.skip_rem_check_set.add(xyz_tuple)
        return True

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
            self.skip_rem_check_vec[mb_var_idx] = False

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
                    self.skip_rem_check_vec[var_y_idx] = False
                    self.skip_rem_check_vec[var_z_idx] = False
