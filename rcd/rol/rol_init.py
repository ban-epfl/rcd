import pandas as pd

from rcd.utilities.utils import *

"""
This file contains a class to initialize the ROL algorithm. It is a copy of the RSL-D algorithm.
"""


class ROLInitializer:
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

    def reset_fields(self, data: pd.DataFrame):
        self.num_vars = len(data.columns)
        self.data = data
        self.var_names = data.columns

        self.flag_arr = np.ones(self.num_vars, dtype=bool)
        self.var_idx_set = set(range(self.num_vars))
        self.markov_boundary_matrix = None

    def learn_and_get_r_order(self, data: pd.DataFrame) -> np.ndarray:
        """
        Run the rsl algorithm on the data to learn and return the learned skeleton graph
        :return: A networkx graph representing the learned skeleton
        """
        self.reset_fields(data)

        self.markov_boundary_matrix = find_markov_boundary_matrix(self.data, self.ci_test)

        var_idx_left_set = self.var_idx_set.copy()
        r_order = []
        while var_idx_left_set:
            # find a removable variable
            removable_var_idx = self.find_removable(list(var_idx_left_set))

            # find the neighbors of the removable variable
            neighbors = self.find_neighborhood(removable_var_idx)

            # update the markov boundary matrix
            self.update_markov_boundary_matrix(removable_var_idx, neighbors)


            # remove the removable variable from the set of variables left
            var_idx_left_set.remove(removable_var_idx)
            r_order.append(removable_var_idx)

        return np.asarray(r_order)

    def find_neighborhood(self, var_idx: int) -> np.ndarray:
        """
        Find the neighborhood of a variable using Lemma 4 of the rsl paper.
        :param var_idx: Index of the variable in the data
        :return: 1D numpy array containing the indices of the variables in the neighborhood
        """

        var_name = self.var_names[var_idx]
        var_mk_arr = self.markov_boundary_matrix[var_idx]
        var_mk_idxs = np.flatnonzero(var_mk_arr)

        # loop through markov boundary matrix row corresponding to var_name
        # use Lemma 4 of rsl paper: var_y_idx is Y and var_z_idx is Z. cond_set is Mb(X) - {Y, Z}
        # at first, assume all variables are neighbors
        neighbors = np.copy(var_mk_arr)

        for mb_idx_y in range(len(var_mk_idxs)):
            for mb_idx_z in range(len(var_mk_idxs)):
                if mb_idx_y == mb_idx_z:
                    continue
                var_y_idx = var_mk_idxs[mb_idx_y]
                var_z_idx = var_mk_idxs[mb_idx_z]
                var_y_name = self.var_names[var_y_idx]
                cond_set = [self.var_names[idx] for idx in set(var_mk_idxs) - {var_y_idx, var_z_idx}]

                if self.ci_test(var_name, var_y_name, cond_set, self.data):
                    # we know that var2 is a co-parent and thus NOT a neighbor
                    neighbors[var_y_idx] = 0
                    break

        # remove all variables that are not neighbors
        neighbors_idx_arr = np.flatnonzero(neighbors)
        return neighbors_idx_arr

    def is_removable(self, var_idx: int) -> bool:
        """
        Check whether a variable is removable using Lemma 3 of the rsl paper.
        :param var_idx:
        :return: True if the variable is removable, False otherwise
        """

        var_mk_arr = self.markov_boundary_matrix[var_idx]
        var_mk_idxs = np.flatnonzero(var_mk_arr)

        # use Lemma 3 of rsl paper: var_y_idx is Y and var_z_idx is Z. cond_set is Mb(X) + {X} - {Y, Z}
        for mb_idx_y in range(len(var_mk_idxs) - 1):  # -1 because no need to check last variable and also symmetry
            for mb_idx_z in range(mb_idx_y + 1, len(var_mk_idxs)):
                var_y_idx = var_mk_idxs[mb_idx_y]
                var_z_idx = var_mk_idxs[mb_idx_z]
                var_y_name = self.var_names[var_y_idx]
                var_z_name = self.var_names[var_z_idx]
                cond_set = [self.var_names[idx] for idx in set(var_mk_idxs) - {var_y_idx, var_z_idx}] + [
                    self.var_names[var_idx]]

                if self.ci_test(var_y_name, var_z_name, cond_set, self.data):
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
