import itertools
import numpy as np

from rcd.rsl.rsl_base import RSLBase

"""
This file contains the implementation for the rsl-W algorithm learning graphs with a bounded clique number from 
i.i.d. samples. The class is initialized with a conditional independence test function, which determines whether two 
variables are independent given another set of variables, using the data provided. For examples of possible 
conditional independence tests, see utilities/ci_tests.py. For details on how to write a custom CI test function, 
look at the constructor (init function) of the RSLBase class.

The class has a learn_and_get_skeleton function, inherited from the base class, that takes in a Pandas DataFrame of 
data, where the ith column contains samples from the ith variable, and returns a networkx graph representing the 
learned skeleton.
"""


class RSLBoundedClique(RSLBase):

    def find_neighborhood(self, var: int) -> np.ndarray:
        """Find the neighborhood of a variable using Proposition 37.

        Args:
            var (int): The variable whose neighborhood we want to find.

        Returns:
            np.ndarray: 1D numpy array containing the variables in the neighborhood.
        """

        var_name = self.var_names[var]
        var_mk_bool_arr = self.markov_boundary_matrix[var]
        var_mk_arr = np.flatnonzero(var_mk_bool_arr)
        var_mk_set = set(var_mk_arr)

        # loop through markov boundary matrix row corresponding to var_name
        # use Proposition 37: var_y is Y and var_z_idx is Z. cond_set is Mb(X) - {Y} - S,
        # where S is a subset of Mb(X) - {Y} of size m-1

        # First, assume all variables are neighbors
        neighbor_bool_arr = np.copy(var_mk_bool_arr)

        # var_s contains the indices of the variables in the subset S
        for var_y in var_mk_arr:
            var_y_name = self.var_names[var_y]
            var_mk_left = list(var_mk_set - {var_y})
            for var_s in itertools.combinations(var_mk_left, self.clique_num - 1):
                cond_set = [self.var_names[idx] for idx in var_mk_set - {var_y} - set(var_s)]
                if self.ci_test(var_name, var_y_name, cond_set, self.data):
                    # we know that var_y is a co-parent and thus NOT a neighbor
                    neighbor_bool_arr[var_y] = 0
                    break
            if not neighbor_bool_arr[var_y]:
                continue

        # return neighbors
        neighbor_arr = np.flatnonzero(neighbor_bool_arr)
        return neighbor_arr

    def is_removable(self, var: int) -> bool:
        """Check whether a variable is removable using Theorem 36.

        Args:
            var (int): The variable to check.

        Returns:
            bool: True if the variable is removable, False otherwise.
        """

        var_name = self.var_names[var]
        var_mk_bool_arr = self.markov_boundary_matrix[var]
        var_mk_arr = np.flatnonzero(var_mk_bool_arr)
        var_mk_set = set(var_mk_arr)

        # use Theorem 36: var_y is Y and var_z_idx is Z. cond_set is Mb(X) + {X} - ({Y, Z} + S)
        # get all subsets with size from 0 to self.clique_num - 2.
        # for each subset, check if there exists a pair of variables that are d-separated given the subset
        for subset_size in range(max(self.clique_num - 1, self.num_vars + 1)):
            # var_s contains the variables in the subset S
            for var_s in itertools.combinations(var_mk_arr, subset_size):
                var_mk_left = list(var_mk_set - set(var_s))
                var_mk_left_set = set(var_mk_left)

                for mb_idx_left_y in range(len(var_mk_left)):
                    var_y = var_mk_left[mb_idx_left_y]
                    var_y_name = self.var_names[var_y]

                    # check second condition
                    cond_set = [self.var_names[idx] for idx in var_mk_left_set - {var_y}]

                    if self.ci_test(var_name, var_y_name, cond_set, self.data):
                        return False

                    # check first condition
                    for mb_idx_left_z in range(mb_idx_left_y + 1, len(var_mk_left)):
                        var_z_idx = var_mk_left[mb_idx_left_z]
                        var_z_name = self.var_names[var_z_idx]
                        cond_set = ([self.var_names[idx] for idx in var_mk_left_set - {var_y, var_z_idx}] + [var_name])

                        if self.ci_test(var_y_name, var_z_name, cond_set, self.data):
                            return False
        return True
