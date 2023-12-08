import numpy as np

from rcd.rsl.rsl_base import RSLBase

"""
This file contains the implementation for the rsl-D algorithm learning diamond-free graphs from i.i.d. samples. 
The class is initialized with a conditional independence test function, which determines whether two variables are 
independent given another set of variables, using the data provided. For examples of possible conditional 
independence tests, see utilities/ci_tests.py. For details on how to write a custom CI test function, look at the 
constructor (init function) of the RSLBase class.

The class has a learn_and_get_skeleton function, inherited from the base class, that takes in a Pandas DataFrame of 
data, where the ith column contains samples from the ith variable, and returns a networkx graph representing the 
learned skeleton.
"""


class RSLDiamondFree(RSLBase):
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
