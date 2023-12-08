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

    def __init__(self, ci_test, clique_num: int):
        """
        Initialize the rsl-W algorithm with a conditional independence test function and an upper bound on the clique
        number.
        :param ci_test: Conditional independence test to use (see RSLBase for details)
        :param clique_num: Upper bound on the clique number of the underlying graph
        """
        super().__init__(ci_test)
        self.clique_num = clique_num

    def find_neighborhood(self, var_idx: int) -> np.ndarray:
        """
        # TODO CHECK
        Find the neighborhood of a variable using Lemma 2 of the rsl paper.
        :param var_idx: Index of the variable in the data
        :return: 1D numpy array containing the indices of the variables in the neighborhood
        """

        var_name = self.var_names[var_idx]
        var_mk_arr = self.markov_boundary_matrix[var_idx]
        var_mk_idxs = np.flatnonzero(var_mk_arr)

        # loop through markov boundary matrix row corresponding to var_name
        # use Lemma 2 of rsl paper: var_y_idx is Y and var_z_idx is Z. cond_set is Mb(X) - {Y} - S,
        # where S is a subset of Mb(X) - {Y} of size m-1

        # at first, assume all variables are neighbors
        neighbors = np.copy(var_mk_arr)

        # var_s_idxs contains the indices of the variables in the subset S
        for mb_idx_y in range(len(var_mk_idxs)):
            var_y_idx = var_mk_idxs[mb_idx_y]
            var_y_name = self.var_names[var_y_idx]
            var_mk_idxs_left = list(set(var_mk_idxs) - {var_y_idx})
            for var_s_idxs in itertools.combinations(var_mk_idxs_left, self.clique_num - 1):
                cond_set = [self.var_names[idx] for idx in set(var_mk_idxs) - {var_y_idx} - set(var_s_idxs)]

                if self.ci_test(var_name, var_y_name, cond_set, self.data):
                    # we know that var2 is a co-parent and thus NOT a neighbor
                    neighbors[var_y_idx] = 0
                    break

        # remove all variables that are not neighbors
        neighbors_idx_arr = np.flatnonzero(neighbors)
        return neighbors_idx_arr

    def is_removable(self, var_idx: int) -> bool:
        # TODO CHECK
        """
        Check whether a variable is removable using Lemma 1 of the rsl paper.
        :param var_idx:
        :return: True if the variable is removable, False otherwise
        """
        var_name = self.var_names[var_idx]
        var_mk_arr = self.markov_boundary_matrix[var_idx]
        var_mk_idxs = np.flatnonzero(var_mk_arr)

        # TODO change comments
        # use Lemma 1 of rsl paper: var_y_idx is Y and var_z_idx is Z. cond_set is Mb(X) + {X} - ({Y, Z} + S)
        # get all subsets with size from 0 to self.clique_num - 2.
        # for each subset, check if there exists a pair of variables that are d-separated given the subset
        for subset_size in range(self.clique_num - 1):
            # var_s_idxs contains the indices of the variables in the subset S
            for var_s_idxs in itertools.combinations(var_mk_idxs, subset_size):
                var_mk_idxs_left = list(set(var_mk_idxs) - set(var_s_idxs))

                for mb_idx_left_y in range(len(var_mk_idxs_left) - 1):  # -1 because symmetry
                    var_y_idx = var_mk_idxs_left[mb_idx_left_y]
                    var_y_name = self.var_names[var_y_idx]

                    # check second condition
                    cond_set = [self.var_names[idx] for idx in set(var_mk_idxs_left) - {var_y_idx}]
                    if self.ci_test(var_name, var_y_name, cond_set, self.data):
                        return False

                    # check first condition
                    for mb_idx_left_z in range(mb_idx_left_y + 1, len(var_mk_idxs_left)):
                        var_z_idx = var_mk_idxs_left[mb_idx_left_z]
                        var_z_name = self.var_names[var_z_idx]
                        cond_set = ([self.var_names[idx] for idx in set(var_mk_idxs_left) - {var_y_idx, var_z_idx}] +
                                    [var_name])

                        if self.ci_test(var_y_name, var_z_name, cond_set, self.data):
                            return False
        return True
