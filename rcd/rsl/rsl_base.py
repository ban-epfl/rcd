import numpy as np

from rcd.utilities.utils import *

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

from typing import Callable, List

REMOVABLE_NOT_FOUND = -1

class _RSLBase:
    """
    Base class for RSL algorithms for learning diamond-free graphs and graphs with a bounded clique number.

    This class is initialized with a conditional independence test function, which determines whether two variables
    are independent given another set of variables, using the data provided.

    The class has a learn_and_get_skeleton function that takes in a data matrix (numpy array), where each column
    corresponds to a variable and each row corresponds to a sample, and returns a networkx graph representing the learned skeleton.
    """

    def __init__(self, ci_test: Callable[[int, int, List[int], np.ndarray], bool], find_markov_boundary_matrix_fun: Callable[[np.ndarray], np.ndarray] = None):
        """
        Initialize the RSL algorithm with the conditional independence test to use.

        Args:
            ci_test (Callable[[int, int, List[int], np.ndarray], bool]): A conditional independence test function that takes in the indices of two variables
                                and a list of variable indices as the conditioning set, and returns True if the two
                                variables are independent given the conditioning set, and False otherwise.
            find_markov_boundary_matrix_fun (Callable[[np.ndarray], np.ndarray], optional): A function to find the Markov boundary matrix.
                This function should take in a numpy array of data, and return a 2D numpy array, where the (i, j)th
                entry is True if the jth variable is in the Markov boundary of the ith variable, and False otherwise.
        """
        if find_markov_boundary_matrix_fun is None:
            self.find_markov_boundary_matrix = compute_mb_gaussian
        else:
            self.find_markov_boundary_matrix = find_markov_boundary_matrix_fun

        self.num_vars = None
        self.data = None
        self.ci_test = ci_test

        # we use a flag array to keep track of which variables need to be checked for removal (i.e., we check if False)
        self.skip_rem_check_vec = None  # SkipCheck_VEC in the paper
        self.markov_boundary_matrix = None
        self.learned_skeleton = None
        self.is_rsl_d = False
        self.clique_num = None

    def learn_and_get_skeleton(self, data: np.ndarray, clique_num: int = None, return_r_order=False) -> nx.Graph:
        """
        Run the RSL algorithm on the data to learn and return the learned skeleton graph.

        Args:
            data (np.ndarray): The data matrix with shape (num_samples, num_vars).
            clique_num (int, optional): The clique number of the graph, used only for specific versions of the algorithm.

        Returns:
            nx.Graph: A networkx graph representing the learned skeleton.
        """
        if not self.is_rsl_d and clique_num is None:
            raise ValueError("Clique number not given!")

        self.num_vars = data.shape[1]
        self.data = data
        self.clique_num = clique_num

        self.skip_rem_check_vec = np.zeros(self.num_vars, dtype=bool)
        self.markov_boundary_matrix = self.find_markov_boundary_matrix(self.data)
        self.learned_skeleton = None

        skeleton = nx.Graph()
        skeleton.add_nodes_from(range(self.num_vars))

        data_included_ci_test = lambda x, y, z: self.ci_test(x, y, z, self.data)

        var_arr = np.arange(self.num_vars)
        var_left_bool_arr = np.ones(self.num_vars, dtype=bool)  # if ith position is True, indicates that i is left
        if return_r_order:
            r_order = np.zeros(self.num_vars, dtype=int)
        for i in range(self.num_vars - 1):
            # only consider variables that are left and have skip check set to False
            var_to_check_arr = var_arr[var_left_bool_arr & ~self.skip_rem_check_vec]

            # sort the variables by the size of their markov boundary
            mb_size = np.sum(self.markov_boundary_matrix[var_to_check_arr], axis=1)
            sort_indices = np.argsort(mb_size, kind='stable')
            sorted_var_arr = var_to_check_arr[sort_indices]

            # find a removable variable
            removable_var = self.find_removable(sorted_var_arr)

            if removable_var == REMOVABLE_NOT_FOUND:
                # if no removable found, then pick the variable with the smallest markov boundary from var_left_bool_arr
                var_left_arr = np.flatnonzero(var_left_bool_arr)
                mb_size_all = np.sum(self.markov_boundary_matrix[var_left_arr], axis=1)
                removable_var = var_left_arr[np.argmin(mb_size_all)]

                self.skip_rem_check_vec[:] = False

            # find the neighbors of the removable variable
            neighbors = self.find_neighborhood(removable_var)

            # update the markov boundary matrix
            update_markov_boundary_matrix(self.markov_boundary_matrix,
                                          data_included_ci_test,
                                          removable_var,
                                          neighbors,
                                          self.is_rsl_d,
                                          skip_check=self.skip_rem_check_vec)

            # add edges between the removable variable and its neighbors
            for neighbor_idx in neighbors:
                skeleton.add_edge(removable_var, neighbor_idx)

            # remove the removable variable from the set of variables left
            var_left_bool_arr[removable_var] = False
            if return_r_order:
                r_order[i] = removable_var

        if return_r_order:
            r_order[-1] = var_arr[var_left_bool_arr][0]
            return r_order
        self.learned_skeleton = skeleton
        return skeleton

    def find_neighborhood(self, var: int) -> np.ndarray:
        """
        Find the neighborhood of a variable.

        Args:
            var (int): The variable whose neighborhood we want to find.

        Returns:
            np.ndarray: 1D numpy array containing the variables in the neighborhood.
        """
        raise NotImplementedError()

    def is_removable(self, var: int) -> bool:
        """
        Check whether a variable is removable.

        Args:
            var (int): The variable to check.

        Returns:
            bool: True if the variable is removable, False otherwise.
        """
        raise NotImplementedError()

    def find_removable(self, var_arr: np.ndarray) -> int:
        """
        Find a removable variable in the given list of variables.

        Args:
            var_arr (np.ndarray): 1D array of variables.

        Returns:
            int: The index of the removable variable, if found, and REMOVABLE_NOT_FOUND if not found.
        """
        for var in var_arr:
            if self.is_removable(var):
                return var
            self.skip_rem_check_vec[var] = True
        return REMOVABLE_NOT_FOUND
