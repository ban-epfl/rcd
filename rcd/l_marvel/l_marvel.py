import itertools
from typing import Set

from rcd.utilities.utils import *

"""
l_marvel.py contains implementation for the L-MARVEL algorithm for learning causal graphs with latent
variables. The class is initialized with a conditional independence test function, which determines whether two
variables are independent given another set of variables, using the data provided. For examples of possible
conditional independence tests, see utilities/ci_tests.py. For details on how to write a custom CI test function,
look at the constructor (init function) of the LMarvel class.

The class has a learn_and_get_skeleton function that takes in a Pandas DataFrame of data, where the ith column
contains samples from the ith variable, and returns a networkx graph representing the learned skeleton.
"""


def learn_and_get_skeleton(ci_test: Callable[[int, int, List[int], np.ndarray], bool], data,
                           find_markov_boundary_matrix_fun=None) -> nx.Graph:
    """
    Learn the skeleton of a causal graph with latent variables using the L-MARVEL algorithm.

    Args:
        ci_test (Callable[[int, int, List[int], np.ndarray], bool]):
            A conditional independence test function that takes in the indices of two variables
            and a list of variable indices as the conditioning set, and returns True if the two
            variables are independent given the conditioning set, and False otherwise.
        data_matrix (np.ndarray): The data matrix with shape (num_samples, num_vars), where each column corresponds
                                  to a variable and each row corresponds to a sample.

    Returns:
        nx.Graph: A networkx graph representing the learned skeleton.
    """

    data_mat = sanitize_data(data)
    l_marvel = _LMarvel(ci_test, find_markov_boundary_matrix_fun)
    learned_skeleton = l_marvel.learn_and_get_skeleton(data_mat)
    return learned_skeleton


REMOVABLE_NOT_FOUND = -1

class _LMarvel:
    """
    Implementation for the L-MARVEL algorithm for learning causal graphs with latent variables.

    This class is initialized with a conditional independence test function, which determines whether two variables are
    independent given another set of variables, using the data provided.

    The class has a learn_and_get_skeleton function that takes in a data matrix (numpy array), where each column
    corresponds to a variable and each row corresponds to a sample, and returns a networkx graph representing the
    learned skeleton.
    """

    def __init__(self, ci_test: Callable[[int, int, List[int], np.ndarray], bool], find_markov_boundary_matrix_fun: Callable[[np.ndarray], np.ndarray] = None):
        """
        Initialize the L-MARVEL algorithm with the conditional independence test to use.

        Args:
            ci_test (Callable[[int, int, List[int], np.ndarray], bool]):
                A conditional independence test function. It takes the indices of two variables
                and a list of variable indices as the conditioning set. It returns True if the two
                variables are independent given the conditioning set, and False otherwise.
                Signature:
                    ci_test(var1: int, var2: int, cond_set: List[int], data: np.ndarray) -> bool.
            find_markov_boundary_matrix_fun (Callable[[np.ndarray], np.ndarray], optional):
                A function to find the Markov boundary matrix. It takes a numpy array of data
                and returns a 2D numpy array. The (i, j)th entry is True if the jth variable is in the
                Markov boundary of the ith variable, and False otherwise.
                Signature:
                    find_markov_boundary_matrix_fun(data: np.ndarray) -> np.ndarray.
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

        # we use a set to keep track of which Y and Z pairs have been checked for a given X (see IsRemovable in the paper)
        self.skip_rem_check_set = None  # SkipCheck_MAT in the paper

        # we use a flag array to keep track of which variables' neighbors need to be learned (i.e., we learn if False)
        self.neighbor_learned_arr = None
        self.var_idx_set = None
        self.markov_boundary_matrix = None
        self.learned_skeleton = None

    def learn_and_get_skeleton(self, data: np.ndarray) -> nx.Graph:
        """
        Run the L-MARVEL algorithm on the data to learn and return the learned skeleton graph.

        Args:
            data (np.ndarray): The data matrix with shape (num_samples, num_vars).

        Returns:
            nx.Graph: A networkx graph representing the learned skeleton.
        """
        # Initialize algorithm state
        self.num_vars = data.shape[1]
        self.data = data

        self.skip_rem_check_vec = np.zeros(self.num_vars, dtype=bool)
        self.skip_rem_check_set = set()
        self.neighbor_learned_arr = np.zeros(self.num_vars, dtype=bool)
        self.var_idx_set = set(range(self.num_vars))
        self.markov_boundary_matrix = self.find_markov_boundary_matrix(self.data)
        self.learned_skeleton = nx.Graph()
        self.learned_skeleton.add_nodes_from(range(self.num_vars))

        data_included_ci_test = lambda x, y, z: self.ci_test(x, y, z, self.data)

        var_arr = np.arange(self.num_vars)
        var_left_bool_arr = np.ones(self.num_vars, dtype=bool)  # Indicates if variable is left

        for _ in range(self.num_vars - 1):
            # sort variables by decreasing Markov boundary size
            # only sort variables that are still left and whose removability has NOT been checked
            var_to_sort_bool_arr = var_left_bool_arr & ~self.skip_rem_check_vec
            var_to_sort_arr = var_arr[var_to_sort_bool_arr]
            sorted_var_arr = sort_vars_by_mkb_size(self.markov_boundary_matrix[var_to_sort_bool_arr], var_to_sort_arr)

            removable_var = REMOVABLE_NOT_FOUND
            for var in sorted_var_arr:
                # Check whether we need to learn the neighbors of var
                if not self.neighbor_learned_arr[var]:
                    neighbors = self.find_neighborhood(var)
                    self.neighbor_learned_arr[var] = True

                    # Add edges between the variable and its neighbors
                    for neighbor in neighbors:
                        self.learned_skeleton.add_edge(var, neighbor)
                else:
                    # If neighbors already learned, get them from the graph
                    neighbors = list(self.learned_skeleton.neighbors(var))

                    # Ensure only to include neighbors that are still left
                    neighbors = [neighbor for neighbor in neighbors if var_left_bool_arr[neighbor]]

                # Check if variable is removable
                if self.is_removable(var, neighbors):
                    removable_var = var
                    break
                else:
                    self.skip_rem_check_vec[var] = True

            if removable_var == REMOVABLE_NOT_FOUND:
                # If no removable found, pick the variable with the smallest Markov boundary from var_left_bool_arr
                var_left_arr = np.flatnonzero(var_left_bool_arr)
                mb_size_all = np.sum(self.markov_boundary_matrix[var_left_arr], axis=1)
                removable_var = var_left_arr[np.argmin(mb_size_all)]

                self.skip_rem_check_vec[:] = False
            else:
                # Remove the removable variable from the set of variables left
                var_left_bool_arr[removable_var] = False

            # Make sure to only include neighbors that are still left
            neighbors = [neighbor for neighbor in self.learned_skeleton.neighbors(removable_var) if var_left_bool_arr[neighbor]]

            # Update the Markov boundary matrix
            update_markov_boundary_matrix(
                self.markov_boundary_matrix,
                data_included_ci_test,
                removable_var,
                neighbors,
                skip_check=self.skip_rem_check_vec,
            )

        return self.learned_skeleton

    def find_neighborhood(self, var: int) -> np.ndarray:
        """
        Find the neighborhood of a variable using Lemma 27.

        Args:
            var (int): The variable whose neighborhood we want to find.

        Returns:
            np.ndarray: 1D numpy array containing the variables in the neighborhood.
        """
        var_mk_bool_arr = self.markov_boundary_matrix[var]
        var_mk_arr = np.flatnonzero(var_mk_bool_arr)
        var_mk_set = set(var_mk_arr)

        neighbor_bool_arr = np.copy(var_mk_bool_arr)

        for var_y in var_mk_arr:
            # Check if Y is already a neighbor of X
            if not self.learned_skeleton.has_edge(var, var_y):
                if not self.is_neighbor(var, var_y, var_mk_set):
                    neighbor_bool_arr[var_y] = False

        # Remove all variables that are not neighbors
        neighbors = np.flatnonzero(neighbor_bool_arr)
        return neighbors

    def is_neighbor(self, var: int, var_y: int, var_mk_set: Set[int]) -> bool:
        """
        Check if var_y is a neighbor of variable var using Lemma 27.

        Args:
            var (int): Index of the variable.
            var_y (int): The variable to check.
            var_mk_set (Set[int]): Set of the variables in the Markov boundary of var.

        Returns:
            bool: True if var_y is a neighbor, False otherwise.
        """
        var_mk_left_list = list(var_mk_set - {var_y})
        # Use Lemma 27 and check all proper subsets of Mb(X) - {Y}
        for cond_set_size in range(len(var_mk_left_list) + 1):
            for var_s in itertools.combinations(var_mk_left_list, cond_set_size):
                cond_set = list(var_s)
                if self.ci_test(var, var_y, cond_set, self.data):
                    # Y is a co-parent and thus NOT a neighbor
                    return False
        return True

    def is_removable(self, var: int, neighbors: np.ndarray) -> bool:
        """
        Check whether a variable is removable using Theorem 32.

        Args:
            var (int): Index of the variable.
            neighbors (np.ndarray): Neighbors of the variable.

        Returns:
            bool: True if the variable is removable, False otherwise.
        """
        var_mk_bool_arr = self.markov_boundary_matrix[var]
        var_mk_arr = np.flatnonzero(var_mk_bool_arr)
        var_mk_set = set(var_mk_arr)

        def cond_1(var_y, var_z):
            # there exists subset W in Mb(X) - {Y, Z}, s.t. Y ind. Z | W
            var_mk_left_list = list(var_mk_set - {var_y, var_z})
            for cond_set_size in range(len(var_mk_left_list) + 1):
                for var_s in itertools.combinations(var_mk_left_list, cond_set_size):
                    cond_set = list(var_s)
                    if self.ci_test(var_y, var_z, cond_set, self.data):
                        return True
            return False

        def cond_2(var_y, var_z):
            # for all subset W in Mb(X) - {Y, Z}, s.t. Y NOT ind. Z | W + {X}
            var_mk_left_left = list(var_mk_set - {var_y, var_z})
            for cond_set_size in range(len(var_mk_left_left) + 1):
                for var_s in itertools.combinations(var_mk_left_left, cond_set_size):
                    cond_set = list(var_s) + [var]
                    if self.ci_test(var_y, var_z, cond_set, self.data):
                        return False
            return True

        # Use Theorem 32 to check if X is removable. Loop over Y in Mb(X) and Z in Ne(X)
        for var_y in var_mk_arr:
            for var_z in neighbors:
                if var_y == var_z:
                    continue
                xyz_tuple = (var, min(var_y, var_z), max(var_y, var_z))
                if xyz_tuple in self.skip_rem_check_set:
                    continue
                if not (cond_1(var_y, var_z) or cond_2(var_y, var_z)):
                    return False
                self.skip_rem_check_set.add(xyz_tuple)
        return True
