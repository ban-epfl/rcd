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


class LMarvel:
    def __init__(self, ci_test, find_markov_boundary_matrix_fun=None):
        """
        Initialize the L-Marvel algorithm with the data and the conditional independence test.

        Args:
            ci_test (Callable[[str, str, List[str], pd.DataFrame], bool]):
                A conditional independence test function. It takes the names of two variables
                and a list of variable names as the conditioning set. It returns True if the two
                variables are independent given the conditioning set, and False otherwise.
                Signature:
                    ci_test(var_name1: str, var_name2: str, cond_set: List[str], data: pd.DataFrame) -> bool.
            find_markov_boundary_matrix_fun (Callable[[pd.DataFrame], np.ndarray]):
                A function to find the Markov boundary matrix. It takes a Pandas DataFrame of data
                and returns a 2D numpy array. The (i, j)th entry is True if the jth variable is in the
                Markov boundary of the ith variable, and False otherwise.
                Signature:
                    find_markov_boundary_matrix_fun(data: pd.DataFrame) -> np.ndarray.
        """
        if find_markov_boundary_matrix_fun is None:
            self.find_markov_boundary_matrix = lambda data: find_markov_boundary_matrix(data, ci_test)
        else:
            self.find_markov_boundary_matrix = find_markov_boundary_matrix_fun

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
        """Reset the algorithm before running it on new data. Used internally by the algorithm.

        Args:
            data (pd.DataFrame): The data to reset the algorithm with.
        """
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
        """Check if the algorithm has been run.

        Returns:
            bool: True if the algorithm has been run, False otherwise.
        """
        return self.learned_skeleton is not None

    def learn_and_get_skeleton(self, data: pd.DataFrame) -> nx.Graph:
        """Run the l-marvel algorithm on the data to learn and return the learned skeleton graph.

        Args:
            data (pd.DataFrame): The data to learn the skeleton from.

        Returns:
            nx.Graph: A networkx graph representing the learned skeleton.
        """
        self.reset_fields(data)

        # initialize graph as a field and not a local variable as some member functions need to access it
        self.learned_skeleton = nx.Graph()
        self.learned_skeleton.add_nodes_from(self.var_names)

        data_included_ci_test = lambda x, y, z: self.ci_test(x, y, z, self.data)

        self.markov_boundary_matrix = self.find_markov_boundary_matrix(self.data)

        var_arr = np.arange(self.num_vars)

        var_left_bool_arr = np.ones(self.num_vars, dtype=bool)  # if ith position is True, indicates that i is left

        for _ in range(self.num_vars - 1):
            # sort variables by decreasing Markov boundary size
            # only sort variables that are still left and whose removability has NOT been checked
            var_to_sort_bool_arr = var_left_bool_arr & ~self.skip_rem_check_vec
            var_to_sort_arr = var_arr[var_to_sort_bool_arr]
            sorted_var_arr = sort_vars_by_mkb_size(self.markov_boundary_matrix[var_to_sort_bool_arr], var_to_sort_arr)

            removable_var = REMOVABLE_NOT_FOUND
            for var in sorted_var_arr:
                # check whether we need to learn the neighbors of var
                if not self.neighbor_learned_arr[var]:
                    neighbors = self.find_neighborhood(var)
                    self.neighbor_learned_arr[var] = True

                    # add edges between the variable and its neighbors
                    for neighbor in neighbors:
                        self.learned_skeleton.add_edge(self.var_names[var], self.var_names[neighbor])
                else:
                    # if neighbors already learned, get them from the graph
                    neighbors = self.learned_skeleton.neighbors(var)

                    # make sure to only include neighbors that are still left
                    neighbors = [neighbor for neighbor in neighbors if var_left_bool_arr[neighbor]]

                # check if variable is removable
                if self.is_removable(var, neighbors):
                    removable_var = var
                    break
                else:
                    self.skip_rem_check_vec[var] = True

            if removable_var == REMOVABLE_NOT_FOUND:
                # if no removable found, then pick the variable with the smallest markov boundary from var_left_bool_arr
                var_left_arr = np.flatnonzero(var_left_bool_arr)
                mb_size_all = np.sum(self.markov_boundary_matrix[var_left_arr], axis=1)
                removable_var = var_left_arr[np.argmin(mb_size_all)]

                self.skip_rem_check_vec[:] = False
            else:
                # remove the removable variable from the set of variables left
                var_left_bool_arr[removable_var] = False

            # get the neighbors of the removable variable
            # make sure to only include neighbors that are still left
            neighbors = [neighbor for neighbor in self.learned_skeleton.neighbors(removable_var) if var_left_bool_arr[neighbor]]

            # update the markov boundary matrix
            update_markov_boundary_matrix(self.markov_boundary_matrix, self.skip_rem_check_vec, self.var_names,
                                          data_included_ci_test, removable_var, neighbors)

        return self.learned_skeleton

    def find_neighborhood(self, var: int) -> np.ndarray:
        """Find the neighborhood of a variable using Lemma 27.

        Args:
            var (int): The variable whose neighborhood we want to find.

        Returns:
            np.ndarray: 1D numpy array containing the variables in the neighborhood.
        """

        var_name = self.var_names[var]
        var_mk_bool_arr = self.markov_boundary_matrix[var]
        var_mk_arr = np.flatnonzero(var_mk_bool_arr)
        var_mk_set = set(var_mk_arr)

        neighbor_bool_arr = np.copy(var_mk_bool_arr)

        for var_y in var_mk_arr:
            # check if Y is already neighbor of X
            if not self.learned_skeleton.has_edge(var, var_y):
                if not self.is_neighbor(var_name, var_y, var_mk_set):
                    neighbor_bool_arr[var_y] = 0

        # remove all variables that are not neighbors
        neighbors = np.flatnonzero(neighbor_bool_arr)
        return neighbors

    def is_neighbor(self, var_name: str, var_y: int, var_mk_set: Set[int]) -> bool:
        """Check if var_y is a neighbor of variable with name var_name using Lemma 27.

        Args:
            var_name (str): Name of the variable.
            var_y (int): The variable to check.
            var_mk_set (Set[int]): Set of the variables in the Markov boundary of var_name.

        Returns:
            bool: True if var_y is a neighbor, False otherwise.
        """

        var_mk_left_list = list(var_mk_set - {var_y})
        # use lemma 27 and check all proper subsets of Mb(X) - {Y}
        for cond_set_size in range(len(var_mk_left_list)):
            for var_s in itertools.combinations(var_mk_left_list, cond_set_size):
                cond_set = [self.var_names[idx] for idx in var_s]
                var_y_name = self.var_names[var_y]
                if self.ci_test(var_name, var_y_name, cond_set, self.data):
                    # we know that var_y is a co-parent and thus NOT a neighbor
                    return False
        return True

    def is_removable(self, var: int, neighbors: np.ndarray) -> bool:
        """Check whether a variable is removable using Theorem 32.

        Args:
            var (int): Index of the variable.
            neighbors (np.ndarray): Neighbors of the variable.

        Returns:
            bool: True if the variable is removable, False otherwise.
        """

        var_name = self.var_names[var]
        var_mk_bool_arr = self.markov_boundary_matrix[var]
        var_mk_arr = np.flatnonzero(var_mk_bool_arr)
        var_mk_set = set(var_mk_arr)

        def cond_1(var_y, var_z):
            # there exists subset W in Mb(X) - {Y, Z}, s.t. Y ind. Z | W
            var_mk_left_list = list(var_mk_set - {var_y, var_z})
            for cond_set_size in range(len(var_mk_left_list) + 1):
                for var_s in itertools.combinations(var_mk_left_list, cond_set_size):
                    cond_set = [self.var_names[idx] for idx in var_s]
                    if self.ci_test(var_y_name, var_z_name, cond_set, self.data):
                        return True
            return False

        def cond_2(var_y, var_z):
            # for all subset W in Mb(X) - {Y, Z}, s.t. Y NOT ind. Z | W + {X}
            var_mk_left_left = list(var_mk_set - {var_y, var_z})
            for cond_set_size in range(len(var_mk_left_left) + 1):
                for var_s in itertools.combinations(var_mk_left_left, cond_set_size):
                    cond_set = [self.var_names[idx] for idx in var_s] + [var_name]
                    if self.ci_test(var_y_name, var_z_name, cond_set, self.data):
                        return False
            return True

        # Use Theorem 32 to check if X is removable. Loop over Y in Mb(X) and Z in Ne(X)
        for var_y in var_mk_arr:
            var_y_name = self.var_names[var_y]
            for var_z in neighbors:
                var_z_name = self.var_names[var_z]
                if var_y == var_z:
                    continue
                xyz_tuple = (var, min(var_y, var_z_name), max(var_y, var_z))
                if xyz_tuple in self.skip_rem_check_set:
                    continue
                if not (cond_1(var_y, var_z) or cond_2(var_y, var_z)):
                    return False
                self.skip_rem_check_set.add(xyz_tuple)
        return True
