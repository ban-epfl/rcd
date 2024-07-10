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


class RSLBase:
    def __init__(self, ci_test, find_markov_boundary_matrix_fun=None):
        """Initialize the rsl algorithm with the conditional independence test to use.

        Args:
            ci_test: A conditional independence test function that takes in the names of two variables and a list of
                     variable names as the conditioning set, and returns True if the two variables are independent given
                     the conditioning set, and False otherwise. The function's signature should be:
                     ci_test(var_name1: str, var_name2: str, cond_set: List[str], data: pd.DataFrame) -> bool
            find_markov_boundary_matrix_fun (optional): A function to find the Markov boundary matrix. This function should
                                                         take in a Pandas DataFrame of data, and return a 2D numpy array,
                                                         where the (i, j)th entry is True if the jth variable is in the Markov
                                                         boundary of the ith variable, and False otherwise. The function's
                                                         signature should be:
                                                         find_markov_boundary_matrix_fun(data: pd.DataFrame) -> np.ndarray
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
        self.markov_boundary_matrix = None
        self.learned_skeleton = None
        self.is_rsl_d = False
        self.clique_num = None

    def reset_fields(self, data: pd.DataFrame, clique_num: int = None):
        """Reset the algorithm before running it on new data.

        Args:
            data (pd.DataFrame): The data to reset the algorithm with.
            clique_num (int, optional): The clique number of the graph. Only used for rsl-W.
        """

        self.num_vars = len(data.columns)
        self.data = data
        self.var_names = data.columns.tolist()

        self.skip_rem_check_vec = np.zeros(self.num_vars, dtype=bool)
        self.markov_boundary_matrix = None
        self.learned_skeleton = None
        self.clique_num = clique_num

    def has_alg_run(self) -> bool:
        """Check if the algorithm has been run.

        Returns:
            bool: True if the algorithm has been run, False otherwise.
        """

        return self.learned_skeleton is not None

    def learn_and_get_skeleton(self, data: pd.DataFrame, clique_num: int = None) -> nx.Graph:
        """Run the rsl algorithm on the data to learn and return the learned skeleton graph.

        Args:
            data (pd.DataFrame): The data to learn the skeleton from.
            clique_num (int, optional): The clique number of the graph, used only for specific versions of the algorithm.

        Returns:
            nx.Graph: A networkx graph representing the learned skeleton.
        """

        # if RSL-W and clique_num is not None, throw an error
        if not self.is_rsl_d and clique_num is None:
            raise ValueError("Clique number not given!")

        self.reset_fields(data, clique_num)

        # initialize graph
        skeleton = nx.Graph()
        skeleton.add_nodes_from(self.var_names)

        data_included_ci_test = lambda x, y, z: self.ci_test(x, y, z, self.data)

        self.markov_boundary_matrix = self.find_markov_boundary_matrix(self.data)

        var_arr = np.arange(self.num_vars)
        var_left_bool_arr = np.ones(self.num_vars, dtype=bool)  # if ith position is True, indicates that i is left

        for _ in range(self.num_vars - 1):
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
            update_markov_boundary_matrix(self.markov_boundary_matrix, self.skip_rem_check_vec, self.var_names,
                                          data_included_ci_test, removable_var, neighbors, self.is_rsl_d)

            # add edges between the removable variable and its neighbors
            for neighbor_idx in neighbors:
                skeleton.add_edge(self.var_names[removable_var], self.var_names[neighbor_idx])

            # remove the removable variable from the set of variables left
            var_left_bool_arr[removable_var] = False

        self.learned_skeleton = skeleton
        return skeleton

    def find_neighborhood(self, var: int) -> np.ndarray:
        """Find the neighborhood of a variable.

        Args:
            var (int): The variable whose neighborhood we want to find.

        Returns:
            np.ndarray: 1D numpy array containing the variables in the neighborhood.
        """

        raise NotImplementedError()

    def is_removable(self, var: int) -> bool:
        """Check whether a variable is removable.

        Args:
            var (int): The variable to check.

        Returns:
            bool: True if the variable is removable, False otherwise.
        """

        raise NotImplementedError()

    def find_removable(self, var_arr: np.ndarray) -> int:
        """Find a removable variable in the given list of variables.

        Args:
            var_arr (np.ndarray): 1D array of variables.

        Returns:
            int: The index of the removable variable, if found, and a constant (e.g., REMOVABLE_NOT_FOUND) if not found.
        """

        for var in var_arr:
            if self.is_removable(var):
                return var
            self.skip_rem_check_vec[var] = True

        return REMOVABLE_NOT_FOUND
