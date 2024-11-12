import itertools
import numpy as np
import networkx as nx
from typing import Callable, List, Set

from rcd.rsl.rsl_d import _RSLDiamondFree
from rcd.utilities.utils import *



def learn_and_get_skeleton(ci_test: Callable[[int, int, List[int], np.ndarray], bool],
                           data,
                           max_iters: int,
                           max_swaps: int,
                           initial_r_order: np.ndarray = None,
                           find_markov_boundary_matrix_fun=None) -> nx.Graph:
    """
    Learn the skeleton of a causal graph using the ROL hill climbing algorithm.

    Args:
        ci_test (Callable[[int, int, List[int], np.ndarray], bool]):
            A conditional independence test function that takes in the indices of two variables
            and a list of variable indices as the conditioning set, and returns True if the two
            variables are independent given the conditioning set, and False otherwise.
        data_matrix (np.ndarray):
            The data matrix with shape (num_samples, num_vars), where each column corresponds
            to a variable and each row corresponds to a sample.
        max_iters (int):
            Maximum number of iterations to run the algorithm for.
        max_swaps (int):
            Maximum swap distance to consider.
        initial_r_order (np.ndarray, optional):
            The initial r-order to use. If not provided, the algorithm will use
            RSL-D to find the initial r-order.

    Returns:
        nx.Graph: A networkx graph representing the learned skeleton.
    """
    data_matrix = sanitize_data(data)
    rol_hc = _ROLHillClimb(ci_test, max_iters, max_swaps, find_markov_boundary_matrix_fun=find_markov_boundary_matrix_fun)
    learned_skeleton = rol_hc.learn_and_get_skeleton(data_matrix, initial_r_order)
    return learned_skeleton

REMOVABLE_NOT_FOUND = -1


class _ROLHillClimb:
    """
    Implementation for the ROL hill climbing algorithm for learning causal graphs.

    This class is initialized with a conditional independence test function, which determines whether two variables
    are independent given another set of variables, using the data provided.

    The class has a learn_and_get_skeleton function that takes in a data matrix (numpy array), where each column
    corresponds to a variable and each row corresponds to a sample, and returns a networkx graph representing the
    learned skeleton.
    """

    def __init__(self, ci_test: Callable[[int, int, List[int], np.ndarray], bool],
                 max_iters: int,
                 max_swaps: int,
                 find_markov_boundary_matrix_fun: Callable[[np.ndarray], np.ndarray] = None):
        """
        Initialize the ROL hill climbing algorithm with the conditional independence test to use.

        Args:
            ci_test (Callable[[int, int, List[int], np.ndarray], bool]):
                A conditional independence test function that takes in the indices of two variables
                and a list of variable indices as the conditioning set, and returns True if the two
                variables are independent given the conditioning set, and False otherwise.
            max_iters (int):
                Maximum number of iterations to run the algorithm for.
            max_swaps (int):
                Maximum swap distance to consider.
            find_markov_boundary_matrix_fun (Callable[[np.ndarray], np.ndarray], optional):
                A function to find the Markov boundary matrix. It takes a numpy array of data,
                and returns a 2D numpy array. The (i, j)th entry is True if the jth variable is in the
                Markov boundary of the ith variable, and False otherwise.
        """
        if find_markov_boundary_matrix_fun is None:
            self.find_markov_boundary_matrix = compute_mb_gaussian
        else:
            self.find_markov_boundary_matrix = find_markov_boundary_matrix_fun

        self.num_vars = None
        self.data = None
        self.ci_test = ci_test
        self.max_iters = max_iters
        self.max_swaps = max_swaps

        # we use a flag array to keep track of which variables need to be checked for removal (i.e., we check if true)
        self.var_idx_set = None
        self.learned_skeleton = None

    def learn_and_get_skeleton(self, data: np.ndarray, initial_r_order: np.ndarray = None) -> nx.Graph:
        """
        Learn the skeleton of the graph using the ROL hill climbing algorithm.

        Args:
            data (np.ndarray):
                The data matrix with shape (num_samples, num_vars).
            initial_r_order (np.ndarray, optional):
                The initial r-order to use. If not provided, the algorithm will use
                RSL-D to find the initial r-order.

        Returns:
            nx.Graph: A networkx graph representing the learned skeleton.
        """
        self.num_vars = data.shape[1]
        self.data = data

        self.learned_skeleton = None

        if initial_r_order is None:
            # Set r-order by running RSL-D
            rol_init = _RSLDiamondFree(self.ci_test, self.find_markov_boundary_matrix)
            initial_r_order = rol_init.learn_and_get_skeleton(self.data,  return_r_order=True)

        # Find the best r-order and then learn the skeleton using it
        curr_r_order = np.copy(initial_r_order)
        curr_cost_vec = self.compute_cost(curr_r_order, 0, self.num_vars)
        total_swaps_made = 0
        for iter_num in range(self.max_iters):
            smaller_cost_found = False
            # For 1 <= a <= b <= self.num_vars such that b-a < self.max_swaps
            for a in range(self.num_vars):
                if smaller_cost_found:
                    break
                for b in range(a + 1, min(a + self.max_swaps + 1, self.num_vars)):
                    new_r_order = np.copy(curr_r_order)

                    # Swap variables at a and b
                    new_r_order[a], new_r_order[b] = new_r_order[b], new_r_order[a]

                    # Compute cost of r-order from a to b
                    new_cost_vec = self.compute_cost(new_r_order, a, b)

                    if new_cost_vec[a:b].sum() < curr_cost_vec[a:b].sum():
                        # Update r-order
                        curr_r_order = new_r_order
                        curr_cost_vec[a:b] = new_cost_vec[a:b]
                        smaller_cost_found = True
                        total_swaps_made += 1
                        break

            if not smaller_cost_found:
                break

        self.learned_skeleton = self.learn_skeleton_using_r_order(curr_r_order)
        return self.learned_skeleton

    def learn_skeleton_using_r_order(self, r_order: np.ndarray) -> nx.Graph:
        """
        Learns the skeleton of the graph using the given r-order.

        Args:
            r_order (np.ndarray):
                The r-order to use for learning the skeleton.

        Returns:
            nx.Graph: A networkx graph representing the learned skeleton.
        """
        # Initialize graph
        learned_skeleton = nx.Graph()
        learned_skeleton.add_nodes_from(range(self.num_vars))

        markov_boundary = self.find_markov_boundary_matrix(self.data)

        for var in r_order:
            # Learn the neighbors of the variable and then remove it from the graph
            neighbors = self.find_neighborhood(var, markov_boundary[var])
            for neighbor in neighbors:
                learned_skeleton.add_edge(var, neighbor)

        return learned_skeleton

    def compute_cost(self, r_order: np.ndarray, starting_index: int, ending_index: int) -> np.ndarray:
        """
        Compute the cost of the given r-order between the specified starting and ending indices.

        Args:
            r_order (np.ndarray):
                The r-order to compute the cost of.
            starting_index (int):
                The starting index of the r-order to compute the cost of.
            ending_index (int):
                The ending index (exclusive) of the r-order to compute the cost of.

        Returns:
            np.ndarray: The cost of the given r-order between the specified starting and ending indices.
        """
        remaining_vars_mkb = sorted(r_order[starting_index:])
        cost_vec = np.zeros(len(r_order))

        # Restrict data to the remaining variables
        remaining_data = self.data[:, remaining_vars_mkb]

        sub_markov_boundary = self.find_markov_boundary_matrix(remaining_data)
        markov_boundary = np.zeros((self.num_vars, self.num_vars), dtype=bool)
        markov_boundary[np.ix_(remaining_vars_mkb, remaining_vars_mkb)] = sub_markov_boundary

        data_included_ci_test = lambda x, y, z: self.ci_test(x, y, z, self.data)

        for index in range(starting_index, ending_index):
            removable_var = r_order[index]

            neighbors = self.find_neighborhood(removable_var, markov_boundary[removable_var])

            cost_vec[index] = len(neighbors)

            # Update Markov boundary matrix
            update_markov_boundary_matrix(
                markov_boundary,
                data_included_ci_test,
                removable_var,
                neighbors,
            )
        return cost_vec

    def find_neighborhood(self, var: int, var_mk_bool_arr) -> np.ndarray:
        """
        Find the neighborhood of a variable using Proposition 40.

        Args:
            var (int): The variable whose neighborhood we want to find.

        Returns:
            np.ndarray: 1D numpy array containing the variables in the neighborhood.
        """
        var_mk_arr = np.flatnonzero(var_mk_bool_arr)
        var_mk_set = set(var_mk_arr)

        # loop through markov boundary matrix row corresponding to var_name
        # use Proposition 40: var_y is Y and var_z is Z. cond_set is Mb(X) - {Y, Z}

        # First, assume all variables are neighbors
        neighbor_bool_arr = np.copy(var_mk_bool_arr)

        for var_y in var_mk_arr:
            for var_z in var_mk_arr:
                if var_y == var_z:
                    continue
                cond_set = list(var_mk_set - {var_y, var_z})

                if self.ci_test(var, var_y, cond_set, self.data):
                    # we know that var_y is a co-parent and thus NOT a neighbor
                    neighbor_bool_arr[var_y] = False
                    break

        neighbor_arr = np.flatnonzero(neighbor_bool_arr)
        return neighbor_arr

    def find_neighbors(self, var: int, var_mk_bool_arr: np.ndarray) -> np.ndarray:
        """
        Find the neighborhood of a variable using Lemma 27.

        Args:
            var (int):
                Index of the variable in the data.
            var_mk_bool_arr (np.ndarray):
                Markov boundary of the variable.

        Returns:
            np.ndarray: 1D numpy array containing the indices of the variables in the neighborhood.
        """
        var_mk_arr = np.flatnonzero(var_mk_bool_arr)
        var_mk_set = set(var_mk_arr)

        neighbor_bool_arr = np.copy(var_mk_bool_arr)

        for var_y in var_mk_arr:
            # Check if Y is already a neighbor of X
            if not self.is_neighbor(var, var_y, var_mk_set):
                neighbor_bool_arr[var_y] = False

        # Remove all variables that are not neighbors
        neighbors = np.flatnonzero(neighbor_bool_arr)
        return neighbors

    def is_neighbor(self, var: int, var_y: int, var_mk_set: Set[int]) -> bool:
        """
        Check if var_y is a neighbor of variable var using Lemma 27.

        Args:
            var (int):
                Index of the variable.
            var_y (int):
                The variable to check.
            var_mk_set (Set[int]):
                Set of the variables in the Markov boundary of var.

        Returns:
            bool: True if var_y is a neighbor, False otherwise.
        """
        var_mk_left_list = list(var_mk_set - {var_y})
        # Use Lemma 27 and check all proper subsets of Mb(X) - {Y}
        for cond_set_size in range(len(var_mk_left_list)):
            for var_s in itertools.combinations(var_mk_left_list, cond_set_size):
                cond_set = list(var_s)
                if self.ci_test(var, var_y, cond_set, self.data):
                    # Y is a co-parent and thus NOT a neighbor
                    return False
        return True
