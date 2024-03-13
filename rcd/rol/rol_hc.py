import itertools
from typing import Set

from rcd.rol.rol_init import ROLInitializer
from rcd.utilities.utils import *


# TODO fix variable names in ROL

class ROLHillClimb:
    def __init__(self, ci_test, max_iters: int, max_swaps: int, find_markov_boundary_matrix_fun=None):
        """Initialize the ROL hill climbing algorithm with the conditional independence test to use.

        Args:
            ci_test: A conditional independence test function that takes in the names of two variables and a list of
                     variable names as the conditioning set, and returns True if the two variables are independent given
                     the conditioning set, and False otherwise. The function's signature should be:
                     ci_test(var_name1: str, var_name2: str, cond_set: List[str], data: pd.DataFrame) -> bool
            max_iters (int): Maximum number of iterations to run the algorithm for.
            max_swaps (int): Maximum swap distance to consider.
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
        self.max_iters = max_iters
        self.max_swaps = max_swaps

        # we use a flag array to keep track of which variables need to be checked for removal (i.e., we check if true)
        self.var_idx_set = None
        self.learned_skeleton = None

    def reset_fields(self, data: pd.DataFrame):
        """Reset the algorithm before running it on new data.

        Args:
            data (pd.DataFrame): The data to reset the algorithm with.
        """

        self.num_vars = len(data.columns)
        self.data = data
        self.var_names = data.columns

        self.learned_skeleton = None

    def has_alg_run(self) -> bool:
        """Check if the algorithm has been run.

        Returns:
            bool: True if the algorithm has been run, False otherwise.
        """
        return self.learned_skeleton is not None

    def learn_and_get_skeleton(self, data: pd.DataFrame, initial_r_order: np.ndarray = None) -> nx.Graph:
        """Learn the skeleton of the graph using the ROL hill climbing algorithm.

        Args:
            data (pd.DataFrame): The data to learn the skeleton from.
            initial_r_order (np.ndarray, optional): The initial r-order to use. If not provided, the algorithm will use
                                                     RSL-D to find the initial r-order

        Returns:
            nx.Graph: A networkx graph representing the learned skeleton.
        """

        self.reset_fields(data)



        if initial_r_order is None:
            # set r-order by running RSL-D
            rol_init = ROLInitializer(self.ci_test)
            initial_r_order = rol_init.learn_and_get_r_order(self.data)

        # find the best r-order and then learn the skeleton using it
        curr_r_order = np.copy(initial_r_order)
        curr_cost_vec = self.compute_cost(curr_r_order, 0, self.num_vars)
        total_swaps_made = 0
        for iter_num in range(self.max_iters):
            smaller_cost_found = False
            # for 1 <= a <= b <= self.num_vars such that b-a < self.max_swaps
            for a in range(self.num_vars):
                if smaller_cost_found:
                    break
                for b in range(a + 2, min(a + self.max_swaps + 1, self.num_vars)):
                    new_r_order = np.copy(curr_r_order)

                    # swap variables at a and b
                    new_r_order[a], new_r_order[b] = new_r_order[b], new_r_order[a]

                    # compute cost of r-order from a to b
                    new_cost_vec = self.compute_cost(curr_r_order, a, b)

                    if new_cost_vec[a:b].sum() < curr_cost_vec[a:b].sum():
                        # update r-order
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
        """Learns the skeleton of the graph using the given r-order.

        Args:
            r_order (np.ndarray): The r-order to use for learning the skeleton.

        Returns:
            nx.Graph: A networkx graph representing the learned skeleton.
        """

        # initialize graph
        learned_skeleton = nx.Graph()
        learned_skeleton.add_nodes_from(self.var_names)

        markov_boundary = self.find_markov_boundary_matrix(self.data)

        for var in r_order:
            # learn the neighbors of the variable and then remove it from the graph
            neighbors = self.find_neighbors(var, markov_boundary[var])
            for neighbor in neighbors:
                learned_skeleton.add_edge(self.var_names[var], self.var_names[neighbor])

        return learned_skeleton

    def compute_cost(self, r_order: np.ndarray, starting_index: int, ending_index: int) -> np.ndarray:
        """Compute the cost of the given r-order between the specified starting and ending indices.

        Args:
            r_order (np.ndarray): The r-order to compute the cost of.
            starting_index (int): The starting index of the r-order to compute the cost of.
            ending_index (int): The ending index (exclusive) of the r-order to compute the cost of.

        Returns:
            np.ndarray: The cost of the given r-order between the specified starting and ending indices.
        """

        remaining_vars_mkb = sorted(r_order[starting_index:])
        cost_vec = np.zeros(len(r_order))

        # restrict data to the remaining variables
        remaining_data = self.data.iloc[:, remaining_vars_mkb]  # TODO move outside of compute cost

        sub_markov_boundary = self.find_markov_boundary_matrix(remaining_data)
        markov_boundary = np.zeros((self.num_vars, self.num_vars), dtype=bool)
        markov_boundary[np.ix_(remaining_vars_mkb, remaining_vars_mkb)] = sub_markov_boundary

        for index in range(starting_index, ending_index):
            var_to_remove = r_order[index]

            neighbors = self.find_neighbors(var_to_remove, markov_boundary[var_to_remove])

            cost_vec[index] = len(neighbors)

            # update markov boundary matrix
            markov_boundary = self.update_markov_boundary_matrix(markov_boundary, var_to_remove, neighbors)
        return cost_vec

    def find_neighbors(self, var: int, var_mk_bool_arr: np.ndarray) -> np.ndarray:
        """Find the neighborhood of a variable using Lemma 27.

        Args:
            var (int): Index of the variable in the data.
            var_mk_bool_arr (np.ndarray): Markov boundary of the variable.

        Returns:
            np.ndarray: 1D numpy array containing the indices of the variables in the neighborhood.
        """

        var_name = self.var_names[var]
        var_mk_arr = np.flatnonzero(var_mk_bool_arr)
        var_mk_set = set(var_mk_arr)

        neighbor_bool_arr = np.copy(var_mk_bool_arr)

        for var_y in var_mk_arr:
            # check if Y is already neighbor of X
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

        # var_mk_left_idxs = list(set(var_x_mk_idxs) - {var_y_idx})
        # # use lemma 27 and check all proper subsets of Mb(X) - {Y}
        # for cond_set_size in range(len(var_mk_left_idxs)):
        #     for var_s_idxs in itertools.combinations(var_mk_left_idxs, cond_set_size):
        #         cond_set = [self.var_names[idx] for idx in var_s_idxs]
        #         var_y_name = self.var_names[var_y_idx]
        #         if self.ci_test(var_name, var_y_name, cond_set, self.data):
        #             # we know that var_y_idx is a co-parent and thus NOT a neighbor
        #             return False
        # return True

    def update_markov_boundary_matrix(self, markov_boundary_matrix, var_idx: int, var_neighbors: np.ndarray):
        """
        Update the Markov boundary matrix after removing a variable.
        :param var_idx: Index of the variable to remove
        :param var_neighbors: 1D numpy array containing the neighbors of var_idx
        """

        var_markov_boundary = np.flatnonzero(markov_boundary_matrix[var_idx])

        # for every variable in the markov boundary of var_idx, remove it from the markov boundary and update flag
        for mb_var_idx in np.flatnonzero(markov_boundary_matrix[var_idx]):  # TODO use indexing instead
            markov_boundary_matrix[mb_var_idx, var_idx] = 0
            markov_boundary_matrix[var_idx, mb_var_idx] = 0

        # find nodes whose co-parent status changes
        # we only remove Y from mkvb of Z iff X is their ONLY common child and they are NOT neighbors)
        for ne_idx_y in range(len(var_neighbors) - 1):  # -1 because no need to check last variable and also symmetry
            for ne_idx_z in range(ne_idx_y + 1, len(var_neighbors)):
                var_y_idx = var_neighbors[ne_idx_y]
                var_z_idx = var_neighbors[ne_idx_z]
                var_y_name = self.var_names[var_y_idx]
                var_z_name = self.var_names[var_z_idx]

                # determine whether the mkbv of var_y_idx or var_z_idx is smaller, and use the smaller one as cond_set
                var_y_markov_boundary = np.flatnonzero(markov_boundary_matrix[var_y_idx])
                var_z_markov_boundary = np.flatnonzero(markov_boundary_matrix[var_z_idx])
                if np.sum(markov_boundary_matrix[var_y_idx]) < np.sum(markov_boundary_matrix[var_z_idx]):
                    cond_set = [self.var_names[idx] for idx in set(var_y_markov_boundary) - {var_z_idx}]
                else:
                    cond_set = [self.var_names[idx] for idx in set(var_z_markov_boundary) - {var_y_idx}]

                if self.ci_test(var_y_name, var_z_name, cond_set, self.data):
                    # we know that Y and Z are co-parents and thus NOT neighbors
                    markov_boundary_matrix[var_y_idx, var_z_idx] = 0
                    markov_boundary_matrix[var_z_idx, var_y_idx] = 0

        return markov_boundary_matrix
