import itertools
import numpy as np
import networkx as nx
from typing import Callable, List, Set, Dict


from rcd.utilities.utils import sanitize_data, compute_mb_gaussian, sort_vars_by_mkb_size, update_markov_boundary_matrix

REMOVABLE_NOT_FOUND = -1

def learn_and_get_skeleton(ci_test: Callable[[int, int, List[int], np.ndarray], bool],
                           data,
                           find_markov_boundary_matrix_fun: Callable[[np.ndarray], np.ndarray] = None) -> nx.Graph:
    """
    Learn the skeleton of a causal graph using the MARVEL algorithm.

    Args:
        ci_test (Callable[[int, int, List[int], np.ndarray], bool]):
            A conditional independence test function that takes in the indices of two variables
            and a list of variable indices as the conditioning set, and returns True if the two
            variables are independent given the conditioning set, and False otherwise.
        data_matrix (np.ndarray):
            The data matrix with shape (num_samples, num_vars), where each column corresponds
            to a variable and each row corresponds to a sample.
        find_markov_boundary_matrix_fun (Callable[[np.ndarray], np.ndarray], optional):
            A function to find the Markov boundary matrix. It takes a numpy array of data,
            and returns a 2D numpy array. The (i, j)th entry is True if the jth variable is in the
            Markov boundary of the ith variable, and False otherwise.

    Returns:
        nx.Graph: A networkx graph representing the learned skeleton.
    """
    data_matrix = sanitize_data(data)
    marvel = _Marvel(ci_test, find_markov_boundary_matrix_fun)
    learned_skeleton = marvel.learn_and_get_skeleton(data_matrix)
    return learned_skeleton


class _Marvel:
    """
    Implementation for the MARVEL algorithm for learning causal graphs with latent variables.

    This class is initialized with a conditional independence test function, which determines whether two variables
    are independent given another set of variables, using the data provided.

    The class has a learn_and_get_skeleton function that takes in a data matrix (numpy array), where each column
    corresponds to a variable and each row corresponds to a sample, and returns a networkx graph representing the
    learned skeleton.
    """

    def __init__(self, ci_test: Callable[[int, int, List[int], np.ndarray], bool],
                 find_markov_boundary_matrix_fun: Callable[[np.ndarray], np.ndarray] = None):
        """
        Initialize the MARVEL algorithm with the conditional independence test to use.

        Args:
            ci_test (Callable[[int, int, List[int], np.ndarray], bool]):
                A conditional independence test function. It takes the indices of two variables
                and a list of variable indices as the conditioning set, and returns True if the two
                variables are independent given the conditioning set, and False otherwise.
                Signature:
                    ci_test(var1: int, var2: int, cond_set: List[int], data: np.ndarray) -> bool.
            find_markov_boundary_matrix_fun (Callable[[np.ndarray], np.ndarray], optional):
                A function to find the Markov boundary matrix. It takes a numpy array of data,
                and returns a 2D numpy array. The (i, j)th entry is True if the jth variable is in the
                Markov boundary of the ith variable, and False otherwise.
                Signature:
                    find_markov_boundary_matrix_fun(data: np.ndarray) -> np.ndarray.
        """
        if find_markov_boundary_matrix_fun is None:
            self.find_markov_boundary_matrix = lambda data: compute_mb_gaussian(data)
        else:
            self.find_markov_boundary_matrix = find_markov_boundary_matrix_fun

        self.num_vars = None
        self.data = None
        self.ci_test = ci_test

        # we use a flag array to keep track of which variables need to be checked for removal (i.e., we check if False)
        self.skip_rem_check_vec = None  # SkipCheck_VEC in the paper

        # we use a flag array to keep track of which variables' neighbors need to be learned (i.e., we learn if False)
        self.neighbor_learned_arr = None

        self.skip_check_cond1_set = None
        self.skip_check_cond2_set = None

        # we use a dictionary that maps x to a dictionary that maps y to a set of variables v, such that x->v<-y is a v-structure
        self.v_structure_dict: Dict[int, Dict[int, Set[int]]] = None

        # we use a flag array to keep track of which variables' v-structures need to be learned (i.e., we learn if True)
        self.v_structure_learned_arr = None

        self.var_idx_set = None
        self.markov_boundary_matrix = None
        self.learned_skeleton: nx.Graph | None = None

    def learn_and_get_skeleton(self, data: np.ndarray) -> nx.Graph:
        """
        Run the MARVEL algorithm on the data to learn and return the learned skeleton graph.

        Args:
            data (np.ndarray):
                The data matrix with shape (num_samples, num_vars).

        Returns:
            nx.Graph: A networkx graph representing the learned skeleton.
        """
        # Initialize algorithm state
        self.num_vars = data.shape[1]
        self.data = data

        self.skip_rem_check_vec = np.zeros(self.num_vars, dtype=bool)
        self.skip_check_cond1_set = set()
        self.skip_check_cond2_set = set()
        self.neighbor_learned_arr = np.zeros(self.num_vars, dtype=bool)
        self.v_structure_dict = dict()
        self.v_structure_learned_arr = np.zeros(self.num_vars, dtype=bool)
        self.var_idx_set = set(range(self.num_vars))
        self.markov_boundary_matrix = self.find_markov_boundary_matrix(self.data)
        self.learned_skeleton = nx.Graph()
        self.learned_skeleton.add_nodes_from(range(self.num_vars))

        data_included_ci_test = lambda x, y, z: self.ci_test(x, y, z, self.data)

        var_idx_arr = np.arange(self.num_vars)

        var_left_bool_arr = np.ones(self.num_vars, dtype=bool)  # Indicates if variable is left

        x_y_sep_set_dict = dict()  # maps x to a dictionary that maps y to the separating set of x and y

        for _ in range(self.num_vars - 1):
            # Sort variables by decreasing Markov boundary size
            # Only sort variables that are still left and whose removability has NOT been checked
            var_to_sort_bool_arr = var_left_bool_arr & ~self.skip_rem_check_vec
            var_idx_to_sort_arr = var_idx_arr[var_to_sort_bool_arr]
            sorted_var_idx = sort_vars_by_mkb_size(self.markov_boundary_matrix[var_to_sort_bool_arr],
                                                   var_idx_to_sort_arr)

            removable_found = False
            for var_idx in sorted_var_idx:
                var_mk_idxs = np.flatnonzero(self.markov_boundary_matrix[var_idx])
                # Check whether we need to learn the neighbors of var_idx
                if not self.neighbor_learned_arr[var_idx]:
                    neighbors, co_parents_arr, y_sep_set_dict = self.find_neighborhood(var_idx)
                    self.neighbor_learned_arr[var_idx] = True
                    x_y_sep_set_dict[var_idx] = y_sep_set_dict

                    # Add edges between the variable and its neighbors
                    for neighbor_idx in neighbors:
                        self.learned_skeleton.add_edge(var_idx, neighbor_idx)
                else:
                    # if neighbors already learned, get them from the graph
                    neighbors = self.learned_skeleton.neighbors(var_idx)

                    # Ensure only to include neighbors that are still left
                    neighbors = [neighbor for neighbor in neighbors if var_left_bool_arr[neighbor]]

                    # get the separating sets from the dictionary
                    y_sep_set_dict = x_y_sep_set_dict[var_idx]

                    # Co-parents are markov boundary variables that are not neighbors
                    co_parents_bool_arr = np.copy(self.markov_boundary_matrix[var_idx])
                    co_parents_bool_arr[neighbors] = False
                    co_parents_arr = np.flatnonzero(co_parents_bool_arr)

                # Check if variable is removable
                if self.cond_1(var_idx, neighbors, var_mk_idxs):
                    if not self.v_structure_learned_arr[var_idx]:
                        self.learn_v_structure(var_idx, neighbors, co_parents_arr, var_mk_idxs, y_sep_set_dict)
                        x_v_structure_dict = self.v_structure_dict.get(var_idx, dict())
                    else:
                        # only keep y and z that are left that form a v-structure: x->z<-y
                        var_left_set = set(sorted_var_idx)
                        x_v_structure_dict = self.v_structure_dict[var_idx]
                        for var_y in list(x_v_structure_dict.keys()):
                            if not var_left_bool_arr[var_y]:
                                del x_v_structure_dict[var_y]
                            else:
                                x_v_structure_dict[var_y] = x_v_structure_dict[var_y].intersection(var_left_set)

                    if self.cond_2(var_idx, neighbors, co_parents_arr, var_mk_idxs, x_v_structure_dict):
                        # Remove the removable variable from the set of variables left
                        var_left_bool_arr[var_idx] = False

                        # Update the Markov boundary matrix
                        update_markov_boundary_matrix(self.markov_boundary_matrix,
                                                      data_included_ci_test,
                                                      var_idx,
                                                      neighbors,
                                                      skip_check=self.skip_rem_check_vec)
                        removable_found = True
                        break
                    else:
                        self.skip_rem_check_vec[var_idx] = True
                else:
                    self.skip_rem_check_vec[var_idx] = True

            if not removable_found:
                # If no removable found, pick the variable with the smallest Markov boundary from var_left_bool_arr
                var_left_arr = np.flatnonzero(var_left_bool_arr)
                mb_size_all = np.sum(self.markov_boundary_matrix[var_left_arr], axis=1)
                var_idx = var_left_arr[np.argmin(mb_size_all)]

                neighbors = self.learned_skeleton.neighbors(var_idx)

                # Ensure only to include neighbors that are still left
                neighbors = [neighbor for neighbor in neighbors if var_left_bool_arr[neighbor]]
                var_left_bool_arr[var_idx] = False

                # update the markov boundary matrix
                update_markov_boundary_matrix(self.markov_boundary_matrix,
                                              data_included_ci_test,
                                              var_idx,
                                              neighbors,
                                              skip_check=self.skip_rem_check_vec)
                self.skip_rem_check_vec[:] = False
        return self.learned_skeleton

    def find_neighborhood(self, var_idx: int):
        """Find the neighborhood of a variable using Lemma 27.

        Args:
            var_idx (int): The variable whose neighborhood we want to find.

        Returns:
            tuple:
                - np.ndarray: 1D numpy array containing the variables in the neighborhood.
                - np.ndarray: 1D numpy array containing the co-parents of the variable.
                - Dict[int, Set[int]]: A dictionary mapping co-parent indices to their separating sets.
        """
        var_mk_arr = np.flatnonzero(self.markov_boundary_matrix[var_idx])
        var_mk_set = set(var_mk_arr)

        neighbors_bool_arr = np.copy(self.markov_boundary_matrix[var_idx])
        co_parents_bool_arr = np.zeros(len(neighbors_bool_arr), dtype=bool)
        y_sep_set_dict = dict()

        for mb_idx_y in range(len(var_mk_arr)):
            var_y_idx = var_mk_arr[mb_idx_y]
            # Check if Y is already a neighbor of X
            if not self.learned_skeleton.has_edge(var_idx, var_y_idx):
                x_y_sep_set = self.get_sep_set(var_idx, var_y_idx, var_mk_arr)
                if x_y_sep_set is not None:
                    # var_y is a co-parent of var_idx and thus NOT a neighbor
                    neighbors_bool_arr[var_y_idx] = False
                    co_parents_bool_arr[var_y_idx] = True
                    y_sep_set_dict[var_y_idx] = x_y_sep_set

        # Remove all variables that are not neighbors
        neighbors_arr = np.flatnonzero(neighbors_bool_arr)
        co_parents_arr = np.flatnonzero(co_parents_bool_arr)
        return neighbors_arr, co_parents_arr, y_sep_set_dict

    def get_sep_set(self, var_idx: int, var_y_idx: int, var_x_mk_idxs: np.ndarray) -> Set[int] | None:
        var_mk_left_idxs = list(set(var_x_mk_idxs) - {var_y_idx})
        # Use Lemma 27 and check all proper subsets of Mb(X) - {Y}
        for cond_set_size in range(len(var_mk_left_idxs) + 1):
            for var_s_idxs in itertools.combinations(var_mk_left_idxs, cond_set_size):
                if self.ci_test(var_idx, var_y_idx, list(var_s_idxs), self.data):
                    # Y is a co-parent and thus NOT a neighbor
                    return set(var_s_idxs)
        return None

    def cond_1(self, var_idx, neighbors, var_mk_idxs):
        num_neighbors = len(neighbors)
        for var_y_idx in range(num_neighbors - 1):
            var_y = neighbors[var_y_idx]
            for var_z_idx in range(var_y_idx + 1, num_neighbors):
                var_z = neighbors[var_z_idx]
                xyz_tuple = (var_idx, var_y, var_z)
                if xyz_tuple in self.skip_check_cond1_set:
                    continue
                # If skip check is false, loop over all subsets S of Mb(X) - {Y, Z} and check if Y ind. Z | S + {X}
                var_mk_left_idxs = list(set(var_mk_idxs) - {var_y, var_z})
                for cond_set_size in range(len(var_mk_left_idxs) + 1):
                    for var_s_idxs in itertools.combinations(var_mk_left_idxs, cond_set_size):
                        cond_set = list(var_s_idxs) + [var_idx]
                        if self.ci_test(var_y, var_z, cond_set, self.data):
                            return False
                self.skip_check_cond1_set.add(xyz_tuple)
        return True

    def cond_2(self, var_idx, neighbors, co_parents_arr, var_mk_idxs, x_v_structure_dict):
        for var_y in co_parents_arr:
            for var_z in neighbors:
                xyz_tuple = (var_idx, var_y, var_z)
                if xyz_tuple in self.skip_check_cond2_set:
                    continue
                # If skip check is false, loop over all v such that x->v<-y is a v-structure
                for var_v in x_v_structure_dict.get(var_y, set()):
                    if var_v == var_z:
                        continue
                    # Loop over all subsets s of Mb(X) - {V, Y, Z} and check if Y ind. Z | S + {X, V}
                    var_mk_left_idxs = list(set(var_mk_idxs) - {var_y, var_z, var_v})
                    for cond_set_size in range(len(var_mk_left_idxs) + 1):
                        for var_s_idxs in itertools.combinations(var_mk_left_idxs, cond_set_size):
                            cond_set = list(var_s_idxs) + [var_idx, var_v]
                            if self.ci_test(var_y, var_z, cond_set, self.data):
                                return False
                self.skip_check_cond2_set.add(xyz_tuple)
        return True


    def learn_v_structure(self, var_idx: int, neighbors: List[int], co_parents_arr: np.ndarray,
                          var_mk_idxs: np.ndarray, y_sep_set_dict: Dict[int, Set[int]]):
        """
        Learns the v-structures of a given variable.

        Args:
            var_idx (int): The index of the variable for which to learn the v-structures.
            neighbors (List[int]): A list of indices representing the neighbors of the variable.
            co_parents_arr (np.ndarray): A list of indices representing the co-parents of the variable.
            var_mk_idxs (np.ndarray): A list of indices representing the variables in the Markov boundary of the variable.
            y_sep_set_dict (Dict[int, Set[int]]): A dictionary mapping indices of other variables to the separating sets
                that distinguish them from the current variable.
        """

        def is_y_z_neighbor(var_y: int, var_z: int) -> bool:
            if self.learned_skeleton.has_edge(var_y, var_z):
                return True
            # check that all subsets S in Mb(X) + {X} - {Y, Z} satisfy Y NOT ind. Z | S
            var_mk_left_idxs = list(set(var_mk_idxs) - {var_y, var_z}) + [var_idx]
            for cond_set_size in range(len(var_mk_left_idxs) + 1):
                for var_s_idxs in itertools.combinations(var_mk_left_idxs, cond_set_size):
                    cond_set = list(var_s_idxs)
                    if self.ci_test(var_y, var_z, cond_set, self.data):
                        return False

            # Add edge in skeleton
            self.learned_skeleton.add_edge(var_y, var_z)
            return True

        for var_y in co_parents_arr:
            for var_z in neighbors:
                sep_set = y_sep_set_dict[var_y]
                if var_z not in sep_set and is_y_z_neighbor(var_y, var_z):
                    x_v_structure_dict = self.v_structure_dict.get(var_idx, dict())
                    z_set = x_v_structure_dict.get(var_y, set())
                    z_set.add(var_z)
                    x_v_structure_dict[var_y] = z_set
                    self.v_structure_dict[var_idx] = x_v_structure_dict
        self.v_structure_learned_arr[var_idx] = True
