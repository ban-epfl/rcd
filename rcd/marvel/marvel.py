import itertools
from typing import Set, Dict

from rcd.utilities.utils import *

"""This file contains the base class implementation for MARVEL for learning causal graphs. The class is initialized 
with a conditional independence test function, which determines whether two variables are independent given another 
set of variables, using the data provided. For examples of possible conditional independence tests, 
see utilities/ci_tests.py. For details on how to write a custom CI test function, look at the constructor (init 
function) of the RSLBase class below.

The class has a learn_and_get_skeleton function that takes in a Pandas DataFrame of data, where the ith column
contains samples from the ith variable, and returns a networkx graph representing the learned skeleton.
"""


class Marvel:
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
        self.skip_check_cond1_set = set()
        self.skip_check_cond2_set = set()
        self.neighbor_learned_arr = np.zeros(self.num_vars, dtype=bool)
        self.v_structure_dict = dict()
        self.v_structure_learned_arr = np.zeros(self.num_vars, dtype=bool)
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
        """Run the marvel algorithm on the data to learn and return the learned skeleton graph.

        Args:
            data (pd.DataFrame): The data to learn the skeleton from.

        Returns:
            nx.Graph: A networkx graph representing the learned skeleton.
        """
        self.reset_fields(data)

        # initialize graph
        self.learned_skeleton = nx.Graph()
        self.learned_skeleton.add_nodes_from(self.var_names)

        self.markov_boundary_matrix = self.find_markov_boundary_matrix(self.data)

        var_idx_arr = np.arange(self.num_vars)

        var_left_bool_arr = np.ones(len(self.var_names),
                                    dtype=bool)  # if ith position is True, indicates that i is left

        x_y_sep_set_dict = dict()  # maps x to a dictionary that maps y to the separating set of x and y

        for _ in range(self.num_vars - 1):
            # sort variables by decreasing Markov boundary size
            # only sort variables that are still left and whose removability has NOT been checked
            var_to_sort_bool_arr = var_left_bool_arr & ~self.skip_rem_check_vec
            var_idx_to_sort_arr = var_idx_arr[var_to_sort_bool_arr]
            sorted_var_idx = sort_vars_by_mkb_size(self.markov_boundary_matrix[var_to_sort_bool_arr],
                                                   var_idx_to_sort_arr)

            removable_found = False
            for var_idx in sorted_var_idx:
                var_mk_idxs = np.flatnonzero(self.markov_boundary_matrix[var_idx])
                # check whether we need to learn the neighbors of var_idx
                if not self.neighbor_learned_arr[var_idx]:
                    neighbors, co_parents_arr, y_sep_set_dict = self.find_neighborhood(var_idx)
                    self.neighbor_learned_arr[var_idx] = True
                    x_y_sep_set_dict[var_idx] = y_sep_set_dict

                    # add edges between the variable and its neighbors
                    for neighbor_idx in neighbors:
                        self.learned_skeleton.add_edge(self.var_names[var_idx], self.var_names[neighbor_idx])
                else:
                    # if neighbors already learned, get them from the graph
                    neighbors = self.learned_skeleton.neighbors(var_idx)

                    # make sure to only include neighbours that are still left
                    neighbors = [neighbor for neighbor in neighbors if var_left_bool_arr[neighbor]]

                    # get the separating sets from the dictionary
                    y_sep_set_dict = x_y_sep_set_dict[var_idx]

                    # co-parents are markov boundary variables that are not neighbors
                    co_parents_bool_arr = np.copy(self.markov_boundary_matrix[var_idx])
                    co_parents_bool_arr[neighbors] = False
                    co_parents_arr = np.flatnonzero(co_parents_bool_arr)

                # check if variable is removable
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
                        # remove the removable variable from the set of variables left
                        var_left_bool_arr[var_idx] = False

                        # update the markov boundary matrix
                        self.update_markov_boundary_matrix(var_idx, neighbors)
                        removable_found = True
                        break
                    else:
                        self.skip_rem_check_vec[var_idx] = True
                else:
                    self.skip_rem_check_vec[var_idx] = True

            if not removable_found:
                # if no removable found, then pick the variable with the smallest markov boundary from var_left_bool_arr
                var_left_arr = np.flatnonzero(var_left_bool_arr)
                mb_size_all = np.sum(self.markov_boundary_matrix[var_left_arr], axis=1)
                var_idx = var_left_arr[np.argmin(mb_size_all)]

                neighbors = self.learned_skeleton.neighbors(var_idx)

                # make sure to only include neighbours that are still left
                neighbors = [neighbor for neighbor in neighbors if var_left_bool_arr[neighbor]]
                var_left_bool_arr[var_idx] = False

                # update the markov boundary matrix
                self.update_markov_boundary_matrix(var_idx, neighbors)
                self.skip_rem_check_vec[:] = False
        return self.learned_skeleton

    def find_neighborhood(self, var_idx: int):
        """Find the neighborhood of a variable using Lemma 27.

        Args:
            var (int): The variable whose neighborhood we want to find.

        Returns:
            np.ndarray: 1D numpy array containing the variables in the neighborhood.
        """

        var_name = self.var_names[var_idx]
        var_mk_arr = self.markov_boundary_matrix[var_idx]
        var_mk_idxs = np.flatnonzero(var_mk_arr)

        neighbors_bool_arr = np.copy(var_mk_arr)
        co_parents_bool_arr = np.zeros(len(var_mk_arr), dtype=bool)
        y_sep_set_dict = dict()

        for mb_idx_y in range(len(var_mk_idxs)):
            var_y_idx = var_mk_idxs[mb_idx_y]
            # check if Y is already neighbor of X
            if not self.learned_skeleton.has_edge(var_idx, var_y_idx):
                x_y_sep_set = self.get_sep_set(var_name, var_y_idx, var_mk_idxs)
                if x_y_sep_set is not None:
                    # var_y is a co-parent of var_idx and thus NOT a neighbor
                    neighbors_bool_arr[var_y_idx] = False
                    co_parents_bool_arr[var_y_idx] = True
                    y_sep_set_dict[var_y_idx] = x_y_sep_set

        # remove all variables that are not neighbors
        neighbors_arr = np.flatnonzero(neighbors_bool_arr)
        co_parents_arr = np.flatnonzero(co_parents_bool_arr)
        return neighbors_arr, co_parents_arr, y_sep_set_dict

    def get_sep_set(self, var_name: str, var_y_idx: int, var_x_mk_idxs: np.ndarray) -> set[int] | None:
        var_mk_left_idxs = list(set(var_x_mk_idxs) - {var_y_idx})
        # use lemma 27 and check all proper subsets of Mb(X) - {Y}
        for cond_set_size in range(len(var_mk_left_idxs)):
            for var_s_idxs in itertools.combinations(var_mk_left_idxs, cond_set_size):
                cond_set = [self.var_names[idx] for idx in var_s_idxs]
                var_y_name = self.var_names[var_y_idx]
                if self.ci_test(var_name, var_y_name, cond_set, self.data):
                    # we know that var_y_idx is a co-parent and thus NOT a neighbor
                    return set(var_s_idxs)
        return None

    def cond_1(self, var_idx, neighbors, var_mk_idxs):
        num_neighbors = len(neighbors)
        var_name = self.var_names[var_idx]
        for var_y_idx in range(num_neighbors - 1):
            var_y = neighbors[var_y_idx]
            var_y_name = self.var_names[var_y]
            for var_z_idx in range(var_y_idx + 1, num_neighbors):
                var_z = neighbors[var_z_idx]
                var_z_name = self.var_names[var_z]
                xyz_tuple = (var_idx, var_y, var_z)
                if xyz_tuple in self.skip_check_cond1_set:
                    continue
                # if skip check is false, loop over all subsets S of Mb(X) - {Y, Z} and check if Y ind. Z | S + {X}
                var_mk_left_idxs = list(set(var_mk_idxs) - {var_y, var_z})
                for cond_set_size in range(len(var_mk_left_idxs) + 1):
                    for var_s_idxs in itertools.combinations(var_mk_left_idxs, cond_set_size):
                        cond_set = [self.var_names[idx] for idx in var_s_idxs] + [var_name]
                        if self.ci_test(var_y_name, var_z_name, cond_set, self.data):
                            return False
                self.skip_check_cond1_set.add(xyz_tuple)
        return True

    def cond_2(self, var_idx, neighbors, co_parents_arr, var_mk_idxs, x_v_structure_dict):
        var_name = self.var_names[var_idx]
        for var_y in co_parents_arr:
            for var_z in neighbors:
                xyz_tuple = (var_idx, var_y, var_z)
                if xyz_tuple in self.skip_check_cond2_set:
                    continue
                # if skip check is false, loop over all v such that x->v<-y is a v-structure
                for var_v in x_v_structure_dict.get(var_y, set()):
                    if var_v == var_z:
                        continue
                    # loop over all subsets s of Mb(X) - {V, Y, Z} and check if Y ind. Z | S + {X, V}
                    var_mk_left_idxs = list(set(var_mk_idxs) - {var_y, var_z, var_v})
                    for cond_set_size in range(len(var_mk_left_idxs) + 1):
                        for var_s_idxs in itertools.combinations(var_mk_left_idxs, cond_set_size):
                            cond_set = [self.var_names[idx] for idx in var_s_idxs] + [var_name, self.var_names[var_v]]
                            if self.ci_test(self.var_names[var_y], self.var_names[var_z], cond_set, self.data):
                                return False
                self.skip_check_cond2_set.add(xyz_tuple)
        return True

    def update_markov_boundary_matrix(self, var_idx: int, var_neighbors: np.ndarray):
        """
        Update the Markov boundary matrix after removing a variable.
        :param var_idx: Index of the variable to remove
        :param var_neighbors: 1D numpy array containing the indices of the neighbors of var_idx
        """

        var_markov_boundary = np.flatnonzero(self.markov_boundary_matrix[var_idx])

        # for every variable in the markov boundary of var_idx, remove it from the markov boundary and update flag
        for mb_var_idx in np.flatnonzero(self.markov_boundary_matrix[var_idx]):  # TODO use indexing instead
            self.markov_boundary_matrix[mb_var_idx, var_idx] = 0
            self.markov_boundary_matrix[var_idx, mb_var_idx] = 0
            self.skip_rem_check_vec[mb_var_idx] = False

        # TODO remove for RSL W
        # if len(var_markov_boundary) > len(var_neighbors):
        #     # Sufficient condition for diamond-free graphs
        #     return

        # find nodes whose co-parent status changes
        # we only remove Y from mkvb of Z iff X is their ONLY common child and they are NOT neighbors)
        for ne_idx_y in range(len(var_neighbors) - 1):  # -1 because no need to check last variable and also symmetry
            for ne_idx_z in range(ne_idx_y + 1, len(var_neighbors)):
                var_y_idx = var_neighbors[ne_idx_y]
                var_z_idx = var_neighbors[ne_idx_z]
                var_y_name = self.var_names[var_y_idx]
                var_z_name = self.var_names[var_z_idx]

                # determine whether the mkbv of var_y_idx or var_z_idx is smaller, and use the smaller one as cond_set
                var_y_markov_boundary = np.flatnonzero(self.markov_boundary_matrix[var_y_idx])
                var_z_markov_boundary = np.flatnonzero(self.markov_boundary_matrix[var_z_idx])
                if np.sum(self.markov_boundary_matrix[var_y_idx]) < np.sum(self.markov_boundary_matrix[var_z_idx]):
                    cond_set = [self.var_names[idx] for idx in set(var_y_markov_boundary) - {var_z_idx}]
                else:
                    cond_set = [self.var_names[idx] for idx in set(var_z_markov_boundary) - {var_y_idx}]

                if self.ci_test(var_y_name, var_z_name, cond_set, self.data):
                    # we know that Y and Z are co-parents and thus NOT neighbors
                    self.markov_boundary_matrix[var_y_idx, var_z_idx] = 0
                    self.markov_boundary_matrix[var_z_idx, var_y_idx] = 0
                    self.skip_rem_check_vec[var_y_idx] = False
                    self.skip_rem_check_vec[var_z_idx] = False

    def learn_v_structure(self, var_idx, neighbors, co_parents_arr, var_mk_idxs, y_sep_set_dict):
        """
        Learns the v-structures of a given variable.

        Args:
            var_idx (int): The index of the variable for which to learn the v-structures.
            neighbors (list): A list of indices representing the neighbors of the variable.
            co_parents_arr (list): A list of indices representing the co-parents of the variable.
            var_mk_idxs (list): A list of indices representing the variables in the Markov boundary of the variable.
            y_sep_set_dict (dict): A dictionary mapping indices of other variables to the separating sets
                that distinguish them from the current variable.

        """

        def is_y_z_neighbor(var_y, var_z):
            if self.learned_skeleton.has_edge(var_y, var_z):
                return True
            # check that all subsets S in Mb(X) + {X} - {Y, Z} satisfy Y NOT ind. Z | S
            var_mk_left_idxs = list(set(var_mk_idxs) - {var_y, var_z}) + [var_idx]
            for cond_set_size in range(len(var_mk_left_idxs) + 1):
                for var_s_idxs in itertools.combinations(var_mk_left_idxs, cond_set_size):
                    cond_set = [self.var_names[idx] for idx in var_s_idxs]
                    if self.ci_test(self.var_names[var_y], self.var_names[var_z], cond_set, self.data):
                        return False

            # add edge in skeleton
            self.learned_skeleton.add_edge(self.var_names[var_y], self.var_names[var_z])
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
