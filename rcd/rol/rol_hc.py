from rcd.rol.rol_init import ROLInitializer
from rcd.utilities.utils import *


class ROLHillClimb:
    def __init__(self, ci_test, max_iters: int, max_swaps: int):
        """
        Initialize the ROL hill climbing algorithm with the data and conditional independence test to use.
        :param ci_test: Conditional independence test to use that takes in the names of two variables and a list of
        variable names as the conditioning set, and returns True if the two variables are independent given the
        conditioning set, and False otherwise. The signature of the function should be:
        ci_test(var_name1: str, var_name2: str, cond_set: List[str], data: pd.DataFrame) -> bool
        :param max_iters: Maximum number of iterations to run the algorithm for
        :param max_swaps: Maximum swap distance to consider
        """
        self.num_vars = None
        self.data = None
        self.var_names = None
        self.ci_test = ci_test
        self.max_iters = max_iters
        self.max_swaps = max_swaps

        # we use a flag array to keep track of which variables need to be checked for removal (i.e., we check if true)
        self.var_idx_set = None
        self.markov_boundary_matrix = None
        self.learned_skeleton = None

    def reset_fields(self, data: pd.DataFrame):
        self.num_vars = len(data.columns)
        self.data = data
        self.var_names = data.columns

        self.markov_boundary_matrix = None
        self.learned_skeleton = None

    def has_alg_run(self):
        return self.learned_skeleton is not None

    def learn_and_get_skeleton(self, data: pd.DataFrame, initial_r_order=None) -> nx.Graph:
        self.reset_fields(data)

        self.markov_boundary_matrix = find_markov_boundary_matrix(self.data, self.ci_test)

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

        self.learned_skeleton = self.learn_skeleton_using_r_order(curr_r_order)
        return self.learned_skeleton

    def learn_skeleton_using_r_order(self, r_order: np.ndarray):
        # initialize graph
        learned_skeleton = nx.Graph()
        learned_skeleton.add_nodes_from(self.var_names)

        markov_boundary = find_markov_boundary_matrix(self.data, self.ci_test)

        for var in r_order:
            # learn the neighbors of the variable and then remove it from the graph
            neighbors = self.find_neighbors(var, markov_boundary[var])
            for neighbor in neighbors:
                learned_skeleton.add_edge(self.var_names[var], self.var_names[neighbor])

        return learned_skeleton

    def compute_cost(self, r_order, starting_index, ending_index):
        """
        Compute the cost of the given r-order between the given starting and ending indices
        :param r_order: The r-order to compute the cost of
        :param starting_index: The starting index of the r-order to compute the cost of
        :param ending_index: The ending index (exclusive) of the r-order to compute the cost of
        :return: The cost of the given r-order between the given starting and ending indices
        """

        remaining_vars_mkb = sorted(r_order[starting_index:])
        cost_vec = np.zeros(len(r_order))

        # restrict data to the remaining variables
        remaining_data = self.data.iloc[:, remaining_vars_mkb]  # TODO move outside of compute cost

        sub_markov_boundary = find_markov_boundary_matrix(remaining_data, self.ci_test)
        markov_boundary = np.zeros((self.num_vars, self.num_vars), dtype=bool)
        markov_boundary[np.ix_(remaining_vars_mkb, remaining_vars_mkb)] = sub_markov_boundary

        for index in range(starting_index, ending_index):
            var_to_remove = r_order[index]

            neighbors = self.find_neighbors(var_to_remove, markov_boundary[var_to_remove])

            cost_vec[index] = len(neighbors)

            # update markov boundary matrix
            markov_boundary = self.update_markov_boundary_matrix(markov_boundary, var_to_remove, neighbors)
        return cost_vec

    def find_neighbors(self, var_idx: int, var_markov_boundary: np.ndarray) -> np.ndarray:
        """
        Find the neighborhood of a variable using Lemma 27.
        :param var_idx: Index of the variable in the data
        :param var_markov_boundary: Markov boundary of the variable
        :return: 1D numpy array containing the indices of the variables in the neighborhood
        """

        var_name = self.var_names[var_idx]

        neighbors = np.copy(var_markov_boundary)

        for var_y_idx in range(len(var_markov_boundary)):
            # check if Y is already neighbor of X
            var_mkb_idxs = np.flatnonzero(var_markov_boundary)
            if not self.is_neighbor(var_name, var_y_idx, var_mkb_idxs):
                neighbors[var_y_idx] = 0

        # remove all variables that are not neighbors
        neighbors_idx_arr = np.flatnonzero(neighbors)
        return neighbors_idx_arr

    def is_neighbor(self, var_name: str, var_y_idx: int, var_x_mk_idxs: np.ndarray) -> bool:
        for mb_idx_z in range(len(var_x_mk_idxs)):
            var_z_idx = var_x_mk_idxs[mb_idx_z]
            if var_y_idx == var_z_idx:
                continue
            var_y_name = self.var_names[var_y_idx]
            cond_set = [self.var_names[idx] for idx in set(var_x_mk_idxs) - {var_y_idx, var_z_idx}]

            if self.ci_test(var_name, var_y_name, cond_set, self.data):
                # we know that var2 is a co-parent and thus NOT a neighbor
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

