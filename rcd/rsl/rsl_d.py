import numpy as np

from rcd.rsl.rsl_base import RSLBase

"""
This file contains the implementation for the rsl-D algorithm learning diamond-free graphs from i.i.d. samples. 
The class is initialized with a conditional independence test function, which determines whether two variables are 
independent given another set of variables, using the data provided. For examples of possible conditional 
independence tests, see utilities/ci_tests.py. For details on how to write a custom CI test function, look at the 
constructor (init function) of the RSLBase class.

The class has a learn_and_get_skeleton function, inherited from the base class, that takes in a Pandas DataFrame of 
data, where the ith column contains samples from the ith variable, and returns a networkx graph representing the 
learned skeleton.
"""


class RSLDiamondFree(RSLBase):

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
        super().__init__(ci_test, find_markov_boundary_matrix_fun)
        self.is_rsl_d = True

    def find_neighborhood(self, var: int) -> np.ndarray:
        """Find the neighborhood of a variable using Proposition 40.

        Args:
            var (int): The variable whose neighborhood we want to find.

        Returns:
            np.ndarray: 1D numpy array containing the variables in the neighborhood.
        """

        var_name = self.var_names[var]
        var_mk_bool_arr = self.markov_boundary_matrix[var]
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
                var_y_name = self.var_names[var_y]
                cond_set = [self.var_names[idx] for idx in var_mk_set - {var_y, var_z}]

                if self.ci_test(var_name, var_y_name, cond_set, self.data):
                    # we know that var_y is a co-parent and thus NOT a neighbor
                    neighbor_bool_arr[var_y] = 0
                    break
            if not neighbor_bool_arr[var_y]:
                continue

        # return neighbors
        neighbor_arr = np.flatnonzero(neighbor_bool_arr)
        return neighbor_arr

    def is_removable(self, var: int) -> bool:
        """Check whether a variable is removable using Theorem 39.

        Args:
            var (int): The variable to check.

        Returns:
            bool: True if the variable is removable, False otherwise.
        """

        var_mk_bool_arr = self.markov_boundary_matrix[var]
        var_mk_arr = np.flatnonzero(var_mk_bool_arr)
        var_mk_set = set(var_mk_arr)

        # use Lemma 3 of rsl paper: var_y is Y and var_z is Z. cond_set is Mb(X) + {X} - {Y, Z}
        for mb_idx_y in range(len(var_mk_arr) - 1):  # -1 because no need to check last variable and also symmetry
            for mb_idx_z in range(mb_idx_y + 1, len(var_mk_arr)):
                var_y = var_mk_arr[mb_idx_y]
                var_z = var_mk_arr[mb_idx_z]
                var_y_name = self.var_names[var_y]
                var_z_name = self.var_names[var_z]
                cond_set = [self.var_names[idx] for idx in var_mk_set - {var_y, var_z}] + [self.var_names[var]]

                if self.ci_test(var_y_name, var_z_name, cond_set, self.data):
                    return False
        return True
