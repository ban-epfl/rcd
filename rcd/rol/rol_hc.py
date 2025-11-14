"""Hill-climbing refinement of removal orders (ROL-HC)."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Set
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from rcd.rsl.rsl_d import _RSLDiamondFree
from rcd.utilities.utils import compute_mb_gaussian, sanitize_data, update_markov_boundary_matrix

if TYPE_CHECKING:
    import pandas as pd


def learn_and_get_skeleton(
    ci_test: Callable[[int, int, list[int], np.ndarray], bool],
    data: np.ndarray | "pd.DataFrame",
    max_iters: int,
    max_swaps: int,
    initial_r_order: np.ndarray | None = None,
    find_markov_boundary_matrix_fun: Callable[[np.ndarray], np.ndarray] | None = None,
) -> nx.Graph:
    """Public helper for the ROL-HC algorithm.

    Parameters
    ----------
    ci_test : Callable[[int, int, list[int], np.ndarray], bool]
        Conditional independence oracle.
    data : ndarray or pandas.DataFrame
        Observational dataset shaped ``(n_samples, n_vars)``.
    max_iters : int
        Maximum number of hill-climb iterations.
    max_swaps : int
        Maximum swap distance considered per iteration.
    initial_r_order : np.ndarray, optional
        Starting removal order. When ``None``, an RSL-D run provides the seed.
    find_markov_boundary_matrix_fun : Callable[[np.ndarray], np.ndarray], optional
        Custom Markov-boundary estimator.

    Returns
    -------
    nx.Graph
        Learned skeleton after hill climbing.
    """

    data_matrix = sanitize_data(data)
    rol_hc = _ROLHillClimb(
        ci_test,
        max_iters,
        max_swaps,
        find_markov_boundary_matrix_fun=find_markov_boundary_matrix_fun,
    )
    learned_skeleton = rol_hc.learn_and_get_skeleton(data_matrix, initial_r_order)
    return learned_skeleton


class _ROLHillClimb:
    """Hill-climbing refinement of removal orders (JMLR Theorem references apply)."""

    def __init__(
        self,
        ci_test: Callable[[int, int, list[int], np.ndarray], bool],
        max_iters: int,
        max_swaps: int,
        find_markov_boundary_matrix_fun: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        if find_markov_boundary_matrix_fun is None:
            self.find_markov_boundary_matrix = compute_mb_gaussian
        else:
            self.find_markov_boundary_matrix = find_markov_boundary_matrix_fun

        self.num_vars: int | None = None
        self.data: np.ndarray | None = None
        self.ci_test = ci_test
        self.max_iters = max_iters
        self.max_swaps = max_swaps

        # we use a flag array to keep track of which variables need to be checked for removal (i.e., we check if true)
        self.var_idx_set: np.ndarray | None = None
        self.learned_skeleton: nx.Graph | None = None

    def learn_and_get_skeleton(
        self,
        data: np.ndarray,
        initial_r_order: np.ndarray | None = None,
    ) -> nx.Graph:
        """Learn the skeleton via hill climbing on removal orders.

        Parameters
        ----------
        data : np.ndarray
            Data matrix with shape ``(n_samples, n_vars)``.
        initial_r_order : np.ndarray, optional
            R-order seed. When omitted, an RSL-D run supplies it.

        Returns
        -------
        nx.Graph
            Learned skeleton.
        """

        self.num_vars = data.shape[1]
        self.data = data

        self.learned_skeleton = None

        if initial_r_order is None:
            # Set r-order by running RSL-D
            rol_init = _RSLDiamondFree(self.ci_test, self.find_markov_boundary_matrix)
            initial_r_order = rol_init.compute_removal_order(self.data)
        else:
            initial_r_order = np.asarray(initial_r_order, dtype=int)
            if initial_r_order.shape != (self.num_vars,):
                raise ValueError("initial_r_order must have length equal to the number of variables")

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
        """Build a skeleton based on a fixed removal order."""

        if self.num_vars is None or self.data is None:
            raise RuntimeError("Learning state has not been initialized.")

        learned_skeleton = nx.Graph()
        learned_skeleton.add_nodes_from(range(self.num_vars))

        markov_boundary = self.find_markov_boundary_matrix(self.data)

        for var in r_order:
            # Learn the neighbors of the variable and then remove it from the graph
            neighbors = self.find_neighborhood(var, markov_boundary[var])
            for neighbor in neighbors:
                learned_skeleton.add_edge(var, neighbor)

        return learned_skeleton

    def compute_cost(
        self,
        r_order: np.ndarray,
        starting_index: int,
        ending_index: int,
    ) -> np.ndarray:
        """Compute the cost contribution of a window of the removal order."""

        if self.data is None or self.num_vars is None:
            raise RuntimeError("Learning state has not been initialized.")

        remaining_vars_mkb = sorted(r_order[starting_index:])
        cost_vec = np.zeros(len(r_order))

        # Restrict data to the remaining variables
        remaining_data = self.data[:, remaining_vars_mkb]

        sub_markov_boundary = self.find_markov_boundary_matrix(remaining_data)
        markov_boundary = np.zeros((self.num_vars, self.num_vars), dtype=bool)
        markov_boundary[np.ix_(remaining_vars_mkb, remaining_vars_mkb)] = sub_markov_boundary

        data_included_ci_test = lambda x, y, z: self.ci_test(x, y, z, self.data)  # noqa: E731

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

    def find_neighborhood(self, var: int, var_mk_bool_arr: np.ndarray) -> np.ndarray:
        """Find the neighborhood of ``var`` via Proposition 40."""

        if self.data is None:
            raise RuntimeError("Learning state has not been initialized.")

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
        """Find the neighborhood of ``var`` using Lemma 27."""

        if self.data is None:
            raise RuntimeError("Learning state has not been initialized.")

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
        """Check if ``var_y`` is a neighbor of ``var`` using Lemma 27."""

        if self.data is None:
            raise RuntimeError("Learning state has not been initialized.")

        var_mk_left_list = list(var_mk_set - {var_y})
        # Use Lemma 27 and check all proper subsets of Mb(X) - {Y}
        for cond_set_size in range(len(var_mk_left_list)):
            for var_s in itertools.combinations(var_mk_left_list, cond_set_size):
                cond_set = list(var_s)
                if self.ci_test(var, var_y, cond_set, self.data):
                    # Y is a co-parent and thus NOT a neighbor
                    return False
        return True

