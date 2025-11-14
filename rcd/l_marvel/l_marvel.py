"""Latent-variable MARVEL (L-MARVEL) implementation.

This module implements the version of L-MARVEL described in our JMLR paper
“Recursive Causal Discovery” (Mokhtarian *et al.*, 2025). It reuses the
notation and theorem numbering from that paper: all references to
Lemmas/Theorems/Propositions refer to their JMLR counterparts.

The class below exposes a ``learn_and_get_skeleton`` API that accepts either a
NumPy array or a pandas ``DataFrame`` and returns a NetworkX skeleton.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import TYPE_CHECKING, Set

import networkx as nx
import numpy as np

from rcd.utilities.utils import (
    REMOVABLE_NOT_FOUND,
    compute_mb_gaussian,
    sanitize_data,
    sort_vars_by_mkb_size,
    update_markov_boundary_matrix,
)

if TYPE_CHECKING:
    import pandas as pd


def learn_and_get_skeleton(
    ci_test: Callable[[int, int, list[int], np.ndarray], bool],
    data: np.ndarray | "pd.DataFrame",
    find_markov_boundary_matrix_fun: Callable[[np.ndarray], np.ndarray] | None = None,
) -> nx.Graph:
    """Learn a latent-variable skeleton using L-MARVEL.

    Parameters
    ----------
    ci_test : Callable[[int, int, list[int], np.ndarray], bool]
        Conditional independence oracle following the project-wide signature.
    data : ndarray or pandas.DataFrame
        Observational data shaped ``(n_samples, n_vars)``.
    find_markov_boundary_matrix_fun : Callable[[np.ndarray], np.ndarray], optional
        Optional custom Markov-boundary estimator. Defaults to the Gaussian estimator.

    Returns
    -------
    nx.Graph
        Learned undirected skeleton.
    """

    data_mat = sanitize_data(data)
    l_marvel = _LMarvel(ci_test, find_markov_boundary_matrix_fun)
    learned_skeleton = l_marvel.learn_and_get_skeleton(data_mat)
    return learned_skeleton


class _LMarvel:
    """Implementation of the L-MARVEL algorithm (JMLR notation applies)."""

    def __init__(
        self,
        ci_test: Callable[[int, int, list[int], np.ndarray], bool],
        find_markov_boundary_matrix_fun: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        if find_markov_boundary_matrix_fun is None:
            self.find_markov_boundary_matrix = compute_mb_gaussian
        else:
            self.find_markov_boundary_matrix = find_markov_boundary_matrix_fun

        self.num_vars: int | None = None
        self.data: np.ndarray | None = None
        self.ci_test = ci_test

        # we use a flag array to keep track of which variables need to be checked for removal (i.e., we check if False)
        self.skip_rem_check_vec: np.ndarray | None = None  # SkipCheck_VEC in the paper

        # we use a set to keep track of which Y and Z pairs have been checked for a given X (see IsRemovable in the paper)
        self.skip_rem_check_set: Set[tuple[int, int, int]] | None = None  # SkipCheck_MAT in the paper

        # we use a flag array to keep track of which variables' neighbors need to be learned (i.e., we learn if False)
        self.neighbor_learned_arr: np.ndarray | None = None
        self.var_idx_set: set[int] | None = None
        self.markov_boundary_matrix: np.ndarray | None = None
        self.learned_skeleton: nx.Graph | None = None

    def learn_and_get_skeleton(self, data: np.ndarray) -> nx.Graph:
        """Execute L-MARVEL on ``data`` and return the learned skeleton."""

        self.num_vars = data.shape[1]
        self.data = data

        self.skip_rem_check_vec = np.zeros(self.num_vars, dtype=bool)
        self.skip_rem_check_set = set()
        self.neighbor_learned_arr = np.zeros(self.num_vars, dtype=bool)
        self.var_idx_set = set(range(self.num_vars))
        self.markov_boundary_matrix = self.find_markov_boundary_matrix(self.data)
        self.learned_skeleton = nx.Graph()
        self.learned_skeleton.add_nodes_from(range(self.num_vars))

        def data_included_ci_test(x: int, y: int, z: list[int]) -> bool:
            return self.ci_test(x, y, z, self.data)

        var_arr = np.arange(self.num_vars)
        var_left_bool_arr = np.ones(self.num_vars, dtype=bool)  # Indicates if variable is left

        for _ in range(self.num_vars - 1):
            # sort variables by decreasing Markov boundary size
            # only sort variables that are still left and whose removability has NOT been checked
            var_to_sort_bool_arr = var_left_bool_arr & ~self.skip_rem_check_vec
            var_to_sort_arr = var_arr[var_to_sort_bool_arr]
            sorted_var_arr = sort_vars_by_mkb_size(
                self.markov_boundary_matrix[var_to_sort_bool_arr],
                var_to_sort_arr,
            )

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
                if self.is_removable(var, np.asarray(neighbors, dtype=int)):
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
            neighbors = [
                neighbor
                for neighbor in self.learned_skeleton.neighbors(removable_var)
                if var_left_bool_arr[neighbor]
            ]

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
        """Find the neighborhood of ``var`` using Lemma 27."""

        if self.markov_boundary_matrix is None or self.learned_skeleton is None or self.data is None:
            raise RuntimeError("Learning state has not been initialized.")

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
        """Check if ``var_y`` is a neighbor of ``var`` using Lemma 27."""

        if self.data is None:
            raise RuntimeError("Learning state has not been initialized.")

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
        """Check whether ``var`` is removable using Theorem 32."""

        if self.markov_boundary_matrix is None or self.skip_rem_check_set is None or self.data is None:
            raise RuntimeError("Learning state has not been initialized.")

        var_mk_bool_arr = self.markov_boundary_matrix[var]
        var_mk_arr = np.flatnonzero(var_mk_bool_arr)
        var_mk_set = set(var_mk_arr)

        def cond_1(var_y: int, var_z: int) -> bool:
            # there exists subset W in Mb(X) - {Y, Z}, s.t. Y ind. Z | W
            var_mk_left_list = list(var_mk_set - {var_y, var_z})
            for cond_set_size in range(len(var_mk_left_list) + 1):
                for var_s in itertools.combinations(var_mk_left_list, cond_set_size):
                    cond_set = list(var_s)
                    if self.ci_test(var_y, var_z, cond_set, self.data):
                        return True
            return False

        def cond_2(var_y: int, var_z: int) -> bool:
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
