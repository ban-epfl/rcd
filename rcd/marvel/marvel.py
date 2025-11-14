"""MARVEL algorithm for causal discovery with latent variables.

This module implements MARVEL as described in our JMLR paper “Recursive Causal
Discovery.” All theorem/lemma references in comments correspond to that paper’s
numbering. The public API mirrors the other RCD subpackages: callers provide a
conditional independence (CI) oracle and optional Markov-boundary estimator.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import TYPE_CHECKING, Dict, List, Set

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
    """Learn a causal skeleton using MARVEL.

    Parameters
    ----------
    ci_test : Callable[[int, int, list[int], np.ndarray], bool]
        Conditional independence oracle following the project-wide signature.
    data : ndarray or pandas.DataFrame
        Observational dataset with shape ``(n_samples, n_vars)``.
    find_markov_boundary_matrix_fun : Callable[[np.ndarray], np.ndarray], optional
        Optional custom Markov-boundary estimator. Defaults to a Gaussian-based
        estimator when ``None``.

    Returns
    -------
    nx.Graph
        Learned undirected skeleton.
    """

    data_matrix = sanitize_data(data)
    marvel = _Marvel(ci_test, find_markov_boundary_matrix_fun)
    learned_skeleton = marvel.learn_and_get_skeleton(data_matrix)
    return learned_skeleton


class _Marvel:
    """Implementation of MARVEL (latent-variable setting)."""

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

        # we use a flag array to keep track of which variables' neighbors need to be learned (i.e., we learn if False)
        self.neighbor_learned_arr: np.ndarray | None = None

        self.skip_check_cond1_set: Set[tuple[int, int, int]] | None = None
        self.skip_check_cond2_set: Set[tuple[int, int, int]] | None = None

        # we use a dictionary that maps x to a dictionary that maps y to a set of variables v, such that x->v<-y is a v-structure
        self.v_structure_dict: Dict[int, Dict[int, Set[int]]] | None = None

        # we use a flag array to keep track of which variables' v-structures need to be learned (i.e., we learn if True)
        self.v_structure_learned_arr: np.ndarray | None = None

        self.var_idx_set: set[int] | None = None
        self.markov_boundary_matrix: np.ndarray | None = None
        self.learned_skeleton: nx.Graph | None = None

    def learn_and_get_skeleton(self, data: np.ndarray) -> nx.Graph:
        """Execute MARVEL on ``data`` and return the learned skeleton."""

        self.num_vars = data.shape[1]
        self.data = data

        self.skip_rem_check_vec = np.zeros(self.num_vars, dtype=bool)
        self.skip_check_cond1_set = set()
        self.skip_check_cond2_set = set()
        self.neighbor_learned_arr = np.zeros(self.num_vars, dtype=bool)
        self.v_structure_dict = {}
        self.v_structure_learned_arr = np.zeros(self.num_vars, dtype=bool)
        self.var_idx_set = set(range(self.num_vars))
        self.markov_boundary_matrix = self.find_markov_boundary_matrix(self.data)
        self.learned_skeleton = nx.Graph()
        self.learned_skeleton.add_nodes_from(range(self.num_vars))

        def data_included_ci_test(x: int, y: int, z: list[int]) -> bool:
            return self.ci_test(x, y, z, self.data)

        var_idx_arr = np.arange(self.num_vars)
        var_left_bool_arr = np.ones(self.num_vars, dtype=bool)  # Indicates if variable is left

        x_y_sep_set_dict: Dict[int, Dict[int, Set[int]]] = {}
        for _ in range(self.num_vars - 1):
            # Sort variables by decreasing Markov boundary size
            # Only sort variables that are still left and whose removability has NOT been checked
            var_to_sort_bool_arr = var_left_bool_arr & ~self.skip_rem_check_vec
            var_idx_to_sort_arr = var_idx_arr[var_to_sort_bool_arr]
            sorted_var_idx = sort_vars_by_mkb_size(
                self.markov_boundary_matrix[var_to_sort_bool_arr],
                var_idx_to_sort_arr,
            )

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
                    neighbors = list(self.learned_skeleton.neighbors(var_idx))

                    # Ensure only to include neighbors that are still left
                    neighbors = [neighbor for neighbor in neighbors if var_left_bool_arr[neighbor]]

                    # get the separating sets from the dictionary
                    y_sep_set_dict = x_y_sep_set_dict[var_idx]

                    # Co-parents are markov boundary variables that are not neighbors
                    co_parents_bool_arr = np.copy(self.markov_boundary_matrix[var_idx])
                    co_parents_bool_arr[neighbors] = False
                    co_parents_arr = np.flatnonzero(co_parents_bool_arr)

                neighbors_arr = np.asarray(neighbors, dtype=int)

                # Check if variable is removable
                if self.cond_1(var_idx, neighbors_arr, var_mk_idxs):
                    if not self.v_structure_learned_arr[var_idx]:
                        self.learn_v_structure(var_idx, neighbors_arr.tolist(), co_parents_arr, var_mk_idxs, y_sep_set_dict)
                        x_v_structure_dict = self.v_structure_dict.get(var_idx, {})
                    else:
                        # only keep y and z that are left that form a v-structure: x->z<-y
                        var_left_set = set(sorted_var_idx)
                        x_v_structure_dict = self.v_structure_dict.get(var_idx, {})
                        for var_y in list(x_v_structure_dict.keys()):
                            if not var_left_bool_arr[var_y]:
                                del x_v_structure_dict[var_y]
                            else:
                                x_v_structure_dict[var_y] = x_v_structure_dict[var_y].intersection(var_left_set)

                    if self.cond_2(var_idx, neighbors_arr, co_parents_arr, var_mk_idxs, x_v_structure_dict):
                        # Remove the removable variable from the set of variables left
                        var_left_bool_arr[var_idx] = False

                        # Update the Markov boundary matrix
                        update_markov_boundary_matrix(
                            self.markov_boundary_matrix,
                            data_included_ci_test,
                            var_idx,
                            neighbors_arr,
                            skip_check=self.skip_rem_check_vec,
                        )
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

                neighbors = list(self.learned_skeleton.neighbors(var_idx))

                # Ensure only to include neighbors that are still left
                neighbors = [neighbor for neighbor in neighbors if var_left_bool_arr[neighbor]]
                var_left_bool_arr[var_idx] = False

                update_markov_boundary_matrix(
                    self.markov_boundary_matrix,
                    data_included_ci_test,
                    var_idx,
                    neighbors,
                    skip_check=self.skip_rem_check_vec,
                )
                self.skip_rem_check_vec[:] = False
        return self.learned_skeleton

    def find_neighborhood(
        self,
        var_idx: int,
    ) -> tuple[np.ndarray, np.ndarray, Dict[int, Set[int]]]:
        """Return neighbors, co-parents, and separating sets for ``var_idx``.

        Parameters
        ----------
        var_idx : int
            Index of the variable whose neighborhood is being queried.

        Returns
        -------
        tuple
            ``(neighbors, co_parents, sep_sets)`` where
            ``neighbors`` is a 1D ``np.ndarray`` of neighbor indices,
            ``co_parents`` is a 1D ``np.ndarray`` of co-parent indices, and
            ``sep_sets`` is a mapping ``var_y -> separating set`` used when
            distinguishing co-parents (Lemma 27).
        """

        if self.markov_boundary_matrix is None or self.learned_skeleton is None or self.data is None:
            raise RuntimeError("Learning state has not been initialized.")

        var_mk_arr = np.flatnonzero(self.markov_boundary_matrix[var_idx])
        var_mk_set = set(var_mk_arr)

        neighbors_bool_arr = np.copy(self.markov_boundary_matrix[var_idx])
        co_parents_bool_arr = np.zeros(len(neighbors_bool_arr), dtype=bool)
        y_sep_set_dict: Dict[int, Set[int]] = {}

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

        neighbors_arr = np.flatnonzero(neighbors_bool_arr)
        co_parents_arr = np.flatnonzero(co_parents_bool_arr)
        return neighbors_arr, co_parents_arr, y_sep_set_dict

    def get_sep_set(
        self,
        var_idx: int,
        var_y_idx: int,
        var_x_mk_idxs: np.ndarray,
    ) -> Set[int] | None:
        """Return a separating set for ``var_idx`` and ``var_y_idx`` using Lemma 27."""

        if self.data is None:
            raise RuntimeError("Learning state has not been initialized.")

        var_mk_left_idxs = list(set(var_x_mk_idxs) - {var_y_idx})
        # Use Lemma 27 and check all proper subsets of Mb(X) - {Y}
        for cond_set_size in range(len(var_mk_left_idxs) + 1):
            for var_s_idxs in itertools.combinations(var_mk_left_idxs, cond_set_size):
                if self.ci_test(var_idx, var_y_idx, list(var_s_idxs), self.data):
                    # Y is a co-parent and thus NOT a neighbor
                    return set(var_s_idxs)
        return None

    def cond_1(self, var_idx: int, neighbors: np.ndarray, var_mk_idxs: np.ndarray) -> bool:
        """Check the first MARVEL removability condition."""

        if self.skip_check_cond1_set is None or self.data is None:
            raise RuntimeError("Learning state has not been initialized.")

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

    def cond_2(
        self,
        var_idx: int,
        neighbors: np.ndarray,
        co_parents_arr: np.ndarray,
        var_mk_idxs: np.ndarray,
        x_v_structure_dict: Dict[int, Set[int]],
    ) -> bool:
        """Check the second MARVEL removability condition."""

        if self.skip_check_cond2_set is None or self.data is None:
            raise RuntimeError("Learning state has not been initialized.")

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

    def learn_v_structure(
        self,
        var_idx: int,
        neighbors: List[int],
        co_parents_arr: np.ndarray,
        var_mk_idxs: np.ndarray,
        y_sep_set_dict: Dict[int, Set[int]],
    ) -> None:
        """Populate the v-structure registry for ``var_idx``.

        Parameters
        ----------
        var_idx : int
            Root variable ``X`` in the notation of the paper.
        neighbors : list[int]
            Indices of the current neighbors of ``X``.
        co_parents_arr : np.ndarray
            Indices of co-parents of ``X`` discovered from the Markov boundary.
        var_mk_idxs : np.ndarray
            Markov-boundary indices for ``X``.
        y_sep_set_dict : dict[int, set[int]]
            Maps each co-parent ``Y`` to the separating set ``S`` that proves
            ``Y`` is not a neighbor of ``X``.
        """

        if self.learned_skeleton is None or self.v_structure_dict is None or self.data is None:
            raise RuntimeError("Learning state has not been initialized.")

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
                    x_v_structure_dict = self.v_structure_dict.get(var_idx, {})
                    z_set = x_v_structure_dict.get(var_y, set())
                    z_set.add(var_z)
                    x_v_structure_dict[var_y] = z_set
                    self.v_structure_dict[var_idx] = x_v_structure_dict
        self.v_structure_learned_arr[var_idx] = True
