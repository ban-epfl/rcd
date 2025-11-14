"""Recursive Skeleton Learning for bounded-clique graphs (RSL-W).

Implements the bounded clique-number variant of RSL as defined in our JMLR
paper “Recursive Causal Discovery,” using the paper’s consolidated
notation/theorem numbering.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from rcd.rsl.rsl_base import _RSLBase
from rcd.utilities.utils import sanitize_data

if TYPE_CHECKING:
    import pandas as pd


def learn_and_get_skeleton(
    ci_test: Callable[[int, int, list[int], np.ndarray], bool],
    data: np.ndarray | "pd.DataFrame",
    clique_num: int,
    find_markov_boundary_matrix_fun: Callable[[np.ndarray], np.ndarray] | None = None,
) -> nx.Graph:
    """Learn a skeleton with bounded clique number using RSL-W.

    Parameters
    ----------
    ci_test : Callable[[int, int, list[int], np.ndarray], bool]
        Conditional independence oracle supplied by the caller.
    data : ndarray or pandas.DataFrame
        Observational dataset arranged as ``(n_samples, n_vars)``.
    clique_num : int
        Upper bound on the clique number of the underlying graph.
    find_markov_boundary_matrix_fun : Callable[[np.ndarray], np.ndarray], optional
        Optional custom Markov-boundary estimator.

    Returns
    -------
    nx.Graph
        Learned undirected skeleton.
    """

    data_matrix = sanitize_data(data)
    rsl_w = _RSLBoundedClique(ci_test, find_markov_boundary_matrix_fun)
    learned_skeleton = rsl_w.learn_and_get_skeleton(data_matrix, clique_num)
    return learned_skeleton


class _RSLBoundedClique(_RSLBase):
    """Specialization of :class:`_RSLBase` for bounded clique number."""

    def __init__(
        self,
        ci_test: Callable[[int, int, list[int], np.ndarray], bool],
        find_markov_boundary_matrix_fun: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """Configure the bounded-clique learner."""

        super().__init__(ci_test, find_markov_boundary_matrix_fun)
        self.is_rsl_d = False  # Ensure it's treated as RSL-W

    def find_neighborhood(self, var: int) -> np.ndarray:
        """Find the neighborhood of ``var`` via Proposition 37."""

        if self.markov_boundary_matrix is None or self.data is None or self.clique_num is None:
            raise RuntimeError("Learning state has not been initialized.")

        var_mk_bool_arr = self.markov_boundary_matrix[var]
        var_mk_arr = np.flatnonzero(var_mk_bool_arr)
        var_mk_set = set(var_mk_arr)

        # loop through markov boundary matrix row corresponding to var_name
        # use Proposition 37: var_y is Y and var_z_idx is Z. cond_set is Mb(X) - {Y} - S,
        # where S is a subset of Mb(X) - {Y} of size m-1

        # First, assume all variables are neighbors
        neighbor_bool_arr = np.copy(var_mk_bool_arr)

        # var_s contains the indices of the variables in the subset S
        for var_y in var_mk_arr:
            for var_s in itertools.combinations(var_mk_set - {var_y}, self.clique_num - 1):
                cond_set = list(var_mk_set - {var_y} - set(var_s))
                if self.ci_test(var, var_y, cond_set, self.data):
                    # we know that var_y is a co-parent and thus NOT a neighbor
                    neighbor_bool_arr[var_y] = False
                    break

        neighbor_arr = np.flatnonzero(neighbor_bool_arr)
        return neighbor_arr

    def is_removable(self, var: int) -> bool:
        """Check whether ``var`` is removable via Theorem 36."""

        if (
            self.markov_boundary_matrix is None
            or self.data is None
            or self.clique_num is None
            or self.num_vars is None
        ):
            raise RuntimeError("Learning state has not been initialized.")

        var_mk_bool_arr = self.markov_boundary_matrix[var]
        var_mk_arr = np.flatnonzero(var_mk_bool_arr)
        var_mk_set = set(var_mk_arr)

        # use Theorem 36: var_y is Y and var_z is Z. cond_set is Mb(X) + {X} - ({Y, Z} + S)
        # get all subsets with size from 0 to self.clique_num - 2.
        # for each subset, check if there exists a pair of variables that are d-separated given the subset
        max_subset_size = max(0, self.clique_num - 2)
        for subset_size in range(max_subset_size + 1):
            # var_s contains the variables in the subset S
            for var_s in itertools.combinations(var_mk_arr, subset_size):
                var_mk_left = list(var_mk_set - set(var_s))
                var_mk_left_set = set(var_mk_left)

                for mb_idx_left_y in range(len(var_mk_left)):
                    var_y = var_mk_left[mb_idx_left_y]

                    # check second condition
                    cond_set = list(var_mk_left_set - {var_y})

                    if self.ci_test(var, var_y, cond_set, self.data):
                        return False

                    # check first condition
                    for mb_idx_left_z in range(mb_idx_left_y + 1, len(var_mk_left)):
                        var_z = var_mk_left[mb_idx_left_z]
                        cond_set = list(var_mk_left_set - {var_y, var_z}) + [var]

                        if self.ci_test(var_y, var_z, cond_set, self.data):
                            return False
        return True
