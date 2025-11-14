"""Recursive Skeleton Learning for diamond-free graphs (RSL-D).

Implements the diamond-free branch of the Recursive Skeleton Learning
algorithms introduced in our JMLR paper “Recursive Causal Discovery.” All
references to propositions or lemmas align with that paper’s numbering.
"""

from __future__ import annotations

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
    find_markov_boundary_matrix_fun: Callable[[np.ndarray], np.ndarray] | None = None,
) -> nx.Graph:
    """Learn a skeleton for a diamond-free graph using RSL-D.

    Parameters
    ----------
    ci_test : Callable[[int, int, list[int], np.ndarray], bool]
        Conditional independence test that accepts the indices of two variables,
        a conditioning set, and the full data matrix.
    data : ndarray or pandas.DataFrame
        Observational dataset shaped ``(n_samples, n_vars)``.
    find_markov_boundary_matrix_fun : Callable[[np.ndarray], np.ndarray], optional
        Optional custom routine for estimating the Markov boundary matrix. When
        omitted, the default Gaussian estimator is used.

    Returns
    -------
    nx.Graph
        Learned undirected skeleton.
    """

    data_matrix = sanitize_data(data)
    rsl_d = _RSLDiamondFree(ci_test, find_markov_boundary_matrix_fun)
    learned_skeleton = rsl_d.learn_and_get_skeleton(data_matrix)
    return learned_skeleton


class _RSLDiamondFree(_RSLBase):
    """Specialization of :class:`_RSLBase` for diamond-free graphs."""

    def __init__(
        self,
        ci_test: Callable[[int, int, list[int], np.ndarray], bool],
        find_markov_boundary_matrix_fun: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """Configure the diamond-free learner.

        Parameters
        ----------
        ci_test : Callable[[int, int, list[int], np.ndarray], bool]
            Conditional independence oracle supplied by the caller.
        find_markov_boundary_matrix_fun : Callable[[np.ndarray], np.ndarray], optional
            Alternative Markov-boundary estimator. Falls back to the Gaussian
            estimator when ``None``.
        """

        super().__init__(ci_test, find_markov_boundary_matrix_fun)
        self.is_rsl_d = True

    def find_neighborhood(self, var: int) -> np.ndarray:
        """Return the neighborhood of ``var`` via Proposition 40.

        Parameters
        ----------
        var : int
            Variable whose neighbors should be retrieved.

        Returns
        -------
        np.ndarray
            Indices of the neighbors of ``var``.
        """

        if self.markov_boundary_matrix is None or self.data is None:
            raise RuntimeError("Learning state has not been initialized.")

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
                cond_set = list(var_mk_set - {var_y, var_z})

                if self.ci_test(var, var_y, cond_set, self.data):
                    # we know that var_y is a co-parent and thus NOT a neighbor
                    neighbor_bool_arr[var_y] = False
                    break

        neighbor_arr = np.flatnonzero(neighbor_bool_arr)
        return neighbor_arr

    def is_removable(self, var: int) -> bool:
        """Check whether ``var`` is removable via Theorem 39.

        Parameters
        ----------
        var : int
            Variable to check for removability.

        Returns
        -------
        bool
            ``True`` when the variable can be safely removed.
        """

        if self.markov_boundary_matrix is None or self.data is None:
            raise RuntimeError("Learning state has not been initialized.")

        var_mk_bool_arr = self.markov_boundary_matrix[var]
        var_mk_arr = np.flatnonzero(var_mk_bool_arr)
        var_mk_set = set(var_mk_arr)

        # use Lemma 3 of rsl paper: var_y is Y and var_z is Z. cond_set is Mb(X) + {X} - {Y, Z}
        for mb_idx_y in range(len(var_mk_arr) - 1):  # -1 because no need to check last variable and also symmetry
            for mb_idx_z in range(mb_idx_y + 1, len(var_mk_arr)):
                var_y = var_mk_arr[mb_idx_y]
                var_z = var_mk_arr[mb_idx_z]
                cond_set = list(var_mk_set - {var_y, var_z}) + [var]

                if self.ci_test(var_y, var_z, cond_set, self.data):
                    return False
        return True
