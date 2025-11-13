"""Base primitives for Recursive Skeleton Learning (RSL) algorithms.

The :class:`_RSLBase` class coordinates conditional independence (CI) tests, Markov
boundary updates, and graph bookkeeping for concrete algorithms such as RSL-D and
RSL-W. Subclasses supply problem-specific logic for determining neighborhoods and
deciding whether a variable is removable.
"""

from __future__ import annotations

from collections.abc import Callable

import networkx as nx
import numpy as np

from rcd.utilities.utils import (
    REMOVABLE_NOT_FOUND,
    compute_mb_gaussian,
    update_markov_boundary_matrix,
)


class _RSLBase:
    """Common functionality for Recursive Skeleton Learning algorithms.

    Parameters
    ----------
    ci_test : Callable[[int, int, list[int], np.ndarray], bool]
        Conditional independence test. The callable receives the indices of two
        variables, a conditioning set, and the data matrix, and returns ``True``
        when the variables are conditionally independent given the set.
    find_markov_boundary_matrix_fun : Callable[[np.ndarray], np.ndarray], optional
        Custom routine that estimates the Markov boundary matrix. If omitted, a
        Gaussian partial-correlation based estimator is used.

    Notes
    -----
    Concrete subclasses must implement :meth:`find_neighborhood` and
    :meth:`is_removable`.
    """

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

        # we use a flag array to keep track of which variables need to be checked for
        # removal (i.e., we check a variable only when the corresponding flag is False)
        self.skip_rem_check_vec: np.ndarray | None = None  # SkipCheck_VEC in the paper
        self.markov_boundary_matrix: np.ndarray | None = None
        self.learned_skeleton: nx.Graph | None = None
        self.is_rsl_d: bool = False
        self.clique_num: int | None = None

    def learn_and_get_skeleton(
        self,
        data: np.ndarray,
        clique_num: int | None = None,
    ) -> nx.Graph:
        """Learn and return the skeleton implied by the data.

        Parameters
        ----------
        data : np.ndarray
            Matrix of shape ``(n_samples, n_vars)`` whose columns correspond to
            variables.
        clique_num : int, optional
            Upper bound on the clique number. Required for algorithms that are
            not diamond-free variants.

        Returns
        -------
        nx.Graph
            Undirected skeleton learned by the recursive elimination procedure.

        Raises
        ------
        ValueError
            If ``clique_num`` is not provided for algorithms that require it.
        """

        self._prepare_learning_state(data, clique_num)

        if self.num_vars is None:
            raise RuntimeError("Learning state failed to initialize number of variables.")

        skeleton = nx.Graph()
        skeleton.add_nodes_from(range(self.num_vars))

        self._run_recursive_elimination(build_skeleton=True, skeleton=skeleton)
        self.learned_skeleton = skeleton
        return skeleton

    def compute_removal_order(
        self,
        data: np.ndarray,
        clique_num: int | None = None,
    ) -> np.ndarray:
        """Compute the removal (r-) order without constructing the skeleton.

        Parameters
        ----------
        data : np.ndarray
            Matrix of shape ``(n_samples, n_vars)`` whose columns correspond to
            variables.
        clique_num : int, optional
            Upper bound on the clique number. Required for non diamond-free
            variants.

        Returns
        -------
        np.ndarray
            Integer array containing the order in which variables are removed.

        Raises
        ------
        ValueError
            If ``clique_num`` is not provided for algorithms that require it.
        """

        self._prepare_learning_state(data, clique_num)
        return self._run_recursive_elimination(build_skeleton=False)

    def find_neighborhood(self, var: int) -> np.ndarray:
        """Return the neighborhood of ``var`` in the current skeleton estimate.

        Parameters
        ----------
        var : int
            Index of the variable whose neighborhood is requested.

        Returns
        -------
        np.ndarray
            Indices that are neighbors of ``var``.
        """

        raise NotImplementedError

    def is_removable(self, var: int) -> bool:
        """Determine whether ``var`` can be removed without violating constraints.

        Parameters
        ----------
        var : int
            Index of the candidate variable.

        Returns
        -------
        bool
            ``True`` if the variable can be removed, ``False`` otherwise.
        """

        raise NotImplementedError

    def find_removable(self, var_arr: np.ndarray) -> int:
        """Locate the first removable variable in ``var_arr``.

        Parameters
        ----------
        var_arr : np.ndarray
            Candidate variable indices ordered by preference.

        Returns
        -------
        int
            Index of the first removable variable, or ``REMOVABLE_NOT_FOUND`` if
            none qualify.
        """

        if self.skip_rem_check_vec is None:
            raise RuntimeError("Learning state has not been initialized.")

        for var in var_arr:
            if self.is_removable(var):
                return var
            self.skip_rem_check_vec[var] = True
        return REMOVABLE_NOT_FOUND

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _prepare_learning_state(
        self,
        data: np.ndarray,
        clique_num: int | None,
    ) -> None:
        """Validate inputs and initialize shared state for a learning run."""

        if not self.is_rsl_d and clique_num is None:
            raise ValueError("Clique number not given!")

        self.num_vars = data.shape[1]
        self.data = data
        self.clique_num = clique_num

        self.skip_rem_check_vec = np.zeros(self.num_vars, dtype=bool)
        self.markov_boundary_matrix = self.find_markov_boundary_matrix(self.data)
        self.learned_skeleton = None

    def _run_recursive_elimination(
        self,
        build_skeleton: bool,
        skeleton: nx.Graph | None = None,
    ) -> np.ndarray:
        """Run the recursive elimination routine and optionally build the skeleton."""

        if self.data is None or self.markov_boundary_matrix is None:
            raise RuntimeError("Learning state has not been initialized.")

        if build_skeleton and skeleton is None:
            raise ValueError("Skeleton must be provided when build_skeleton is True.")

        if self.skip_rem_check_vec is None:
            raise RuntimeError("Skip-check vector is not initialized.")

        if self.num_vars is None:
            raise RuntimeError("Number of variables is unknown.")

        num_vars = self.num_vars
        r_order = np.zeros(num_vars, dtype=int)
        var_arr = np.arange(num_vars)
        var_left_mask = np.ones(num_vars, dtype=bool)

        data_included_ci_test = lambda x, y, z: self.ci_test(x, y, z, self.data)  # noqa: E731

        for iter_idx in range(num_vars - 1):
            # only consider variables that are still left and whose removal needs checking
            candidate_vars = var_arr[var_left_mask & ~self.skip_rem_check_vec]

            # sort the remaining candidates by the size of their Markov boundary
            mb_size = np.sum(self.markov_boundary_matrix[candidate_vars], axis=1)
            sorted_candidates = candidate_vars[np.argsort(mb_size, kind="stable")]

            removable_var = self.find_removable(sorted_candidates)

            if removable_var == REMOVABLE_NOT_FOUND:
                # if no removable variable was found, fall back to the smallest Markov boundary
                remaining_vars = np.flatnonzero(var_left_mask)
                mb_size_all = np.sum(self.markov_boundary_matrix[remaining_vars], axis=1)
                removable_var = remaining_vars[np.argmin(mb_size_all)]
                self.skip_rem_check_vec[:] = False

            # find the neighbors of the removable variable using the subclass rule
            neighbors = self.find_neighborhood(removable_var)

            # update the Markov boundary matrix to reflect the removal
            update_markov_boundary_matrix(
                self.markov_boundary_matrix,
                data_included_ci_test,
                removable_var,
                neighbors,
                self.is_rsl_d,
                skip_check=self.skip_rem_check_vec,
            )

            if build_skeleton and skeleton is not None:
                # add edges between the removable variable and its neighbors
                for neighbor_idx in neighbors:
                    skeleton.add_edge(removable_var, neighbor_idx)

            # mark the variable as removed and record it in the r-order
            var_left_mask[removable_var] = False
            r_order[iter_idx] = removable_var

        remaining_var = np.flatnonzero(var_left_mask)
        if remaining_var.size != 1:
            raise RuntimeError("Recursive elimination did not terminate correctly.")
        r_order[-1] = remaining_var[0]
        return r_order
