"""Conditional independence (CI) tests used throughout the RCD package."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from rcd.utilities.utils import sanitize_data


CiTest = Callable[[int, int, list[int], np.ndarray | pd.DataFrame], bool]


def is_d_separated(graph: nx.DiGraph, x: int, y: int, z: Iterable[int]) -> bool:
    """Return ``True`` when ``x`` and ``y`` are *d*-separated by ``z``."""

    z_set = set(z)

    def has_path_to_y(node: int, visited: set[int], through_collider: bool) -> bool:
        if node == y:
            return True
        if node in visited:
            return False
        visited.add(node)

        for neighbor in graph.successors(node):
            if neighbor not in z_set and not through_collider:
                if has_path_to_y(neighbor, visited, False):
                    return True

        for neighbor in graph.predecessors(node):
            if neighbor not in z_set:
                if has_path_to_y(neighbor, visited, node in z_set):
                    return True

        return False

    return not has_path_to_y(x, set(), False)


def get_perfect_ci_test(adj_mat: np.ndarray) -> CiTest:
    """Return an oracle CI test induced by ``adj_mat``."""

    dag = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)

    def perfect_ci(x_idx: int, y_idx: int, cond_set: list[int], data) -> bool:  # data is unused
        return is_d_separated(dag, x_idx, y_idx, cond_set)

    return perfect_ci


def fisher_z(
    x_idx: int,
    y_idx: int,
    cond_set: list[int],
    data: np.ndarray | pd.DataFrame,
    significance_level: float = 0.01,
) -> bool:
    """Gaussian Fisher-Z conditional independence test."""

    data_mat = sanitize_data(data)
    num_samples = data_mat.shape[0]
    indices = [x_idx, y_idx] + list(cond_set)
    data_subset = data_mat[:, indices]

    R = np.corrcoef(data_subset, rowvar=False)
    P = np.linalg.pinv(R)

    ro = -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])
    zro = 0.5 * np.log((1 + ro) / (1 - ro))

    c_val = stats.norm.ppf(1 - significance_level / 2)
    threshold = c_val / np.sqrt(max(num_samples - len(cond_set) - 3, 1))
    return abs(zro) < threshold


def chi_square(
    x_idx: int,
    y_idx: int,
    cond_set: list[int],
    data: np.ndarray | pd.DataFrame,
    significance_level: float = 0.05,
    return_statistic: bool = False,
) -> bool | tuple[float, float, int]:
    """Pearson chi-squared conditional independence test."""

    return power_divergence(
        x_idx,
        y_idx,
        cond_set,
        data,
        significance_level=significance_level,
        lambda_="pearson",
        return_statistic=return_statistic,
    )


def g_sq(
    x_idx: int,
    y_idx: int,
    cond_set: list[int],
    data: np.ndarray | pd.DataFrame,
    significance_level: float = 0.05,
    return_statistic: bool = False,
) -> bool | tuple[float, float, int]:
    """G-squared (likelihood ratio) conditional independence test."""

    return power_divergence(
        x_idx,
        y_idx,
        cond_set,
        data,
        significance_level=significance_level,
        lambda_="log-likelihood",
        return_statistic=return_statistic,
    )


def log_likelihood(
    x_idx: int,
    y_idx: int,
    cond_set: list[int],
    data: np.ndarray | pd.DataFrame,
    significance_level: float = 0.05,
    return_statistic: bool = False,
) -> bool | tuple[float, float, int]:
    """Alias for the G-test (maintained for backwards compatibility)."""

    return g_sq(
        x_idx,
        y_idx,
        cond_set,
        data,
        significance_level=significance_level,
        return_statistic=return_statistic,
    )


def freeman_tuckey(
    x_idx: int,
    y_idx: int,
    cond_set: list[int],
    data: np.ndarray | pd.DataFrame,
    significance_level: float = 0.05,
    return_statistic: bool = False,
) -> bool | tuple[float, float, int]:
    """Freeman–Tukey power divergence CI test."""

    return power_divergence(
        x_idx,
        y_idx,
        cond_set,
        data,
        significance_level=significance_level,
        lambda_="freeman-tukey",
        return_statistic=return_statistic,
    )


def modified_log_likelihood(
    x_idx: int,
    y_idx: int,
    cond_set: list[int],
    data: np.ndarray | pd.DataFrame,
    significance_level: float = 0.05,
    return_statistic: bool = False,
) -> bool | tuple[float, float, int]:
    """Modified log-likelihood CI test."""

    return power_divergence(
        x_idx,
        y_idx,
        cond_set,
        data,
        significance_level=significance_level,
        lambda_="mod-log-likelihood",
        return_statistic=return_statistic,
    )


def neyman(
    x_idx: int,
    y_idx: int,
    cond_set: list[int],
    data: np.ndarray | pd.DataFrame,
    significance_level: float = 0.05,
    return_statistic: bool = False,
) -> bool | tuple[float, float, int]:
    """Neyman power divergence CI test."""

    return power_divergence(
        x_idx,
        y_idx,
        cond_set,
        data,
        significance_level=significance_level,
        lambda_="neyman",
        return_statistic=return_statistic,
    )


def cressie_read(
    x_idx: int,
    y_idx: int,
    cond_set: list[int],
    data: np.ndarray | pd.DataFrame,
    significance_level: float = 0.05,
    return_statistic: bool = False,
) -> bool | tuple[float, float, int]:
    """Cressie–Read power divergence CI test."""

    return power_divergence(
        x_idx,
        y_idx,
        cond_set,
        data,
        significance_level=significance_level,
        lambda_="cressie-read",
        return_statistic=return_statistic,
    )


def power_divergence(
    x_idx: int,
    y_idx: int,
    cond_set: list[int],
    data: np.ndarray | pd.DataFrame,
    *,
    significance_level: float = 0.05,
    lambda_: float | str = "cressie-read",
    return_statistic: bool = False,
) -> bool | tuple[float, float, int]:
    """Generic power-divergence CI test covering multiple classical statistics."""

    cond_list = list(cond_set)
    data_df = _coerce_dataframe(data)

    if x_idx in cond_list or y_idx in cond_list:
        raise ValueError("Conditioning set must not contain the test variables.")

    if len(cond_list) == 0:
        contingency = data_df.groupby([x_idx, y_idx]).size().unstack(y_idx, fill_value=0)
        chi, p_value, dof, _ = stats.chi2_contingency(contingency, lambda_=lambda_)
    else:
        chi = 0.0
        dof = 0
        for _, df in data_df.groupby(cond_list):
            if df.empty:
                continue
            try:
                contingency = df.groupby([x_idx, y_idx]).size().unstack(y_idx, fill_value=0)
                c_val, _, d_val, _ = stats.chi2_contingency(contingency, lambda_=lambda_)
                chi += c_val
                dof += d_val
            except ValueError:
                # insufficient support for this conditioning assignment
                continue
        if dof == 0:
            return True if not return_statistic else (0.0, 1.0, 0)
        p_value = 1 - stats.chi2.cdf(chi, df=dof)

    if return_statistic:
        return chi, p_value, dof
    return p_value >= significance_level


def pearsonr(
    x_idx: int,
    y_idx: int,
    cond_set: list[int],
    data: np.ndarray | pd.DataFrame,
    significance_level: float = 0.05,
    return_statistic: bool = False,
) -> bool | tuple[float, float]:
    """Pearson (partial) correlation test."""

    data_mat = sanitize_data(data)
    cond_list = list(cond_set)

    if len(cond_list) == 0:
        coef, p_value = stats.pearsonr(data_mat[:, x_idx], data_mat[:, y_idx])
    else:
        z_mat = data_mat[:, cond_list]
        x_coef = np.linalg.lstsq(z_mat, data_mat[:, x_idx], rcond=None)[0]
        y_coef = np.linalg.lstsq(z_mat, data_mat[:, y_idx], rcond=None)[0]

        residual_x = data_mat[:, x_idx] - z_mat @ x_coef
        residual_y = data_mat[:, y_idx] - z_mat @ y_coef
        coef, p_value = stats.pearsonr(residual_x, residual_y)

    if return_statistic:
        return coef, p_value
    return p_value >= significance_level


def _coerce_dataframe(data: np.ndarray | pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with integer columns regardless of input type."""

    data_mat = sanitize_data(data)
    num_vars = data_mat.shape[1]
    return pd.DataFrame(data_mat, columns=list(range(num_vars)))

