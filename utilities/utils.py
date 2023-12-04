from typing import Tuple

import networkx as nx


def f1_score_edges(true_graph: nx.Graph, est_graph: nx.Graph, return_only_f1=True) -> float | tuple[
    float, float, float]:
    edges = set(true_graph.edges())
    edges_est = set(est_graph.edges())

    # compute F1 score
    precision = len(edges.intersection(edges_est)) / len(edges_est)
    recall = len(edges.intersection(edges_est)) / len(edges)
    f1_score = 2 * precision * recall / (precision + recall)

    if return_only_f1:
        return f1_score
    else:
        return precision, recall, f1_score
