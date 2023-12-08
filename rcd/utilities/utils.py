import networkx as nx


def f1_score_edges(true_graph: nx.Graph, est_graph: nx.Graph, return_only_f1=True) -> float | tuple[float, float, float]:
    """
    Compute the F1 score of the estimated graph with respect to the true graph, using edges as the unit of comparison.
    :param true_graph: The true graph
    :param est_graph: The estimated/predicted graph
    :param return_only_f1: If True, return only the F1 score. If False, return precision, recall, and F1 score.
    :return:
    """
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
