import math

from src.evaluation import citation_precision, citation_recall, mean, recall_at_k, reciprocal_rank


def test_retrieval_metrics():
    retrieved = [8, 3, 5, 9]
    relevant = {3, 5}
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert reciprocal_rank(retrieved, relevant) == 0.5


def test_citation_metrics():
    assert citation_precision([3, 3, 99], {3, 7}) == 0.5
    assert citation_recall([3], {3, 7}) == 0.5


def test_mean_empty_is_nan():
    assert math.isnan(mean([]))
    assert mean([0.0, 1.0]) == 0.5
