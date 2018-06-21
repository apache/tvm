# pylint: disable=invalid-name
"""Metrics for evaluating tuning process"""

import numpy as np

from ..util import get_rank

def max_curve(trial_scores):
    """ f(n) = max([s[i] fo i < n])

    Parameters
    ----------
    trial_scores: Array of float
        the score of i th trial

    Returns
    -------
    curve: Array of float
        function values
    """
    ret = np.empty(len(trial_scores))
    keep = -1e9
    for i, score in enumerate(trial_scores):
        keep = max(keep, score)
        ret[i] = keep
    return ret

def mean_curve(trial_scores):
    """ f(n) = mean([s[i] fo i < n])

    Parameters
    ----------
    trial_scores: Array of float
        the score of i th trial

    Returns
    -------
    curve: Array of float
        function values
    """
    ret = np.empty(len(trial_scores))
    keep = 0
    for i, score in enumerate(trial_scores):
        keep += score
        ret[i] = keep / (i+1)
    return ret

def recall_curve(trial_ranks, top=None):
    """
    if top is None, f(n) = sum([I(rank[i] < n) for i < n]) / n
    if top is K,    f(n) = sum([I(rank[i] < K) for i < n]) / K

    Parameters
    ----------
    trial_ranks: Array of int
        the rank of i th trial in labels
    top: int or None
        top-n recall

    Returns
    -------
    curve: Array of float
        function values
    """
    if not isinstance(trial_ranks, np.ndarray):
        trial_ranks = np.array(trial_ranks)

    ret = np.zeros(len(trial_ranks))
    if top is None:
        for i in range(len(trial_ranks)):
            ret[i] = np.sum(trial_ranks[:i] <= i) / (i+1)
    else:
        for i in range(len(trial_ranks)):
            ret[i] = 1.0 * np.sum(trial_ranks[:i] < top) / top
    return ret

def cover_curve(trial_ranks):
    """
    f(n) = max k s.t. {1,2,...,k} is a subset of {ranks[i] for i < n}

    Parameters
    ----------
    trial_ranks: Array of int
        the rank of i th trial in labels

    Returns
    -------
    curve: Array of float
        function values
    """
    ret = np.empty(len(trial_ranks))
    keep = -1
    cover = set()
    for i, rank in enumerate(trial_ranks):
        cover.add(rank)
        while keep+1 in cover:
            keep += 1
        ret[i] = keep + 1
    return ret / len(trial_ranks)


# feval wrapper for xgboost
def xgb_max_curve_score(N):
    """evaluate max curve score for xgb"""
    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        scores = labels[trials]
        curve = max_curve(scores)
        return "Smax@%d" % N, curve[N] / np.max(labels)
    return feval

def xgb_recalln_curve_score(N):
    """evaluate recall-n curve score for xgb"""
    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = recall_curve(ranks)
        return "recall@%d" % N, curve[N]
    return feval

def xgb_average_recalln_curve_score(N):
    """evaluate average recall-n curve score for xgb"""
    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = recall_curve(ranks)
        return "a-recall@%d" % N, np.sum(curve[:N]) / N
    return feval

def xgb_recallk_curve_score(N, topk):
    """evaluate recall-k curve score for xgb"""
    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = recall_curve(ranks, topk)
        return "recall@%d" % topk, curve[N]
    return feval

def xgb_cover_curve_score(N):
    """evaluate cover curve score for xgb"""
    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = cover_curve(ranks)
        return "cover@%d" % N, curve[N]
    return feval

def xgb_null_score(_):
    """empty score function for xgb"""
    def feval(__, ___):
        return "null", 0
    return feval

def average_recall(preds, labels, N):
    """evaluate average recall-n for predictions and labels"""
    trials = np.argsort(preds)[::-1]
    ranks = get_rank(labels[trials])
    curve = recall_curve(ranks)
    return np.sum(curve[:N]) / N
