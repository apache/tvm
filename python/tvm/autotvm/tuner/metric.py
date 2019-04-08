# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
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

def average_recall(preds, labels, N):
    """evaluate average recall-n for predictions and labels"""
    trials = np.argsort(preds)[::-1]
    ranks = get_rank(labels[trials])
    curve = recall_curve(ranks)
    return np.sum(curve[:N]) / N
