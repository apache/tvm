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
# pylint: disable=unused-variable, invalid-name
"""Functions to cluster and select tasks for selective tuning.
"""

import logging

import numpy as np

logger = logging.getLogger('autotvm')


def compute_similarity(task1, task2):
    """Compute the similarity of two tasks' tuning spaces.

    Parameters
    ----------
    task1: autotvm.task.Task
        the first task.

    task2: autotvm.task.Task
        the second task.

    Returns
    -------
        return the similarity rate.
    """
    # Different op must have zero similarity
    if task1.name != task2.name:
        return 0

    space1 = task1.config_space.space_map
    space2 = task2.config_space.space_map

    # Config space unmatch. May be due to different schedule templates
    union_space = set(space1)
    union_space.update(space2)
    if (len(space1) != len(space2) or len(space1) != len(union_space)):
        return 0

    return np.prod([space1[name].similar(space2[name]) for name in union_space])


def compute_psm(tasks):
    """Compute a pairwise similarity matrix (PSM) for given tasks.

    Parameters
    ----------
    tasks: List[autotvm.task.Task]
        the tasks to be computed.

    Returns
    -------
    psm: List[List[Float]]
        a NxN PSM. PSM(i, j) is the similarity of task i and task j.
    """
    psm = [[1.0 for _ in range(len(tasks))] for _ in range(len(tasks))]
    for idx1, task1 in enumerate(tasks):
        for idx2 in range(idx1 + 1, len(tasks)):
            psm[idx1][idx2] = psm[idx2][idx1] = compute_similarity(task1, tasks[idx2])

    if logger.isEnabledFor(logging.DEBUG):
        print('Pairwise Similarity Matrix:')
        for row in psm:
            print('%s -> %.2f' % (', '.join(
                ['{:.2f}'.format(r) if r >= 0.01 else '----'
                 for r in row]), sum(row)))
    return psm


def psm_clustering(tasks):
    """Cluster given tasks to several groups and select one task per group using
    pairwise simularity matrix (PSM) and graph cliques. We first compute the PSM
    that includes simularity rate (SM) of any two tasks. The simularity rate is the
    ratio of overlapped tuning space between tasks. Accordingly, we create a graph
    with tasks as nodes and SM (>=0.01) as edge weight, and then find all cliques
    in the graph as clusters. For the task that has been included by more than one
    cliques, we assign it to the clique with higher weight sum between the task and
    other tasks in the clique. Finally, every non-empty clique is a cluster, and
    the task with the highest weight sum is the representative task of that cluster,
    meaning that all other tasks in the cluster will depend on it.

    Parameters
    ----------
    tasks: List[autotvm.task.Task]
        the tasks to be clustered.

    Returns
    -------
    (centroids, labels): Tuple[List[int], List[int]]
        the index of selected tasks and the cluster each task belongs to.
    """
    import networkx as nx

    def weight_sum(psm, prim, targets):
        """Sum of the simularity rates of prim task to target tasks"""
        return sum([psm[prim][t] for t in targets])

    # Precompute the pairwise similarity matrix (PSM)
    psm = compute_psm(tasks)

    # Create a graph with task index as nodes and PSM as edge weights
    graph = nx.Graph()
    graph.add_nodes_from(range(len(tasks)))
    graph.add_edges_from([(i, j) for i in range(len(tasks))
                          for j in range(i + 1, len(tasks))
                          if psm[i][j] >= 0.01])

    # Cluster assignment for each task (List[clique index], assigned cluster index)
    assigned = [([], None) for _ in range(len(tasks))]

    # Find cliques and initialize clusters
    clusters = []
    for cidx, clique in enumerate(nx.find_cliques(graph)):
        clusters.append(set())
        for idx in clique:
            assigned[idx][0].append(cidx)

    # Assign the tasks that only belong to one clique to the cluster
    for idx in range(len(tasks)):
        if len(assigned[idx]) == 1:
            clusters[assigned[idx][0][0]].add(idx)
            assigned[idx][1] = assigned[idx][0][0]

    changed = True
    while changed:
        if logger.isEnabledFor(logging.DEBUG):
            print('Round')
            for idx, clut in enumerate(clusters):
                print('%d: %s' % (idx, ','.join([str(i) for i in clut])))
        changed = False
        for idx in range(len(tasks)):
            if len(assigned[idx]) == 1:
                continue
            new_cidx = max(assigned[idx][0], key=lambda c: weight_sum(psm, idx, clusters[c]))
            if new_cidx != assigned[idx][1]:
                changed = True
                clusters[new_cidx].add(idx)
                if assigned[idx][1] is not None:
                    clusters[assigned[idx][1]].remove(idx)
                assigned[idx] = (assigned[idx][0], new_cidx)

    # Create labels
    labels = [label for _, label in assigned]

    # For each cluster, select the task that has the maximum weight sum to other tasks in cluster
    centroids = []
    for clut in clusters:
        if clut:
            centroids.append(max(clut, key=lambda p: weight_sum(psm, p, clut)))
        else: # Empty cluster
            centroids.append(-1)
    return centroids, labels

def mark_depend(tasks):
    """Mark some tasks to depend on other tasks by assigning task.depend = other_task.
    It means the tuner will only use the top schedules of the dependent task when
    that task has been tuned already.

    Parameters
    ----------
    tasks: List[tvm.autotvm.task.Task]
        the tasks to be analyzed and marked.
    """

    assert all([t.workload is not None
                for t in tasks]), "One or more tasks have undefined workload"

    if not tasks or tasks[0].target.target_name != 'cuda':
        logging.warning('Make dependent tasks for CPU is unstable')

    centroids, labels = psm_clustering(tasks)

    if logger.isEnabledFor(logging.DEBUG):
        print('Selected task index: %s' % ', '.join([str(c) for c in centroids]))
        print('Dependent task index: %s' % ', '.join([str(p) for p in labels]))

    for idx, task in enumerate(tasks):
        if labels[idx] != -1:
            task.depend = tasks[centroids[labels[idx]]]
        else:  # Outliers depend on itself to guarantee the performance
            logger.debug('task %s does not have dependent', str(task))

    logger.info('Select %d tasks over %d tasks ',
                sum([1 if t.depend == t else 0 for t in tasks]), len(tasks))
