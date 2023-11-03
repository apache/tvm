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
# pylint: disable=invalid-name, too-many-locals, unnecessary-list-index-lookup
"""Partitioned Boolean Quadratic Programming Tuner"""
from ._base import INVALID_LAYOUT_TIME
from .base_graph_tuner import BaseGraphTuner
from .utils import is_boundary_node, has_multiple_inputs


class PBQPTuner(BaseGraphTuner):
    """An approximation method to deal with intractably
    large size of graph tuning problem.

    This graph coloring algorithm mainly comes from:

    Lang Hames and Bernhard Scholz.
    Nearly optimal register allocation with pbqp.JMLC 2006.
    LNCS, vol.4228,pp. 346-361, 2016
    """

    def __init__(self, *args, **kwargs):
        """Create a partitioned boolean quadratic programming tuner."""
        super(PBQPTuner, self).__init__(*args, **kwargs)

        # Remove input and ruled_out nodes
        input_names = self._input_shapes.keys()
        for node_idx in self._out_nodes_dict:
            node = self._node_list[node_idx]
            if is_boundary_node(node, input_names):
                for out_node_idx in self._out_nodes_dict[node_idx]:
                    self._in_nodes_dict[out_node_idx].remove(node_idx)

        self._adj_dict = {}
        for node_idx in self._in_nodes_dict:
            self._adj_dict[node_idx] = list(self._in_nodes_dict[node_idx]) + list(
                self._out_nodes_dict[node_idx]
            )

        self._record_cost_dict = {}
        for key in self._in_nodes_dict:
            self._record_cost_dict[key] = []
            for record in self._node_list[key]["record_candidates"]:
                self._record_cost_dict[key].append(record[1].costs[0])

        self._max_degree = -1
        self._node_degree_dict = {}
        for node_idx in self._in_nodes_dict:
            node_degree = self._get_degree(node_idx)
            self._node_degree_dict[node_idx] = node_degree
            self._max_degree = max(self._max_degree, node_degree)

        self._stack = []
        self._buckets = [[] for _ in range(self._max_degree + 2)]
        for node_idx in sorted(self._in_nodes_dict):
            node_degree = self._get_degree(node_idx)
            self._buckets[node_degree].append(node_idx)

        self._is_optimal = True

    def _get_degree(self, node_idx):
        """Get node degree."""
        return len(self._adj_dict[node_idx])

    def _reorder_adj_nodes(self, node_idx):
        """Update buckets list with current adjacency list."""
        for adj_node in self._adj_dict[node_idx]:
            current_degree = self._get_degree(adj_node)
            prev_degree = self._node_degree_dict[adj_node]
            if prev_degree != current_degree:
                self._buckets[prev_degree].remove(adj_node)
                self._buckets[current_degree].insert(0, adj_node)
                self._node_degree_dict[adj_node] = current_degree

    def _remove_node(self, node_idx):
        """Remove node from graph. Update adjacency list accordingly."""
        node_degree = self._get_degree(node_idx)
        self._buckets[node_degree].remove(node_idx)
        for adj_node in self._adj_dict[node_idx]:
            self._adj_dict[adj_node].remove(node_idx)

    def _insert_edge(self, node_x, node_y, adj_cost_matrix):
        """Insert an edge between two nodes."""
        self._layout_transform_interlayer_cost[(node_x, node_y)] = adj_cost_matrix
        self._layout_transform_interlayer_cost[(node_y, node_x)] = []
        for i in range(len(adj_cost_matrix[0])):
            self._layout_transform_interlayer_cost[(node_y, node_x)].append([])
            for cost_vec in adj_cost_matrix:
                self._layout_transform_interlayer_cost[(node_y, node_x)][i].append(cost_vec[i])

        self._adj_dict[node_x].append(node_y)
        self._adj_dict[node_y].append(node_x)

    def _backward_insert_node(self, node_idx):
        """Reinsert node in backward pass."""
        for adj_node in self._adj_dict[node_idx]:
            self._adj_dict[adj_node].append(node_idx)

    def _RI_reduction(self, node_idx):
        """Reduce nodes with degree 1."""
        adj_node = self._adj_dict[node_idx][0]
        ltf_matrix = self._layout_transform_interlayer_cost[(adj_node, node_idx)]
        for i, cost_vec in enumerate(ltf_matrix):
            min_cost = INVALID_LAYOUT_TIME
            for j, cost in enumerate(cost_vec):
                min_cost = min(min_cost, cost + self._record_cost_dict[node_idx][j])
            self._record_cost_dict[adj_node][i] += min_cost
        self._remove_node(node_idx)
        self._reorder_adj_nodes(node_idx)
        self._stack.append(node_idx)

    def _RII_reduction(self, node_idx):
        """Reduce nodes with degree 2."""
        adj_node_x, adj_node_y = self._adj_dict[node_idx]
        ltf_matrix_x = self._layout_transform_interlayer_cost[(adj_node_x, node_idx)]
        ltf_matrix_y = self._layout_transform_interlayer_cost[(adj_node_y, node_idx)]
        delta_matrix = [[] for _ in range(len(ltf_matrix_x))]
        for i, cost_vec_x in enumerate(ltf_matrix_x):
            for j, cost_vec_y in enumerate(ltf_matrix_y):
                min_cost = INVALID_LAYOUT_TIME
                for k in range(len(self._record_cost_dict[node_idx])):
                    min_cost = min(
                        min_cost,
                        cost_vec_x[k] + cost_vec_y[k] + self._record_cost_dict[node_idx][k],
                    )
                delta_matrix[i].append(min_cost)

        if adj_node_x == adj_node_y:
            for i, delta_row in enumerate(delta_matrix):
                self._record_cost_dict[adj_node_x][i] += delta_row[i]
        elif adj_node_x in self._adj_dict[adj_node_y]:
            for i, _ in enumerate(delta_matrix):
                for j, delta in enumerate(delta_matrix[i]):
                    self._layout_transform_interlayer_cost[(adj_node_x, adj_node_y)][i][j] += delta
                    self._layout_transform_interlayer_cost[(adj_node_y, adj_node_x)][j][i] += delta
        else:
            self._insert_edge(adj_node_x, adj_node_y, delta_matrix)

        self._remove_node(node_idx)
        self._reorder_adj_nodes(node_idx)
        self._stack.append(node_idx)

    def _RN_reduction(self, node_idx):
        """Reduce nodes with degree greater than 2."""
        min_cost = INVALID_LAYOUT_TIME
        record_idx = -1

        for i, record_cost in enumerate(self._record_cost_dict[node_idx]):
            current_cost = record_cost
            for adj_node in self._adj_dict[node_idx]:
                ltf_matrix = self._layout_transform_interlayer_cost[(node_idx, adj_node)]
                adj_record_cost = list(self._record_cost_dict[adj_node])
                for j, ltf_cost in enumerate(ltf_matrix[i]):
                    adj_record_cost[j] += ltf_cost
                current_cost += min(adj_record_cost)
            if current_cost < min_cost:
                min_cost = current_cost
                record_idx = i

        if record_idx < 0:
            raise RuntimeError(
                f"Can't find a soltuion for node {node_idx} when applying RN reduction"
            )
        self._optimal_record_dict[node_idx] = record_idx
        self._is_optimal = False

        for adj_node in self._adj_dict[node_idx]:
            ltf_matrix = self._layout_transform_interlayer_cost[(node_idx, adj_node)]
            for i, ltf_cost in enumerate(ltf_matrix[record_idx]):
                self._record_cost_dict[adj_node][i] += ltf_cost

        self._remove_node(node_idx)
        self._reorder_adj_nodes(node_idx)
        self._stack.append(node_idx)

    def _forward(self):
        """Forward pass in PBQP to reduce nodes."""
        while True:
            if self._buckets[1]:
                node_idx = self._buckets[1][0]
                self._RI_reduction(node_idx)
            elif self._max_degree >= 2 and self._buckets[2]:
                node_idx = self._buckets[2][0]
                self._RII_reduction(node_idx)
            elif self._max_degree >= 3:
                max_degree_node = -1
                for i in range(self._max_degree, 2, -1):
                    if self._buckets[i]:
                        max_degree_node = self._buckets[i][0]
                        self._RN_reduction(max_degree_node)
                        break
                if max_degree_node < 0:
                    break
            else:
                break

    def _backward(self):
        """Backward pass in PBQP to generate optimal solution."""
        # Solve nodes left in the forward graph
        for node_idx in self._buckets[0]:
            record_costs = self._record_cost_dict[node_idx]
            min_cost = min(record_costs)
            self._optimal_record_dict[node_idx] = record_costs.index(min_cost)

        # Solve nodes with one or two degrees
        for node_idx in reversed(self._stack):
            self._backward_insert_node(node_idx)
            if node_idx not in self._optimal_record_dict:
                record_costs = list(self._record_cost_dict[node_idx])
                for adj_node in self._adj_dict[node_idx]:
                    adj_optimal_idx = self._optimal_record_dict[adj_node]
                    for i, _ in enumerate(record_costs):
                        record_costs[i] += self._layout_transform_interlayer_cost[
                            (node_idx, adj_node)
                        ][i][adj_optimal_idx]
                min_cost = min(record_costs)
                self._optimal_record_dict[node_idx] = record_costs.index(min_cost)

    def run(self, **kwargs):
        """Run partitioned boolean quadratic programming tuner."""
        self._logger.info("Start to run PBQP algorithm...")
        # Define virtual record lists and layout transformaton matrices
        # for multi-input nodes.
        input_names = self._input_shapes.keys()
        temp = {}
        for key, val in self._in_nodes_dict.items():
            target_input_idx = -1
            target_input_pos = -1
            if has_multiple_inputs(self._node_list, key, input_names, self._opt_out_op):
                for i, item in enumerate(val):
                    node = self._node_list[item]
                    if not is_boundary_node(node, input_names):
                        target_input_idx = item
                        target_input_pos = i
                        break

                # Skip boundary operator
                if target_input_idx < 0:
                    continue

                temp[(target_input_idx, key)] = []
                record_candidates = self._node_list[target_input_idx]["record_candidates"]
                for j in range(len(record_candidates)):
                    temp[(target_input_idx, key)].append([])
                    for k in range(len(record_candidates)):
                        temp[(target_input_idx, key)][j].append(
                            0 if j == k else INVALID_LAYOUT_TIME
                        )

                for j in range(target_input_pos + 1, len(val)):
                    input_idx = val[j]
                    input_node = self._node_list[input_idx]
                    if is_boundary_node(input_node, input_names):
                        continue
                    temp[(input_idx, key)] = self._layout_transform_interlayer_cost[
                        (input_idx, target_input_idx)
                    ]
        self._layout_transform_interlayer_cost.update(temp)

        # Create reverse layout transformation matrices
        temp = {}
        for idx_pair, ltf_matrix in self._layout_transform_interlayer_cost.items():
            reverse_key = (idx_pair[1], idx_pair[0])
            reverse_matrix = [[] for _ in range(len(ltf_matrix[0]))]
            for i, _ in enumerate(ltf_matrix):
                for j, ltf in enumerate(ltf_matrix[i]):
                    reverse_matrix[j].append(ltf)
            temp[reverse_key] = reverse_matrix
        self._layout_transform_interlayer_cost.update(temp)

        self._forward()
        self._backward()
        is_optimal = "optimal" if self._is_optimal else "sub-optimal"
        msg = f"Finished PBQPExecutor run. Got {is_optimal} solution."
        self._logger.info(msg)
