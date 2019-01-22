"""Partitioned Boolean Quadratic Programming Tuner"""
from ._base import INVALID_LAYOUT_TIME
from .base_graph_tuner import BaseGraphTuner
from .utils import is_input_node, has_multiple_inputs


class PBQPTuner(BaseGraphTuner):
    """An approximation method to deal with intractably
    large size of graph tuning problem.
    """
    def __init__(self, *args, **kwargs):
        """Create a partitioned boolean quadratic programming tuner.
        """
        super(PBQPTuner, self).__init__(*args, **kwargs)

        # Remove input nodes
        for node_idx in self._out_nodes_dict.keys():
            if is_input_node(self._node_list, node_idx):
                for out_node_idx in self._out_nodes_dict[node_idx]:
                    self._in_nodes_dict[out_node_idx].remove(node_idx)

        self._adj_dict = {}
        for node_idx in self._in_nodes_dict.keys():
            self._adj_dict[node_idx] = list(self._in_nodes_dict[node_idx]) + \
                                       list(self._out_nodes_dict[node_idx])

        self._sch_cost_dict = {}
        for key, val in self._sch_dict.items():
            self._sch_cost_dict[key] = []
            for item in val:
                self._sch_cost_dict[key].append(item["cost"])

        self._max_degree = -1
        self._node_degree_dict = {}
        for node_idx in self._in_nodes_dict.keys():
            node_degree = self._get_degree(node_idx)
            self._node_degree_dict[node_idx] = node_degree
            self._max_degree = max(self._max_degree, node_degree)

        self._stack = []
        self._buckets = [[] for _ in range(self._max_degree + 2)]
        for node_idx in sorted(self._in_nodes_dict.keys()):
            node_degree = self._get_degree(node_idx)
            self._buckets[node_degree].append(node_idx)

        self._min_cost = 0
        self._is_optimal = True

    def _get_degree(self, node_idx):
        return len(self._adj_dict[node_idx])

    def _reorder_adj_nodes(self, node_idx):
        for adj_node in self._adj_dict[node_idx]:
            current_degree = self._get_degree(adj_node)
            prev_degree = self._node_degree_dict[adj_node]
            if prev_degree != current_degree:
                self._buckets[prev_degree].remove(adj_node)
                self._buckets[current_degree].insert(0, adj_node)
                self._node_degree_dict[adj_node] = current_degree

    def _remove_node(self, node_idx):
        node_degree = self._get_degree(node_idx)
        self._buckets[node_degree].remove(node_idx)
        for adj_node in self._adj_dict[node_idx]:
            self._adj_dict[adj_node].remove(node_idx)

    def _insert_edge(self, node_x, node_y, adj_cost_matrix):
        self._layout_transform_matrix_dict[(node_x, node_y)] = adj_cost_matrix
        self._layout_transform_matrix_dict[(node_y, node_x)] = []
        for i in range(len(adj_cost_matrix[0])):
            self._layout_transform_matrix_dict[(node_y, node_x)].append([])
            for j in range(len(adj_cost_matrix)):
                self._layout_transform_matrix_dict[(node_y, node_x)][i] \
                    .append(adj_cost_matrix[j][i])

        self._adj_dict[node_x].append(node_y)
        self._adj_dict[node_y].append(node_x)

    def _backward_insert_node(self, node_idx):
        for adj_node in self._adj_dict[node_idx]:
            self._adj_dict[adj_node].append(node_idx)

    def _RI_reduction(self, node_idx):
        adj_node = self._adj_dict[node_idx][0]
        ltf_matrix = self._layout_transform_matrix_dict[(adj_node, node_idx)]
        for i, cost_vec in enumerate(ltf_matrix):
            min_cost = INVALID_LAYOUT_TIME
            for j in range(len(cost_vec)):
                min_cost = min(min_cost, cost_vec[j] + self._sch_cost_dict[node_idx][j])
            self._sch_cost_dict[adj_node][i] += min_cost
        self._remove_node(node_idx)
        self._reorder_adj_nodes(node_idx)
        self._stack.append(node_idx)

    def _RII_reduction(self, node_idx):
        adj_node_x, adj_node_y = self._adj_dict[node_idx]
        ltf_matrix_x = self._layout_transform_matrix_dict[(adj_node_x, node_idx)]
        ltf_matrix_y = self._layout_transform_matrix_dict[(adj_node_y, node_idx)]
        delta_matrix = [[] for _ in range(len(ltf_matrix_x))]
        for i, cost_vec_x in enumerate(ltf_matrix_x):
            for j, cost_vec_y in enumerate(ltf_matrix_y):
                min_cost = INVALID_LAYOUT_TIME
                for k in range(len(self._sch_cost_dict[node_idx])):
                    min_cost = min(min_cost, cost_vec_x[k] + cost_vec_y[k]
                                   + self._sch_cost_dict[node_idx][k])
                delta_matrix[i].append(min_cost)

        if adj_node_x == adj_node_y:
            for i in range(len(delta_matrix)):
                self._sch_cost_dict[adj_node_x][i] += delta_matrix[i][i]
        elif adj_node_x in self._adj_dict[adj_node_y]:
            for i in range(len(delta_matrix)):
                for j in range(len(delta_matrix[i])):
                    self._layout_transform_matrix_dict[(adj_node_x, adj_node_y)][i][j] \
                        += delta_matrix[i][j]
                    self._layout_transform_matrix_dict[(adj_node_y, adj_node_x)][j][i] \
                        += delta_matrix[i][j]
        else:
            self._insert_edge(adj_node_x, adj_node_y, delta_matrix)

        self._remove_node(node_idx)
        self._reorder_adj_nodes(node_idx)
        self._stack.append(node_idx)

    def _RN_reduction(self, node_idx):
        min_cost = INVALID_LAYOUT_TIME
        sch_idx = -1

        for i, sch_cost in enumerate(self._sch_cost_dict[node_idx]):
            current_cost = sch_cost
            for adj_node in self._adj_dict[node_idx]:
                ltf_matrix = self._layout_transform_matrix_dict[(node_idx, adj_node)]
                adj_sch_cost = list(self._sch_cost_dict[adj_node])
                for j, ltf_cost in enumerate(ltf_matrix[i]):
                    adj_sch_cost[j] += ltf_cost
                current_cost += min(adj_sch_cost)
            if current_cost < min_cost:
                min_cost = current_cost
                sch_idx = i

        if sch_idx < 0:
            raise RuntimeError("Can't find a soltuion for node %d when "
                               "applying RN reduction" % node_idx)
        self._optimal_sch_dict[node_idx] = sch_idx
        self._is_optimal = False
        self._min_cost += self._sch_cost_dict[node_idx][sch_idx]

        for adj_node in self._adj_dict[node_idx]:
            ltf_matrix = self._layout_transform_matrix_dict[(node_idx, adj_node)]
            for i, ltf_cost in enumerate(ltf_matrix[sch_idx]):
                self._sch_cost_dict[adj_node][i] += ltf_cost

        self._remove_node(node_idx)
        self._reorder_adj_nodes(node_idx)
        self._stack.append(node_idx)

    def _forward(self):
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
        # Solve nodes left in the forward graph
        for node_idx in self._buckets[0]:
            sch_costs = self._sch_cost_dict[node_idx]
            min_cost = min(sch_costs)
            self._optimal_sch_dict[node_idx] = sch_costs.index(min_cost)
            self._min_cost += min_cost

        # Solve nodes with one or two degrees
        for node_idx in reversed(self._stack):
            self._backward_insert_node(node_idx)
            if node_idx not in self._optimal_sch_dict:
                sch_costs = list(self._sch_cost_dict[node_idx])
                for adj_node in self._adj_dict[node_idx]:
                    adj_optimal_idx = self._optimal_sch_dict[adj_node]
                    for i in range(len(sch_costs)):
                        sch_costs[i] += \
                            self._layout_transform_matrix_dict \
                                [(node_idx, adj_node)][i][adj_optimal_idx]
                min_cost = min(sch_costs)
                self._optimal_sch_dict[node_idx] = sch_costs.index(min_cost)

    def run(self):
        self._logger.info("Start to run PBQP algorithm...")
        # Define virtual schedule lists and layout transformaton matrices
        # for multi-input nodes.
        input_names = self._input_shapes.keys()
        for key, val in self._in_nodes_dict.items():
            target_input_idx = -1
            target_input_pos = -1
            if has_multiple_inputs(self._node_list, key, input_names):
                for i, item in enumerate(val):
                    if not is_input_node(self._node_list, item):
                        target_input_idx = item
                        target_input_pos = i
                        break
                self._layout_transform_matrix_dict[(target_input_idx, key)] = []
                layout_matrix = self._layout_transform_matrix_dict[(target_input_idx, key)]
                for j in range(len(self._sch_dict[target_input_idx])):
                    layout_matrix.append([])
                    for k in range(len(self._sch_dict[target_input_idx])):
                        layout_matrix[j].append(0 if j == k else INVALID_LAYOUT_TIME)

                for j in range(target_input_pos + 1, len(val)):
                    input_idx = val[j]
                    if is_input_node(self._node_list, input_idx):
                        continue
                    self._layout_transform_matrix_dict[(input_idx, key)] = \
                        self._layout_transform_matrix_dict[(input_idx, target_input_idx)]
                    del self._layout_transform_matrix_dict[(input_idx, target_input_idx)]

        # Create reverse layout transformation matrices
        for idx_pair, ltf_matrix in self._layout_transform_matrix_dict.items():
            reverse_key = (idx_pair[1], idx_pair[0])
            reverse_matrix = [[] for _ in range(len(ltf_matrix[0]))]
            for i in range(len(ltf_matrix)):
                for j in range(len(ltf_matrix[i])):
                    reverse_matrix[j].append(ltf_matrix[i][j])
            self._layout_transform_matrix_dict[reverse_key] = reverse_matrix

        self._forward()
        self._backward()
        is_optimal = "optimal" if self._is_optimal else "sub-optimal"
        self._logger.info("Finished PBQPExecutor run. Got %s solution." % is_optimal)
