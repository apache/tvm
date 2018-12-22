# pylint: disable=import-error,too-many-locals,too-many-statements,too-many-branches,arguments-differ,unused-variable

"""Dynamic programming tuner."""
import sys
import numpy as np

from .base_graph_tuner import BaseGraphTuner
from .dynamic_programming_stage import DPStage
from .utils import has_multiple_inputs, is_input_node

if sys.version_info[0] == 3:
    import queue
else:
    import Queue as queue

class DPTuner(BaseGraphTuner):
    """Tuner which uses dynamic programming to solve MDP problem.

    Note: currently dynamic programming is used to solve this MDP problem. However,
    this problem is intrinsically non-polynomial. DP can't apply for more complicated
    models, such as networks with many element-wise sum operators. In this case, a simple
    greedy method greedy_schedule can be used to generate suboptimal schedules.

    TODO  Analyse time/memory complexity of the algorithm and automatically switch to
    TODO  greedy method if DP is prohibitively expensive.
    """
    def __init__(self, *args, **kwargs):
        """Create a dynamic programming tuner.
        """
        super(DPTuner, self).__init__(*args, **kwargs)
        self._num_states = self._max_num_states = None

    def _check_num_states(self, num_states):
        self._num_states += num_states
        if self._max_num_states is not None:
            if self._num_states > self._max_num_states:
                raise RuntimeError("Too many states detected while running dynamic "
                                   "programming: got %d states but upper limit is %d." %
                                   (self._num_states, self._max_num_states))

    def _forward(self):
        """Forward pass in DP to generate states for all stages.
        """
        self._logger.info("Start forward pass...")
        input_names = self._input_shapes.keys()
        for node_idx, node in enumerate(self._node_list):
            if node["op"] == self._target_op or has_multiple_inputs(self._node_list, node_idx,
                                                                    input_names):
                stage = DPStage(idx=node_idx, target_op=self._target_op,
                                **self._global_data_dict)
                self._check_num_states(stage.full_states.size)
                self._stage_dict[node_idx] = stage
        self._logger.info("Finished forward pass.")

    def _backward(self):
        """Backward pass in DP to generate optimal solution.
        """
        self._logger.info("Start backward pass...")
        input_names = self._input_shapes.keys()
        optimal_sch_dict = {}
        # Pick optimal schedule for output nodes
        output_idx_list = []
        for key, val in self._out_nodes_dict.items():
            if not val:
                output_idx_list.append(key)
        states_list, aligned_node_list = DPStage.align_states(output_idx_list, self._stage_dict,
                                                              self._sch_dict)
        num_states = states_list[0][3].size
        self._check_num_states(num_states * len(output_idx_list))
        aligned_node_shape = states_list[0][3].shape
        min_time = 0
        min_pos = -1
        for states in states_list:
            min_time += np.amax(states[3])
        flatten_states_list = [current_states[3].flatten() for current_states in states_list]
        for i in range(num_states):
            current_time = 0
            for j, current_states in enumerate(states_list):
                current_time += flatten_states_list[j][i]
            if min_time > current_time:
                min_time = current_time
                min_pos = i
        for i, states in enumerate(states_list):
            current_major_axis = states[1]
            current_sch_idx = (min_pos % (states[2] *
                                          aligned_node_shape[current_major_axis])) // states[2]
            optimal_sch_dict[aligned_node_list[i]] = current_sch_idx
        # Pick optimal schedule for dependencies of output nodes
        for i in range(len(states_list), len(aligned_node_list)):
            multiplier = 1
            for j in range(i + 1, len(aligned_node_list)):
                multiplier *= aligned_node_shape[j]
            optimal_sch_dict[aligned_node_list[i]] = min_pos // multiplier % aligned_node_shape[i]

        # Backward pass to get optimal schedules for other nodes
        bfs_q = queue.Queue()
        visited = set()
        for out_idx in output_idx_list:
            bfs_q.put(out_idx)
        while not bfs_q.empty():
            node_idx = bfs_q.get()
            visited.add(node_idx)
            if is_input_node(self._node_list, node_idx):
                continue
            optimal_sch_idx = optimal_sch_dict[node_idx]
            full_states = self._stage_dict[node_idx].full_states
            if not has_multiple_inputs(self._node_list, node_idx, input_names):
                input_idx = self._in_nodes_dict[node_idx][0]
                if is_input_node(self._node_list, input_idx):
                    continue
                if input_idx not in visited:
                    bfs_q.put(input_idx)
                    if input_idx not in optimal_sch_dict:
                        dep_list = self._stage_dict[node_idx].dep
                        dep_idx = tuple([optimal_sch_dict[item] for item in dep_list])
                        tmp = np.argmin(full_states, axis=1)
                        optimal_input_sch_idx = tmp[(optimal_sch_idx,) + dep_idx]
                        optimal_sch_dict[input_idx] = optimal_input_sch_idx
            else:
                input_idx_list = self._in_nodes_dict[node_idx]
                optimal_sch_dict[input_idx_list[0]] = optimal_sch_idx
                full_states_idx = self._stage_dict[node_idx].full_states_idx
                tmp = full_states[optimal_sch_idx]
                new_states_idx, new_states_pos = [], []
                visited_states_idx, visited_states_pos = [], []
                for i in range(1, len(full_states_idx)):
                    if full_states_idx[i] in optimal_sch_dict:
                        visited_states_idx.append(full_states_idx[i])
                        visited_states_pos.append(i - 1)
                    else:
                        new_states_idx.append(full_states_idx[i])
                        new_states_pos.append(i - 1)
                if visited_states_idx:
                    tmp = np.transpose(tmp, tuple(visited_states_pos + new_states_pos))
                    tmp = tmp[tuple([optimal_sch_dict[idx] for idx in visited_states_idx])]
                min_pos = np.argmin(tmp)
                multiplier = 1
                for i in range(len(new_states_idx)):
                    multiplier *= full_states.shape[new_states_pos[i] + 1]
                for pos, idx in zip(new_states_pos, new_states_idx):
                    multiplier //= full_states.shape[pos + 1]
                    optimal_sch_dict[idx] = min_pos // multiplier
                    min_pos %= multiplier
                for input_idx in input_idx_list:
                    if input_idx not in visited:
                        bfs_q.put(input_idx)

        self._optimal_sch_dict = optimal_sch_dict
        for node_idx, _ in self._sch_dict.items():
            if self._node_list[node_idx]["op"] != self._target_op:
                continue
            if node_idx not in self._optimal_sch_dict:
                self._optimal_sch_dict[node_idx] = 0
        self._logger.info("Finished backward pass...")

    def run(self, max_num_states=None):
        """Run dynamic programming solver.

        Parameters
        ----------
        max_num_states : int, optional
            Maximum number of total states during DP.
            If states number exceeds this argument, an exception will be thrown.
            This argument prevents out of memory issue or too long execution
            time of DP.

        """
        self._num_states = 0
        self._max_num_states = max_num_states
        self._logger.info("Start to run dynamic programming algorithm...")
        self._forward()
        self._backward()
        self._logger.info("Finished DPExecutor run.")
