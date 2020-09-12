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
# pylint: disable=too-many-instance-attributes,too-many-branches,too-many-statements,too-many-arguments,too-many-locals,invalid-name
"""Stage class for dynamic programming tuner"""
import numpy as np

from .utils import is_boundary_node


class DPStage(object):
    """Class to represent node in Markov decision process. A stage has states
    to represent different schedules of the current node. Since in this problem
    the action is the schedule selected for current node, action can be fully
    represented by states. No extra attribute needs for action.

    In most cases, instance of this class should be created through DPTuner.
    """

    def __init__(
        self,
        idx,
        input_shapes,
        node_list,
        counted_nodes_set,
        layout_transform_interlayer_cost,
        stage_dict,
        in_nodes_dict,
        out_nodes_dict,
        dep_dict,
        target_ops,
        dtype="float32",
    ):
        """Initialize a stage and create all states.

        Parameters
        ----------
        idx : int
            Index for current node.

        input_shapes : dict of string to tuple of int
            Input shapes for current graph.

        node_list : list of dict
            List of all nodes for current graph.

        counted_nodes_set : set of int
            Global set recording whether the execution time of a node has been counted.

        layout_transform_interlayer_cost : dict of tuple to list
            Dictionary maps node index pair to layout transformation time between them.

        stage_dict : dict of int to Stage
            Global dictionary for all stages mapping node index to stage.

        in_nodes_dict : dict of int to list of int
            Dictionary maps node index to corresponding input node index.

        out_nodes_dict : dict of int to list of int
            Dictionary maps node index to corresponding output node index.

        dep_dict : dict of int to set of int
            Dictionary maps node index to dependent node index.

        target_ops : list of str
            Target operators

        dtype : str, optional
            Data type.
        """
        self._global_input_shapes = input_shapes
        self._global_input_names = input_shapes.keys()
        self._global_node_list = node_list
        self._global_counted_nodes_set = counted_nodes_set
        self._global_layout_transform_interlayer_cost = layout_transform_interlayer_cost
        self._global_stage_dict = stage_dict
        self._global_in_nodes_dict = in_nodes_dict
        self._global_out_nodes_dict = out_nodes_dict
        self._global_dep_dict = dep_dict

        self._idx = idx
        self._node_entry = self._global_node_list[idx]
        self._target_ops = target_ops
        self._wkl = self._node_entry["workloads"][0]
        self._record_list = self._node_entry["record_candidates"]
        self._dep = []
        self._dtype = dtype
        self._states = None
        self._full_states = None
        self._full_states_idx = None
        self._create_states()

    def _create_states(self):
        """Create states."""
        node = self._global_node_list[self._idx]
        if node["op"] in self._target_ops:
            self._create_op_states()
        else:
            self._create_multi_inputs_states()

    def _create_op_states(self):
        """State creation routine for nodes with target_op."""
        input_idx = self._global_in_nodes_dict[self._idx][0]
        input_node_entry = self._global_node_list[input_idx]
        if is_boundary_node(input_node_entry, self._global_input_names):
            self._full_states = np.array([record[1].costs[0] for record in self._record_list])
            self._states = self._full_states
        else:
            input_stage = self._global_stage_dict[input_idx]
            input_dep = input_stage.dep
            input_states = input_stage.states
            input_flatten_states = input_states.flatten()
            input_record_list = input_node_entry["record_candidates"]
            num_schedules = len(self._record_list)
            num_input_schedules = len(input_record_list)
            num_input_states = input_flatten_states.shape[0]

            full_states_shape = tuple(
                [num_schedules, num_input_schedules]
                + [
                    len(self._global_node_list[dep_idx]["record_candidates"])
                    for dep_idx in input_dep
                ]
            )
            self._full_states = np.zeros(full_states_shape).flatten().astype("float32")
            self._full_states_idx = [self._idx, input_idx] + input_dep
            dep_multiplier = 1
            for i in range(2, len(full_states_shape)):
                dep_multiplier *= full_states_shape[i]
            input_node_time_counted = input_idx in self._global_counted_nodes_set

            for i in range(num_schedules):
                current_sch_time = float(self._record_list[i][1].costs[0])
                for j in range(num_input_states):
                    input_sch_idx = j // dep_multiplier
                    layout_transform_time = self._global_layout_transform_interlayer_cost[
                        (input_idx, self._idx)
                    ][input_sch_idx][i]

                    if input_node_time_counted:
                        total_time = current_sch_time + layout_transform_time
                    else:
                        total_time = (
                            current_sch_time + layout_transform_time + input_flatten_states[j]
                        )
                    current_state_idx = i * num_input_states + j
                    self._full_states[current_state_idx] = total_time

            if not input_node_time_counted:
                self._global_counted_nodes_set.add(input_idx)
            self._full_states = self._full_states.reshape(full_states_shape)

            # If out degree of input node is 1, we can remove the dimension of input node,
            # since the states of input node will not be needed any more. Otherwise, input
            # node should become a dependency.
            if len(self._global_out_nodes_dict[input_idx]) == 1:
                self._states = np.amin(self._full_states, axis=1)
                self._dep = list(input_dep)
            else:
                self._states = self._full_states
                self._dep = [
                    input_idx,
                ] + input_dep

        # Update global dependency dictionary.
        # This is to monitor the dependency states to decide
        # when a dependency can be eliminated, so that total
        # number of states can be largely reduced.
        for dep_idx in self._dep:
            self._global_dep_dict[dep_idx].remove(self._idx)
            for child in self._global_out_nodes_dict[self._idx]:
                self._global_dep_dict[dep_idx].add(child)
        if len(self._global_out_nodes_dict[self._idx]) > 1:
            self._global_dep_dict[self._idx] = set()
            for child in self._global_out_nodes_dict[self._idx]:
                self._global_dep_dict[self._idx].add(child)

    def _create_multi_inputs_states(self):
        """State creation routine for multi_input operator

        In tvm, layout transformation for an elemwise-like follow the rule which
        all input operators transform their layouts to the leftmost input operator
        layout. For example:
                            elemwise-sum
                            |    |    |
                            |    |    |
                           op0  op1  op2
        In this block, the possible layout transformations are: op1 -> op0 and op2 -> op0.
        In graph tuning, a 3-D array with shape (k0, k1, k2) can represent the layout
        transformations between these three nodes. It is also possible some earlier states
        belong to other nodes(We name them as dependency) are required for dynamic programming.
        The final states array for this elemwise-sum can be with shape (e0, k0, k1, e1, k2).
        To iterate through all states, we first align the shape of op0, op1 and op2 to be
        (e0, k0, k1, e1, k2) by broadcasting the original states. We also record the axis of
        each input node in the states array, together with the multiplier. For example,
        the axis index for op0 is 1, and multiplier is k1 * e1 * k2. If current iterating index
        in the flatten array is i, the index of op0 can be computed as:
        i % (k0 * k1 * e1 * k2) // (k1 * e1 * k2).
        """
        full_input_node_list = list(self._global_in_nodes_dict[self._idx])
        input_index_list = []
        # Remove input and ruled_out nodes
        for input_idx in full_input_node_list:
            input_node = self._global_node_list[input_idx]
            if not is_boundary_node(input_node, self._global_input_names):
                input_index_list.append(input_idx)

        # Generate new states
        states_list, aligned_node_list = DPStage.align_states(
            input_index_list, self._global_stage_dict, self._global_node_list
        )
        target_node_idx, target_major_axis, target_multiplier, target_states = states_list[0]
        aligned_shape = target_states.shape
        self._full_states = np.zeros(aligned_shape).astype("float32").flatten()
        self._full_states_idx = list(aligned_node_list)
        num_states = self._full_states.shape[0]
        node_time_counted = [item[0] in self._global_counted_nodes_set for item in states_list]
        target_states = target_states.flatten()
        src_states_list = [states_list[i][3].flatten() for i in range(1, len(states_list))]

        for i in range(num_states):
            target_sch_idx = (
                i % (target_multiplier * aligned_shape[target_major_axis])
            ) // target_multiplier
            if node_time_counted[0]:
                new_state = 0
            else:
                new_state = target_states[i]

            for j in range(1, len(states_list)):
                src_states = src_states_list[j - 1]
                src_node_idx, src_major_axis, src_multiplier, _ = states_list[j]
                src_sch_idx = (
                    i % (src_multiplier * aligned_shape[src_major_axis])
                ) // src_multiplier
                layout_transform_time = self._global_layout_transform_interlayer_cost[
                    (src_node_idx, target_node_idx)
                ][src_sch_idx][target_sch_idx]

                if node_time_counted[j]:
                    new_state += layout_transform_time
                else:
                    new_state += layout_transform_time + src_states[i]
                self._full_states[i] = new_state

        for i, node_counted in enumerate(node_time_counted):
            if not node_counted:
                self._global_counted_nodes_set.add(states_list[i][0])
        self._full_states = self._full_states.reshape(aligned_shape)

        # Remove dependency to reduce states
        reduced_states = np.array(self._full_states)
        reduced_states_transpose = [states_list[0][1]]
        reduced_states_dep_list = []
        self._dep = []
        for i in range(len(reduced_states.shape)):
            if i != states_list[0][1]:
                reduced_states_transpose.append(i)
                reduced_states_dep_list.append(aligned_node_list[i])
        reduced_states = np.transpose(reduced_states, reduced_states_transpose)
        shift = 0
        for i, dep in enumerate(reduced_states_dep_list):
            if dep not in self._global_dep_dict or len(self._global_dep_dict[dep]) == 1:
                self._global_dep_dict.pop(dep, None)
                reduced_states = np.amin(reduced_states, axis=i + 1 - shift)
                shift += 1
            else:
                self._dep.append(dep)
        self._states = reduced_states

        # Update dependency
        for dep in self._dep:
            self._global_dep_dict[dep].remove(self._idx)
            for child in self._global_out_nodes_dict[self._idx]:
                self._global_dep_dict[dep].add(child)
        if len(self._global_out_nodes_dict[self._idx]) > 1:
            self._global_dep_dict[self._idx] = set()
            for child in self._global_out_nodes_dict[self._idx]:
                self._global_dep_dict[self._idx].add(child)

    @property
    def dep(self):
        """Get dependency list."""
        return self._dep

    @property
    def states(self):
        """Get states."""
        return self._states

    @property
    def full_states(self):
        """Get complete states."""
        return self._full_states

    @property
    def full_states_idx(self):
        """Get node index of complete states."""
        return self._full_states_idx

    @staticmethod
    def align_states(input_index_list, stage_dict, node_list):
        """Align all input node states shapes to be the same and transpose/reshape properly.

        This is used in creating multi_input operator states.

        Parameters
        ----------
        input_index_list : list of int
            List of input node index.

        stage_dict : dict of int to Stage
            Global dictionary of node index to stage.

        node_list : list of dict
            List of all nodes for current graph.

        Returns
        -------
        states_list : list of tuple
            List of aligned states.

        aligned_node_list : list in int
            List of node index for aligned states.
        """
        aligned_node_list = list(input_index_list)
        states_list = []
        for input_idx in input_index_list:
            input_node_stage = stage_dict[input_idx]
            for dep_idx in input_node_stage.dep:
                if dep_idx not in aligned_node_list:
                    aligned_node_list.append(dep_idx)
        aligned_shape = []
        for idx in aligned_node_list:
            aligned_shape.append(len(node_list[idx]["record_candidates"]))
        for input_idx in input_index_list:
            input_node_stage = stage_dict[input_idx]
            input_node_shape_idx_list = [input_idx] + input_node_stage.dep
            transpose_idx_list = []
            reshape_list = []
            major_axis = -1
            for i, idx in enumerate(aligned_node_list):
                if input_idx == idx:
                    major_axis = i
                if idx in input_node_shape_idx_list:
                    transpose_idx_list.append(idx)
                    reshape_list.append(aligned_shape[i])
                else:
                    reshape_list.append(1)
            transpose_list = [input_node_shape_idx_list.index(idx) for idx in transpose_idx_list]
            input_node_states = np.transpose(input_node_stage.states, tuple(transpose_list))
            input_node_states = np.reshape(input_node_states, tuple(reshape_list))
            input_node_states = np.broadcast_to(input_node_states, aligned_shape)
            multiplier = 1
            for i in range(major_axis + 1, len(aligned_shape)):
                multiplier *= aligned_shape[i]
            states_list.append((input_idx, major_axis, multiplier, input_node_states))
        return states_list, aligned_node_list
