# pylint: disable=too-many-instance-attributes,too-many-branches,too-many-statements,too-many-arguments,too-many-locals
"""Stage class for dynamic programming tuner"""
import numpy as np

from .utils import is_input_node


class DPStage(object):
    """Class to represent node in Markov decision process. A stage has states
    to represent different schedules of the current node. Since in this problem
    the action is the schedule selected for current node, action can be fully
    represented by states. No extra attribute needs for action.

    In most cases, instance of this class should be created through DPTuner.
    """
    def __init__(self, idx, wkl_dict, sch_dict, input_shapes, node_list,
                 data_layout, counted_nodes_set, layout_time_matrix_dict,
                 stage_dict, in_nodes_dict, out_nodes_dict, dep_dict, target_op,
                 infer_layout_shape_func, dtype="float32"):
        """Initialize a stage and create all states.

        Parameters
        ----------
        idx : int
            Index for current node.

        wkl_dict : dict of str to namedtuple
            Workload dictionary maps node index to workload.

        sch_dict : dict of int to list of dict
            Schedule dictionary maps node index to schedule candidates. Each element
            in the value list is a dictionary which has "schedule" and "cost" entries.

        input_shapes : dict of string to tuple of int
            Input shapes for current graph.

        node_list : list of dict
            List of all nodes for current graph.

        data_layout : str
            Data layout for target operator.

        counted_nodes_set : set of int
            Global set recording whether the execution time of a node has been counted.

        layout_time_matrix_dict : dict of tuple to list
            Dictionary maps node index pair to layout transformation time between them.

        stage_dict : dict of int to Stage
            Global dictionary for all stages mapping node index to stage.

        in_nodes_dict : dict of int to list of int
            Dictionary maps node index to corresponding input node index.

        out_nodes_dict : dict of int to list of int
            Dictionary maps node index to corresponding output node index.

        dep_dict : dict of int to set of int
            Dictionary maps node index to dependent node index.

        target_op : str
            Operator name.

        infer_layout_shape_func : function
            Function to infer actual input and output shapes for layout
            transformation given a workload, current schedule and target schedule.

        dtype : str, optional
            Data type.
        """
        self._global_wkl_dict = wkl_dict
        self._global_sch_dict = sch_dict
        self._global_input_shapes = input_shapes
        self._global_input_names = input_shapes.keys()
        self._global_node_list = node_list
        self._global_counted_nodes_set = counted_nodes_set
        self._global_layout_time_matrix_dict = layout_time_matrix_dict
        self._global_stage_dict = stage_dict
        self._global_in_nodes_dict = in_nodes_dict
        self._global_out_nodes_dict = out_nodes_dict
        self._global_dep_dict = dep_dict

        self._idx = idx
        self._target_op = target_op
        self._data_layout = data_layout
        self._batch_size = list(self._global_input_shapes.values())[0][0]
        self._wkl = self._global_wkl_dict[self._idx]
        self._sch_list = self._global_sch_dict[self._idx]
        self._dep = []
        self._infer_layout_shape_func = infer_layout_shape_func
        self._dtype = dtype
        self._states = None
        self._full_states = None
        self._full_states_idx = None
        self._create_states()

    def _create_states(self):
        """Create states."""
        node = self._global_node_list[self._idx]
        if node["op"] == self._target_op:
            self._create_op_states()
        else:
            self._create_multi_inputs_states()

    def _create_op_states(self):
        """State creation routine for nodes with target_op."""
        input_idx = -1
        for index in self._global_in_nodes_dict[self._idx]:
            input_idx = index
            if not is_input_node(self._global_node_list[input_idx],
                                 self._global_input_names):
                break

        if is_input_node(self._global_node_list[input_idx],
                         self._global_input_names):
            self._full_states = np.array([sch["cost"] for sch in self._sch_list])
            self._states = self._full_states
        else:
            input_stage = self._global_stage_dict[input_idx]
            input_dep = input_stage.dep
            input_states = input_stage.states
            input_flatten_states = input_states.flatten()
            input_sch_list = self._global_sch_dict[input_idx]
            num_schedules = len(self._sch_list)
            num_input_schedules = len(input_sch_list)
            num_input_states = input_flatten_states.shape[0]

            full_states_shape = tuple([num_schedules, num_input_schedules] +
                                      [len(self._global_sch_dict[dep_idx])
                                       for dep_idx in input_dep])
            self._full_states = np.zeros(full_states_shape).flatten().astype("float32")
            self._full_states_idx = [self._idx, input_idx] + input_dep
            dep_multiplier = 1
            for i in range(2, len(full_states_shape)):
                dep_multiplier *= full_states_shape[i]
            input_node_time_counted = input_idx in self._global_counted_nodes_set

            for i in range(num_schedules):
                current_sch_time = float(self._sch_list[i]["cost"])
                for j in range(num_input_states):
                    input_sch_idx = j // dep_multiplier
                    layout_transform_time = \
                        self._global_layout_time_matrix_dict\
                            [(input_idx, self._idx)][input_sch_idx][i]

                    if input_node_time_counted:
                        total_time = current_sch_time + layout_transform_time
                    else:
                        total_time = \
                            current_sch_time + layout_transform_time + input_flatten_states[j]
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
                self._dep = [input_idx,] + input_dep

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
        """State creation routine for multi_input operator"""
        full_input_node_list = list(self._global_in_nodes_dict[self._idx])
        input_node_list = []
        # Remove input and parameter nodes
        for input_idx in full_input_node_list:
            if not is_input_node(self._global_node_list[input_idx],
                                 self._global_input_names):
                input_node_list.append(input_idx)

        # Generate new states
        states_list, aligned_node_list = DPStage.align_states(input_node_list,
                                                              self._global_stage_dict,
                                                              self._global_sch_dict)
        aligned_shape = states_list[0][3].shape
        self._full_states = np.zeros(aligned_shape).astype("float32").flatten()
        self._full_states_idx = list(aligned_node_list)
        num_states = self._full_states.shape[0]
        node_time_counted = [item[0] in self._global_counted_nodes_set for item in states_list]
        target_node_idx = states_list[0][0]
        target_states = states_list[0][3].flatten()
        target_multiplier = states_list[0][2]
        target_major_axis = states_list[0][1]
        src_states_list = [states_list[i][3].flatten() for i in range(1, len(states_list))]

        for i in range(num_states):
            target_sch_idx = (i % (target_multiplier *
                                   aligned_shape[target_major_axis])) // target_multiplier
            if node_time_counted[0]:
                new_state = 0
            else:
                new_state = target_states[i]
            for j in range(1, len(states_list)):
                src_node_idx = states_list[j][0]
                src_states = src_states_list[j - 1]
                src_multiplier = states_list[j][2]
                src_major_axis = states_list[j][1]
                src_sch_idx = (i % (src_multiplier *
                                    aligned_shape[src_major_axis])) // src_multiplier
                layout_transform_time = \
                    self._global_layout_time_matrix_dict\
                        [(src_node_idx, target_node_idx)][src_sch_idx][target_sch_idx]

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
                reduced_states = np.amin(reduced_states, axis=i+1-shift)
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

        # Update sch_dict
        leftmost_in_node = self._global_in_nodes_dict[self._idx][0]
        self._global_sch_dict[self._idx] = self._global_sch_dict[leftmost_in_node]

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
    def align_states(node_list, stage_dict, sch_dict):
        """Align all input node states shapes to be the same and transpose/reshape properly.

        This is used in creating multi_input operator states.

        Parameters
        ----------
        node_list : list of int
            List of input node index.

        stage_dict : dict of int to Stage
            Global dictionary of node index to stage.

        sch_dict : dict of int to list
            Schedule dictionary maps node index to schedule candidates. Each element
            in the value list is a dictionary which has "schedule" and "cost" entries.

        Returns
        -------
        states_list : list of tuple
            List of aligned states.

        aligned_node_list : list in int
            List of node index for aligned states.
        """
        aligned_node_list = list(node_list)
        states_list = []
        for input_idx in node_list:
            input_node_stage = stage_dict[input_idx]
            for dep_idx in input_node_stage.dep:
                if dep_idx not in aligned_node_list:
                    aligned_node_list.append(dep_idx)
        aligned_shape = tuple([len(sch_dict[idx]) for idx in aligned_node_list])
        for input_idx in node_list:
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
