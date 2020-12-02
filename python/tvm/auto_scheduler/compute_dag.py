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

""" The auto-scheduler's computational graph and related program analyses. """

import hashlib

import tvm._ffi
from tvm.runtime import Object
from tvm.runtime._ffi_node_api import LoadJSON, SaveJSON
from tvm.te import ComputeOp, PlaceholderOp

from . import _ffi_api
from .loop_state import State, StateObject
from .utils import get_const_tuple
from .workload_registry import workload_key_to_tensors


@tvm._ffi.register_object("auto_scheduler.ComputeDAG")
class ComputeDAG(Object):
    """
    The auto-scheduler's computational graph and related program analyses.

    We convert a compute declaration described by `tvm.compute` (could be a single operator or a
    subgraph) to a ComputeDAG. It keeps the input/output tensors, all operations in the DAG, and
    some static analysis results for the DAG (e.g. the total float operation count,
    consumer/producer relations of operations, whether an operation stage should
    be tiled/compute inlined).
    These analyses can help the search policy to make decisions during the search.
    ComputeDAG is also responsible for the interaction between auto-scheduler's `LoopState` and
    TVM schedule (e.g. applying the `LoopState` transform steps to a TVM schedule, providing
    `LoopState` with extra information got from TVM schedule).

    Parameters
    ----------
    compute : Union[List[Tensor], str, Schedule]
        Input/output tensors or workload key for a compute declaration.
    """

    # Layout Rewrite Options
    NoRewrite = 0
    InsertTransformStage = 1
    RewriteForPreTransformed = 2

    def __init__(self, compute_or_sche):
        if isinstance(compute_or_sche, str):
            compute = workload_key_to_tensors(compute_or_sche)
            sche = None
        elif isinstance(compute_or_sche, (list, tvm.ir.container.Array)):
            for item in compute_or_sche:
                if not isinstance(item, tvm.te.Tensor):
                    raise ValueError(
                        "The input of ComputeDAG should be a list of Tensor, but got %s"
                        % type(item)
                    )
            compute = compute_or_sche
            sche = None
        elif isinstance(compute_or_sche, tvm.te.Schedule):
            compute = None
            sche = compute_or_sche
        else:
            raise ValueError(
                "Invalid compute type: %s. ComputeDAG expects string, list of Tensor, or Schedule"
                % type(compute_or_sche)
            )
        self.compute = compute
        self.sche = sche
        self.__init_handle_by_constructor__(_ffi_api.ComputeDAG, compute, sche)

    def get_init_state(self):
        """Get the init state of this ComputeDAG.

        Returns
        -------
        state : State
            The initial State without any transform steps.
        """
        return State(self.init_state, self)

    def apply_steps_from_state(self, state, layout_rewrite=NoRewrite):
        """
        Apply the history transform steps from a State to get a TVM schedule.

        Parameters
        ----------
        state : Union[State, StateObject]
            The state from which we get transform steps.

        layout_rewrite: Bool
            Rewrite the layout of placeholders specified by "layout_free_placeholders" attr
            to make it most friendly for the generated schedule to read from.

        Returns
        -------
            A `te.schedule` and the a list of `te.Tensor` to be used in `tvm.lower` or `tvm.build`.
        """
        state_obj = state if isinstance(state, StateObject) else state.state_object
        return _ffi_api.ComputeDAGApplyStepsFromState(self, state_obj, layout_rewrite)

    def print_python_code_from_state(self, state):
        """
        Print transform steps in the history of a State as TVM's python schedule code.

        This is used to print transformation steps for debugging.
        Use `apply_steps_from_state` if you want to get a schedule for code generation.

        Parameters
        ----------
        state : Union[State, StateObject]
            The state from which we get transform steps.

        Returns
        -------
        str : Str
            The Python schedule code.
        """
        state_obj = state if isinstance(state, StateObject) else state.state_object
        return _ffi_api.ComputeDAGPrintPythonCodeFromState(self, state_obj)

    def infer_bound_from_state(self, state):
        """
        Infer and fill the bound of all iterators of a state.

        The states may lose complete bound information after some transform steps
        (e.g., compute_at).
        We can call this function to infer and fill all the bound information.
        This function calls TVM InferBound pass internally to get the bound.
        The returned state of this function is guaranteed to have complete iterator extent
        information.

        Parameters
        ----------
        state : Union[State, StateObject]
            The state from which we get transform steps.

        Returns
        -------
        updated_state : State
            The State with complete bound information.
        """
        state_obj = state if isinstance(state, StateObject) else state.state_object
        updated_state = State(_ffi_api.ComputeDAGInferBoundFromState(self, state_obj), self)
        # Copy the stage_id_map from the original state to make sure the old indices are still
        # valid
        if isinstance(state, State):
            for k, v in state.stage_id_map.items():
                updated_state.stage_id_map[k] = v
        return updated_state

    def rewrite_layout_from_state(self, state):
        """
        Rewrite the layout of the DAG according to the history transform steps of a state.

        Parameters
        ----------
        state : Union[State, StateObject]
            The state from which we get transform steps.

        Returns
        -------
        updated_dag : ComputeDAG
            The compute dag with rewritten layout.
        """
        state_obj = state if isinstance(state, StateObject) else state.state_object
        return _ffi_api.ComputeDAGRewriteLayoutFromState(self, state_obj)

    def hash_key(self):
        """Return the hash key of this compute DAG.

        Returns
        -------
        key: str
            The hash key of this compute DAG
        """
        # TODO(merrymercy): Implement this more carefully and move this to c++ as a member function
        # of ComputeDAG
        str_key = ""
        for op in self.ops:
            t = op.output(0)
            if isinstance(op, PlaceholderOp):
                str_key += "placeholder,"
                str_key += str(get_const_tuple(t.shape)) + ","
                str_key += t.dtype + ";"
            elif isinstance(op, ComputeOp):
                str_key += str(t.op.body) + ","
                str_key += str(get_const_tuple(t.shape)) + ","
                str_key += t.dtype + ";"
            else:
                raise ValueError("Invalid op: " + op)

        str_key = str_key.encode(encoding="utf-8")
        return hashlib.md5(str_key).hexdigest()

    def __str__(self):
        # pretty print
        MAX_LINE_WIDTH = 256

        raw_lines = super().__str__().split("\n")
        lines = []
        for line in raw_lines:
            if len(line) > MAX_LINE_WIDTH:
                line = (
                    line[: MAX_LINE_WIDTH // 2] + " ..(OMITTED).. " + line[-MAX_LINE_WIDTH // 2 :]
                )
            lines.append(line)
        return "\n".join(lines)

    def __getstate__(self):
        return {"compute": SaveJSON(self.compute), "sche": SaveJSON(self.sche)}

    def __setstate__(self, state):
        self.compute = LoadJSON(state["compute"])  # pylint: disable=assignment-from-no-return
        self.sche = LoadJSON(state["sche"])  # pylint: disable=assignment-from-no-return
        self.__init_handle_by_constructor__(_ffi_api.ComputeDAG, self.compute, self.sche)
