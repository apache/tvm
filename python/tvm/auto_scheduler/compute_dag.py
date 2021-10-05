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
import json

import tvm._ffi
from tvm.runtime import Object
from tvm.runtime._ffi_node_api import LoadJSON, SaveJSON

from . import _ffi_api
from .loop_state import State, StateObject
from .utils import get_const_tuple
from .workload_registry import workload_key_to_tensors


class LayoutRewriteOption:
    """
    Options for applying layout rewrite.

    The NO_REWRITE and INSERT_TRANSFORM_STAGE are expected to be used when tuning a standalone op,
    and the REWRITE_FOR_PRE_TRANSFORMED is expected to be used when tuning ops inside a network.
    """

    # Do not perform layout rewrite
    NO_REWRITE = 0
    # Insert layout transformation stages for input placeholders in the compute DAG
    INSERT_TRANSFORM_STAGE = 1
    # Do not insert layout transformation stages and assume the input placeholders
    # are pre-transformed.
    # Note: The lowered function with this option does not accept the origial input shapes,
    # so this option must be used along with `AutoSchedulerLayoutRewrite` pass in Relay.
    REWRITE_FOR_PRE_TRANSFORMED = 2

    @staticmethod
    def get_target_default(target, in_relay_integration=False):
        """Get the default layout rewrite option for the specified target.
        Currently we only enable layout rewrite for cpu / mali backend for now

        Parameters
        ----------
        target: tvm.target.Target
            The compilation target.
        in_relay_integration: bool
            If this check is ask for relay integration.

        Returns
        -------
        layout_rewrite_option: LayoutRewriteOption
            The default layout rewrite option for the specified target.
        """
        layout_rewrite_option = LayoutRewriteOption.NO_REWRITE
        if target.kind.name == "llvm" or (
            "device" in target.attrs and target.attrs["device"] == "mali"
        ):
            layout_rewrite_option = (
                LayoutRewriteOption.REWRITE_FOR_PRE_TRANSFORMED
                if in_relay_integration
                else LayoutRewriteOption.INSERT_TRANSFORM_STAGE
            )

        return layout_rewrite_option


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
    compute : Union[List[Tensor], str, tvm.te.Schedule]
        Input/output tensors or workload key for a compute declaration.
    """

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
        self.__init_handle_by_constructor__(_ffi_api.ComputeDAG, compute, sche)

    def get_init_state(self):
        """Get the init state of this ComputeDAG.

        Returns
        -------
        state : State
            The initial State without any transform steps.
        """
        return State(self.init_state, self)

    def apply_steps_from_state(self, state, layout_rewrite=LayoutRewriteOption.NO_REWRITE):
        """
        Apply the history transform steps from a State to get a TVM schedule.

        Parameters
        ----------
        state : Union[State, StateObject]
            The state from which we get transform steps.

        layout_rewrite: LayoutRewriteOption = NoRewrite
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

    def workload_key(self):
        """Return the workload key of this compute DAG.
        The workload key is a JSON string from a tuple of (hash of DAG, tensor shapes...)

        Returns
        -------
        key: str
            The workload key of this compute DAG
        """
        str_dag = _ffi_api.ComputeDAGPrintDAG(self, True)
        hash_func = tvm._ffi.get_global_func(
            "auto_scheduler.compute_dag.hash_func", allow_missing=True
        )

        if hash_func is None:
            str_dag = str_dag.encode("utf-8")
            hash_key = hashlib.md5(str_dag).hexdigest()
        else:
            hash_key = hash_func(str_dag)

        io_shapes = []
        for tensor in self.tensors:
            io_shapes.append(get_const_tuple(tensor.shape))
        return json.dumps([hash_key] + io_shapes)

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
        return {"tensors": SaveJSON(self.tensors)}

    def __setstate__(self, state):
        # Since we always use tensors to recover the ComputeDAG, we do not support
        # (de)serialization of the ComputeDAG constructed by a schedule.
        self.__init_handle_by_constructor__(_ffi_api.ComputeDAG, LoadJSON(state["tensors"]), None)


def get_shape_from_rewritten_layout(rewritten_layout, axis_names):
    """Get the orginal shape from a rewritten layout string.

    Parameters
    ----------
    rewritten_layout: str
        The layout after rewrite
    axis_names: List[str]
        Specify the order of axes by names

    Returns
    -------
    shape: List[PrimExpr]
        The original shape
    """
    return _ffi_api.GetShapeFromRewrittenLayout(rewritten_layout, axis_names)
