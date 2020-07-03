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

""" Computational graph and its analysis tools """

import hashlib

import tvm._ffi
from tvm.runtime import Object
from tvm.te import PlaceholderOp, ComputeOp

from .loop_state import State, StateObject
from .utils import get_const_tuple
from .workload_registry import workload_key_to_tensors

from . import _ffi_api


@tvm._ffi.register_object("ansor.ComputeDAG")
class ComputeDAG(Object):
    """
    The Ansor computational graph and related program analyses.

    We convert a compute declaration described by `tvm.compute` (could be a single operator or a
    subgraph) to a ComputeDAG. It keeps the input/output tensors of the target compute declaration,
    a list of all related operations in topo order as well as a set of analyses over each operation
    stage (e.g. the total float operation count, consumer/producer relations of each operation
    stage, whether a operation stage should be tiled/compute inlined ...). These analyses can
    help the search policy to do some specific decisions during schedule search process.

    ComputeDAG is also responsible for the interaction between Ansor LoopState and TVM schedule
    (e.g. applying the LoopState transform steps to TVM schedule, providing LoopState with extra
    information get from TVM schedule ...).

    Parameters
    ----------
    compute : Union[List[Tensor], str]
        `Tensor`s or workload key for a compute declaration.
    """
    def __init__(self, compute):
        if isinstance(compute, str):
            compute = workload_key_to_tensors(compute)
        elif isinstance(compute, list):
            for item in compute:
                if not isinstance(item, tvm.te.Tensor):
                    raise ValueError("The input of ComputeDAG should be a list of Tensor")
        else:
            raise ValueError("Invalid compute: " + compute +
                             " . `ComputeDAG` expects a string or list of Tensor")
        self.__init_handle_by_constructor__(_ffi_api.ComputeDAG, compute)

    def get_init_state(self):
        """ Get the init state of this ComputeDAG.

        Returns
        -------
        state : State
            The initial State without any transform steps.
        """
        return State(self.init_state, self)

    def apply_steps_from_state(self, state):
        """
        Apply the history transform steps of a State to TVM schedule.

        Parameters
        ----------
        state : Union[State, StateObject]
            The target state to be applied to TVM schedule.

        Returns
        -------
            A `te.schedule` and the target `te.Tensor`s to be used in `tvm.lower` or `tvm.build`
        """
        state_obj = state if isinstance(state, StateObject) else state.state_object
        return _ffi_api.ComputeDAGApplyStepsFromState(self, state_obj)

    def print_python_code_from_state(self, state):
        """
        Print transform steps in the history of a State as TVM's python schedule primitive.

        Parameters
        ----------
        state : Union[State, StateObject]
            The target state to be applied to TVM schedule.

        Returns
        -------
        str : Str
            The Python schedule code.
        """
        state_obj = state if isinstance(state, StateObject) else state.state_object
        return _ffi_api.ComputeDAGPrintPythonCodeFromState(self, state_obj)

    def infer_bound_from_state(self, state):
        """
        Infer and fill the bound of all iterators of a state using TVM schedule.

        State api supports to define a split step with its split factor to be a blank placeholder,
        so sometimes we may get a State will incomplete iterator extent information.
        And another situation is after some steps (for exp. compute_at), it may be hard to track
        the extent change of all iterators.

        We perform infer bound using TVM schedule and fill the State with those information. After
        applying this methods, the State is guaranteed to have complete interator extent
        information.

        Parameters
        ----------
        state : Union[State, StateObject]
            The target state to be applied to TVM schedule.

        Returns
        -------
        state : State
            The State with complete bound information.
        """
        state_obj = state if isinstance(state, StateObject) else state.state_object
        return State(_ffi_api.ComputeDAGInferBoundFromState(self, state_obj), self)

    def __hash__(self):
        # TODO(merrymercy): Implement this more carefully and move this to c++ as a member function
        # of ComputeDAG
        str_key = ''
        for op in self.ops:
            t = op.output(0)
            if isinstance(op, PlaceholderOp):
                str_key += 'placeholder,'
                str_key += str(get_const_tuple(t.shape)) + ','
                str_key += t.dtype + ';'
            elif isinstance(op, ComputeOp):
                str_key += str(t.op.body) + ','
                str_key += str(get_const_tuple(t.shape)) + ','
                str_key += t.dtype + ';'
            else:
                raise ValueError("Invalid op: " + op)

        str_key = str_key.encode(encoding='utf-8')
        return hashlib.md5(str_key).hexdigest()
