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

import tvm._ffi
from tvm.runtime import Object
from .loop_state import State, StateObject
from . import _ffi_api


@tvm._ffi.register_object("ansor.ComputeDAG")
class ComputeDAG(Object):
    """
    Computation declaration graph

    Parameters
    ----------
    tensors : List[Tensor]
    """
    def __init__(self, tensors):
        self.__init_handle_by_constructor__(_ffi_api.ComputeDAG, tensors)

    def get_init_state(self):
        """ Get init state of this ComputeDAG

        Returns
        -------
        state : State
        """
        return State(_ffi_api.ComputeDAGGetInitState(self), self)

    def apply_steps_from_state(self, state, layout_rewrite_level=LayoutRewriteLevel.NO_REWRITE):
        """
        Apply transform steps according to the history of a state

        Parameters
        ----------
        state : StateObject
        layout_rewrite_level : LayoutRewriteLevel

        Returns
        -------
        sch : Schedule
        args : List[Tensor]
        """
        state_obj = state if isinstance(state, StateObject) else state.state_object
        return _ffi_api.ComputeDAGApplyStepsFromState(self, state_obj, layout_rewrite_level)

    def print_python_code_from_state(self, state):
        """
        Print transform steps in the history of a state as TVM's python schedule primitive

        Parameters
        ----------
        state : StateObject

        Returns
        -------
        str : Str
        """
        state_obj = state if isinstance(state, StateObject) else state.state_object
        return _ffi_api.ComputeDAGPrintPythonCodeFromState(self, state_obj)

    def infer_bound_from_state(self, state):
        """
        Infer bound for a state

        Parameters
        ----------
        state : StateObject

        Returns
        -------
        state : State
        """
        state_obj = state if isinstance(state, StateObject) else state.state_object
        return State(_ffi_api.ComputeDAGInferBoundFromState(self, state_obj), self)
