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
from tvm import te
from .loop_state import State
from . import _ffi_api


class LayoutRewriteLevel(object):
    NO_REWRITE = 0           # No layout rewrite
    PLACEHOLDER_REWRITE = 1  # Only rewrite layout of placeholder in the compute dag
    COMPUTE_REWRITE = 2      # Only rewrite compute body for new layout in the compute dag
    BOTH_REWRITE = 3         # Rewrite both placeholder and compute body in the compute dag


@tvm._ffi.register_object("ansor.ComputeDAG")
class ComputeDAG(Object):
    """
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
        return State(_ffi_api.ComputeDAGGetInitState(self))

    def apply_steps_from_state(self, state, layout_rewrite_level=None):
        """
        Parameters
        ----------
        state : StateObject
        layout_rewrite_level : LayoutRewriteLevel(***)

        Returns
        -------
        sch : Schedule
        args : List[Tensor]
        """
        sch, args = _ffi_api.ComputeDAGApplyStepsFromState(self, state)
        return sch, args

    def print_python_code_from_state(self, state):
        """
        Parameters
        ----------
        state : StateObject

        Returns
        -------
        str : Str
        """
        return _ffi_api.ComputeDAGPrintPythonCodeFromState(self, state)

    def infer_bound_from_state(self, state):
        """
        Parameters
        ----------
        state : StateObject

        Returns
        -------
        state : StateObject
        """
        return _ffi_api.ComputeDAGInferBoundFromState(self, state)

def gen_schedule(state, bufs):
    if not state or not state.complete:
        return te.create_schedule([x.op for x in bufs])
    else:
        dag = ComputeDAG(bufs)
        # only update compute body, layout_rewrite_level = LayoutRewriteLevel.COMPUTE_REWRITE,
        # since kernel layout has already been rewritten in relay pass
        schedule, _ = dag.apply_steps_from_state(state, layout_rewrite_level=LayoutRewriteLevel.COMPUTE_REWRITE)
    return schedule
