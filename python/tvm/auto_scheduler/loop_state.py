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
# pylint: disable=unused-import

"""
The definition of the "state" in search.

Each LoopState corresponds to a schedule for its ComputeDAG.
A LoopState consists of: 1. a current loop structure; 2. a list of transformation steps used to
construct the loop structure.
The loop structure keeps a preview of how the schedule will finally look like after lowering the
current state (e.g. number of iterators, the extent of each iterator, the compute_at locations ...).
During the schedule search process, the loop structure can provide search policy with necessary
information on how to manipulate the current state.
The transform history is a sequence of `TransformStep` which will finally be mapped to TVM schedule
primitives. The steps can also be used for the serialization of a state.

The LoopState can be seen as a lightweight loop structure IR specifically for schedule search.
We don't use the existing TVM IR but to extend a new structure on it is because:
1. We want fast incremental change to the loop structures. The search policy needs to get the
immediate loop structures update rather than after TVM lowering;
2. We want serializable transform history for replay, backtracking, and mutation;
3. We may create some macro schedule primitives that represent the combination of several
TVM schedule primitives.

When the search is complete, we will lower the state to TVM IR with TVM's schedule primitives.
Since we share a lot of common objects during search, the transformation is implemented in
copy on write style. All objects are immutable, which is similar to TVM IR.
"""

import tvm._ffi
from tvm.te.tensor import Operation, Tensor
from tvm.runtime import Object
from . import _ffi_api


@tvm._ffi.register_object("auto_scheduler.Iterator")
class Iterator(Object):
    """ A loop iterator structure. """


@tvm._ffi.register_object("auto_scheduler.Stage")
class Stage(Object):
    """ A stage in the compute declaration. Similar to tvm.te.schedule.Stage. """


@tvm._ffi.register_object("auto_scheduler.State")
class StateObject(Object):
    """ The internal State object """
    def __eq__(self, other):
        return _ffi_api.StateEqual(self, other)


class State:
    """
    A state in the search process. It consists of the current loop structure
    and a list of transformation steps used to construct it.

    Each State corresponds to a specific schedule for its ComputeDAG.

    Parameters
    ----------
    state_object : StateObject
        The StateObject corresponding to C++ internal State object.
    dag : ComputeDAG
        The original ComputeDAG of this State.

    Notes
    -----
    This is a wrapper class of StateObject to deal with copy-on-write property
    """
    def __init__(self, state_object, dag):
        self.state_object = state_object
        self.compute_dag = dag

        self.stage_id_map = {}    # A dict maps operation to stage id
        self._update_stage_id_map()

    @property
    def stages(self):
        """
        Returns
        -------
        stages : List[Stage]
        """
        return self.state_object.stages

    @property
    def stage_ops(self):
        """
        Returns
        -------
        ops: List[Operation]
        """
        return [stage.op for stage in self.stages]

    def reorder(self, stage, order):
        """ Schedule primitive corresponds to te.reorder.

        Parameters
        ----------
        stage : Union[int, Operation, Tensor]
            The Stage to be reordered, can be a Stage order index, Stage operation or stage
            output tensor.
        order : List[Iterator]
            Iterators in the expected order.
        """
        stage_id = self._resolve_stage_id(stage)

        self.state_object = _ffi_api.StateReorder(self.state_object, stage_id, order)

    def split(self, stage, iterator, lengths, inner_to_outer=True):
        """ Schedule primitive corresponds to te.split.

        This API supports multiple split factors. (e.g. with 2 split factors, the original iterator
        will be split to 3 parts, use `inner_to_outer` to control the split order)

        Parameters
        ----------
        stage : Union[int, Operation, Tensor]
            The Stage to be split, can be a Stage order index, Stage operation or stage
            output tensor.
        iterator : Iterator
            The iterator to be split.
        lengths: List[int]
            The multiple split factors. Can be None to be filled by search policy.
        inner_to_outer: boolean = True
            Whether the factor go from inner to outer, or from outer to inner.

        Returns
        -------
        res_its : List[Iterator]
            The splitted new Iterators
        """
        stage_id = self._resolve_stage_id(stage)

        self.state_object, res = _ffi_api.StateSplit(self.state_object, stage_id, iterator, lengths,
                                                     inner_to_outer)
        return res

    def fuse(self, stage, iters):
        """ Schedule primitive corresponds to te.fuse.

        Parameters
        ----------
        stage : Union[int, Operation, Tensor]
            The Stage to be fused, can be a Stage order index, Stage operation or stage
            output tensor.
        iters : List[Iterator]
            The iterators to be fused

        Returns
        -------
        res_it : Iterator
            The fused Iterator
        """
        stage_id = self._resolve_stage_id(stage)

        self.state_object, res = _ffi_api.StateFuse(self.state_object, stage_id, iters)
        return res

    def copy(self):
        """ Do deep copy of this State. """
        state = State(self.state_object, self.compute_dag)
        state.stage_id_map = self.stage_id_map.copy()
        return state

    def _resolve_stage_id(self, stage_id):
        if isinstance(stage_id, Operation):
            return self.stage_id_map[stage_id]
        if isinstance(stage_id, Tensor):
            return self.stage_id_map[stage_id.op]
        if isinstance(stage_id, int):
            return stage_id
        raise ValueError("Invalid stage: " + stage_id +
                         " . Expect to be a int, Operation or Tensor")

    def _update_stage_id_map(self):
        for index, stage in enumerate(self.stages):
            self.stage_id_map[stage.op] = index

    def __getitem__(self, key):
        if isinstance(key, Tensor):
            key = key.op
        if isinstance(key, Operation):
            return self.stages[self.stage_id_map[key]]
        raise ValueError("Invalid item: " + key +
                         " . Expect to be a Operation or Tensor")

    def __str__(self):
        return str(self.state_object)

    def __eq__(self, other):
        return _ffi_api.StateEqual(self.state_object, other.state_object)
