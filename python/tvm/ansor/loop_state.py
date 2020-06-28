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
The definition of the "state" in search. A state consists a current loop structure
and the transform history to reach its current loop structure.
To enable flexible manipulation of the loop structure, we implemented a lightweight
loop structure IR (Intermediate Representation) specifically for search.

Basically this is a simplified TVM IR with schedule primitives.
We don't use the existing TVM IR because
1. We want fast incremental change to the loop structures
2. We want serializable transformation history for replay, backtracking, and mutation
3. We may create some new macro schedule primitives

After the search is done, we will lower this IR to TVM IR with TVM's schedule primitives.
Because we share a lot common objects during search,  the transformation is
implemented in copy on write style.  All objects are immutable, which is
similar to TVM IR.
"""

import tvm._ffi
from tvm.te.tensor import Operation, Tensor
from tvm.runtime import Object
from . import _ffi_api


@tvm._ffi.register_object("ansor.Iterator")
class Iterator(Object):
    """A for loop iterator"""


@tvm._ffi.register_object("ansor.Stage")
class Stage(Object):
    """A stage in the compute declaration. Similar to tvm.te.schedule.Stage"""

    @property
    def iters(self):
        """
        Returns
        -------
        iters : List[Iterator]
        """
        if not hasattr(self, "iterators_cache"):
            setattr(self, "iterators_cache", _ffi_api.StageGetIterators(self))
        return getattr(self, "iterators_cache")


@tvm._ffi.register_object("ansor.State")
class StateObject(Object):
    """The internal State object """
    def __eq__(self, other):
        return _ffi_api.StateEqual(self, other)


class State:
    """
    A state in the search process. It consists of the current loop structure
    and the history steps to reach this state.

    Notes
    -----
    This is a wrapper class of StateObject to deal with copy-on-write property
    """
    def __init__(self, state_object, dag):
        self.state_object = state_object
        self.compute_dag = dag

        self.stages_cache = None  # A list to cache all stages
        self.stage_id_map = {}    # A dict maps operation to stage id
        self._update_stage_id_map()

    @property
    def stages(self):
        """
        Returns
        -------
        stages : List[Stage]
        """
        if not self.stages_cache:
            self.stages_cache = _ffi_api.StateGetStages(self.state_object)
        return self.stages_cache

    @property
    def stage_ops(self):
        """
        Returns
        -------
        ops: List[Operation]
        """
        if not self.stages_cache:
            self.stages_cache = _ffi_api.StateGetStages(self.state_object)
        return [stage.op for stage in self.stages_cache]

    def transform_steps_size(self):
        """ Return the size of transform_steps
        """
        return _ffi_api.StateGetTransformStepsSize(self.state_object)

    def reorder(self, stage_id, order):
        """
        Parameters
        ----------
        stage_id : Union[int, Operation, Tensor]
            The index of the stage to reorder
        order : List[Iterator]
            Iterators in the expected order
        """
        stage_id = self._resolve_stage_id(stage_id)

        self.state_object = _ffi_api.StateReorder(self.state_object, stage_id, order)
        self._clear_cache()

    def split(self, stage_id, iterator, lengths, inner_to_outer=True):
        """
        Parameters
        ----------
        stage_id : Union[int, Operation, Tensor]
            The index of the stage to split
        iterator : Iterator
            The iterator to split
        lengths: List[int]
            The split factors
        inner_to_outer: bool
            True to use `factor` to split from inner to outer,
            False to use `nparts` to split from outer to inner

        Returns
        -------
        res_its : List[Iterator]
            The splitted new Iterators
        """
        stage_id = self._resolve_stage_id(stage_id)

        self.state_object, res = _ffi_api.StateSplit(self.state_object, stage_id, iterator, lengths,
                                                     inner_to_outer)
        self._clear_cache()
        return res

    def fuse(self, stage_id, iters):
        """
        Parameters
        ----------
        stage_id : Union[int, Operation, Tensor]
            The index of the stage to fuse
        iters : List[Iterator]
            The iterators to be fused

        Returns
        -------
        res_it : Iterator
            The fused Iterator
        """
        stage_id = self._resolve_stage_id(stage_id)

        self.state_object, res = _ffi_api.StateFuse(self.state_object, stage_id, iters)
        self._clear_cache()
        return res

    def _resolve_stage_id(self, stage_id):
        if isinstance(stage_id, Operation):
            return self.stage_id_map[stage_id]
        elif isinstance(stage_id, tvm.te.Tensor):
            return self.stage_id_map[stage_id.op]
        elif isinstance(stage_id, int):
            return stage_id
        else:
            raise ValueError("Invalid stage_id")

    def _update_stage_id_map(self):
        if not self.stages_cache:
            self.stages_cache = _ffi_api.StateGetStages(self.state_object)
        for index, stage in enumerate(self.stages_cache):
            self.stage_id_map[stage.op] = index

    def _clear_cache(self):
        self.stages_cache = None

    def copy(self):
        state = State(self.state_object, self.compute_dag)
        state.stage_id_map = self.stage_id_map.copy()
        return state

    def __getitem__(self, key):
        if not self.stages_cache:
            self.stages_cache = _ffi_api.StateGetStages(self.state_object)
        if isinstance(key, Tensor):
            key = key.op
        if isinstance(key, Operation):
            return self.stages_cache[self.stage_id_map[key]]
        raise ValueError("Item must be Tensor")

    def __str__(self):
        return str(self.state_object)

    def __eq__(self, other):
        return _ffi_api.StateEqual(self.state_object, other.state_object)
