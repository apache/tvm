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

    def follow_split(self, stage_id, iterator, src_step_id, n_split):
        """
        Parameters
        ----------
        stage_id : Union[int, Operation, Tensor]
            The index of the stage to split
        iterator : Iterator
            The iterator to split
        src_step_id : int
            The index of the split step to follow in the history
        n_split : int
            The number of split level

        Returns
        -------
        res_its : List[Iterator]
            The splitted new Iterators
        """
        stage_id = self._resolve_stage_id(stage_id)

        self.state_object, res = _ffi_api.StateFollowSplit(self.state_object, stage_id, iterator,
                                                           src_step_id, n_split)
        self._clear_cache()
        return res

    def follow_fused_split(self, stage_id, iterator, src_step_ids, level,
                           factor_or_nparts):
        """
        Parameters
        ----------
        stage_id : Union[int, Operation, Tensor]
            The index of the stage to split
        iterator : Iterator
            The iterator to split
        src_step_ids : List[int]
            The indices of the split steps to follow in the history
        level : int
            Use the length in this split level
        factor_or_nparts : bool
            True to use `factor` for split from inner to outer,
            False to use `nparts` for split from outer to inner

        Returns
        -------
        res_its : List[Iterator]
            The splitted new Iterators
        """
        stage_id = self._resolve_stage_id(stage_id)

        self.state_object, res = _ffi_api.StateFollowFusedSplit(self.state_object, stage_id,
                                                                iterator, src_step_ids, level,
                                                                factor_or_nparts)
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

    def vectorize(self, stage_id, iterator):
        """
        Parameters
        ----------
        stage_id : Union[int, Operation, Tensor]
            The index of the stage to vectorize
        iterator : Iterator
            The iterator to be vectorized

        Returns
        -------
        res_it : Iterator
            The vectorized Iterator
        """
        stage_id = self._resolve_stage_id(stage_id)

        self.state_object, res = _ffi_api.StateVectorize(self.state_object, stage_id, iterator)
        self._clear_cache()
        return res

    def parallel(self, stage_id, iterator):
        """
        Parameters
        ----------
        stage_id : Union[int, Operation, Tensor]
            The index of the stage to parallel
        iterator : Iterator
            The iterator to be parallelized

        Returns
        -------
        res_it : Iterator
            The parallelized Iterator
        """
        stage_id = self._resolve_stage_id(stage_id)

        self.state_object, res = _ffi_api.StateParallel(self.state_object, stage_id, iterator)
        self._clear_cache()
        return res

    def unroll(self, stage_id, iterator, max_unroll=-1):
        """
        Parameters
        ----------
        stage_id : Union[int, Operation, Tensor]
            The index of the stage to unroll
        iterator : Iterator
            The iterator to be unrolled
        max_unroll: int
            The maximum length of the iterator that can be unrolled

        Returns
        -------
        res_it : Iterator
            The unrolled Iterator
        """
        stage_id = self._resolve_stage_id(stage_id)

        self.state_object, res = _ffi_api.StateUnroll(self.state_object, stage_id, iterator,
                                                      max_unroll)
        self._clear_cache()
        return res

    def bind_thread(self, stage_id, iterator, thread_name):
        """
        Parameters
        ----------
        stage_id : Union[int, Operation, Tensor]
            The index of the stage to bind
        iterator : Iterator
            The iterator to be bound
        thread_name : str
            The name of the thread (e.g. "blockIdx.x", "threadIdx.y", "vthread")

        Returns
        -------
        res_it : Iterator
            The bound Iterator
        """
        trans_table = {
            "vthread": 4,
            "blockIdx.x": 5,
            "threadIdx.x": 6,
            "blockIdx.y": 7,
            "threadIdx.y": 8,
        }
        thread_id = trans_table[thread_name]

        stage_id = self._resolve_stage_id(stage_id)

        self.state_object, res = _ffi_api.StateBindThread(self.state_object, stage_id, iterator,
                                                          thread_id)
        self._clear_cache()
        return res

    def compute_at(self, stage_id, target_stage_id, target_iter):
        """
        Parameters
        ----------
        stage_id : Union[int, Operation, Tensor]
            The index of source stage
        target_stage_id : Union[int, Operation, Tensor]
            The index of the target stage of compute_at
        target_iter : Iterator
            The target Iterator of compute_at
        """
        stage_id = self._resolve_stage_id(stage_id)
        target_stage_id = self._resolve_stage_id(target_stage_id)

        self.state_object = _ffi_api.StateComputeAt(self.state_object, stage_id,
                                                    target_stage_id, target_iter)
        self._clear_cache()

    def compute_root(self, stage_id):
        """
        Parameters
        ----------
        stage_id : Union[int, Operation, Tensor]
            The index of the stage to compute root
        """
        stage_id = self._resolve_stage_id(stage_id)

        self.state_object = _ffi_api.StateComputeRoot(self.state_object, stage_id)
        self._clear_cache()

    def compute_inline(self, stage_id):
        """
        Parameters
        ----------
        stage_id : Union[int, Operation, Tensor]
            The index of the stage to compute inline
        """
        stage_id = self._resolve_stage_id(stage_id)

        self.state_object = _ffi_api.StateComputeInline(self.state_object, stage_id)
        self._clear_cache()

    def cache_read(self, stage_id, scope_name, reader_stage_ids):
        """
        Parameters
        ----------
        stage_id : Union[int, Operation, Tensor]
            The index of the stage to do cache_read
        scope_name : str
        reader_stage_ids : List[int]

        Returns
        -------
        new_stage_id : int
            The added staged id
        """
        stage_id = self._resolve_stage_id(stage_id)

        if isinstance(reader_stage_ids, list):
            tmp_list = []
            for reader_stage_id in reader_stage_ids:
                tmp_list.append(self._resolve_stage_id(reader_stage_id))
            reader_stage_ids = tmp_list
        else:
            raise ValueError("reader_stage_ids must be list of Tensor or int")

        self.state_object, new_stage_id = _ffi_api.StateCacheRead(self.state_object, stage_id,
                                                                  scope_name, reader_stage_ids,
                                                                  self.compute_dag)
        return self._insert_new_stage(new_stage_id)

    def cache_write(self, stage_id, scope_name):
        """
        Parameters
        ----------
        stage_id : Union[int, Operation, Tensor]
            The index of the stage to do cache read
        scope_name : str

        Returns
        -------
        new_stage_id : int
            The added staged id
        """
        stage_id = self._resolve_stage_id(stage_id)

        self.state_object, new_stage_id = _ffi_api.StateCacheWrite(self.state_object, stage_id,
                                                                   scope_name, self.compute_dag)
        return self._insert_new_stage(new_stage_id)

    def pragma(self, stage_id, iterator, pragma_type):
        """
        Parameters
        ----------
        stage_id : Union[int, Operation, Tensor]
            The index of the stage to add pragma
        iterator : Iterator
            The iterator to add pragma
        pragma_type : str
        """
        stage_id = self._resolve_stage_id(stage_id)

        self.state_object = _ffi_api.StatePragma(self.state_object, stage_id, iterator,
                                                 pragma_type)
        self._clear_cache()

    def rfactor(self, stage_id, iterator, factor_iter_id):
        """
        Parameters
        ----------
        stage_id : Union[int, Operation, Tensor]
            The index of the stage to do reduction factor
        iterator : Iterator
        factor_iter_id : int

        Returns
        -------
        new_stage_id : int
            The added staged id
        """
        stage_id = self._resolve_stage_id(stage_id)

        self.state_object, new_stage_id = _ffi_api.StateRfactor(self.state_object, stage_id,
                                                                iterator, factor_iter_id,
                                                                self.compute_dag)
        return self._insert_new_stage(new_stage_id)

    def storage_align(self, stage_id, iterator, factor, offset):
        """
        Parameters
        ----------
        stage_id : Union[int, Operation, Tensor]
            The index of the stage to do storage align
        iterator : Iterator
        factor : int
        offset : int
        """
        stage_id = self._resolve_stage_id(stage_id)

        self.state_object = _ffi_api.StateStorageAlign(self.state_object, stage_id, iterator,
                                                       factor, offset)
        self._clear_cache()

    def tensorize(self, stage_id, iterator, ti_func_name):
        """ The `ti_func_name` corresponds to a global registered funcion
        that returns a Tensorintrin

        Parameters
        ----------
        stage_id : Union[int, Operation, Tensor]
            The index of the stage to do storage align
        iterator : Iterator
            The iterator to be tensorized
        ti_func_name : str
            Tensorize intrinsic function name

        Returns
        -------
        res_it : Iterator
            The tensorized Iterator
        """
        stage_id = self._resolve_stage_id(stage_id)

        self.state_object, res = _ffi_api.StateTensorize(self.state_object,
                                                         stage_id, iterator,
                                                         ti_func_name)
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

    def _insert_new_stage(self, new_stage_id):
        new_stage_id = int(new_stage_id)
        self.stages_cache = _ffi_api.StateGetStages(self.state_object)
        added_op = self.stages_cache[new_stage_id].op

        # Add a new stage will change all ops. But we still want to use the old ops to index stages,
        # So we keep updating them and do not remove the old ops.

        # Update stage_id_map for old ops, so we can still use the old ops to index stages.
        for key, value in self.stage_id_map.items():
            if value >= new_stage_id:
                self.stage_id_map[key] = value + 1
        self.stage_id_map[added_op] = new_stage_id

        # Update stage_id_map for new ops
        self._update_stage_id_map()

        return added_op

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
