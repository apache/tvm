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
2. We want serializable history for replay and backtracking
3. We may create some Macro schedule primitives

After search is done, we will lower this IR to TVM IR with TVM schedule primitives.
Because we share a lot common objects during search,  the transformation is
implemented in copy on write style.  All objects are immutable, which is
similar to TVM IR.
"""

import tvm._ffi
from tvm.runtime import Object
from . import _ffi_api


@tvm._ffi.register_object("ansor.Iterator")
class Iterator(Object):
    """A for loop iterator"""
    pass


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
    def __init__(self, state_object):
        self.state_object = state_object

        self.stages_cache = None

    def clear_cache(self):
        self.stages_cache = None

    def copy(self):
        return State(self.state_object)

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

    def transform_steps_size(self):
        """ Return the size of transform_steps
        """
        return _ffi_api.StateGetTransformStepsSize(self.state_object)

    def reorder(self, stage_id, order):
        """
        Parameters
        ----------
        stage_id : Int
            The index of the stage to reorder
        order : List[Iterator]
            Iterators in the expected order
        """
        self.state_object = _ffi_api.StateReorder(self.state_object, stage_id, order)
        self.clear_cache()

    def split(self, stage_id, it, lengths, inner_to_outer=True):
        """
        Parameters
        ----------
        stage_id : Int
            The index of the stage to split
        it : Iterator
            The iterator to split
        lengths: List[Int]
            The split factors
        inner_to_outer: Bool
            True to use `factor` to split from inner to outer,
            False to use `nparts` to split from outer to inner

        Returns
        -------
        res_its : List[Iterator]
            The splitted new Iterators
        """
        self.state_object, res = _ffi_api.StateSplit(self.state_object, stage_id, it, lengths,
                                                     inner_to_outer)
        self.clear_cache()
        return res

    def follow_split(self, stage_id, it, src_step_id, n_split):
        """
        Parameters
        ----------
        stage_id : Int
            The index of the stage to split
        it : Iterator
            The iterator to split
        src_step_id : Int
            The index of the split step to follow in the history
        n_split : Int
            The number of split level

        Returns
        -------
        res_its : List[Iterator]
            The splitted new Iterators
        """
        self.state_object, res = _ffi_api.StateFollowSplit(self.state_object, stage_id, it,
                                                           src_step_id, n_split)
        self.clear_cache()
        return res

    def follow_fused_split(self, stage_id, it, src_step_ids, level,
                           factor_or_nparts):
        """
        Parameters
        ----------
        stage_id : Int
            The index of the stage to split
        it : Iterator
            The iterator to split
        src_step_ids : List[Int]
            The indices of the split steps to follow in the history
        level : Int
            Use the length in this split level
        factor_or_nparts : Bool
            True to use `factor` for split from inner to outer,
            False to use `nparts` for split from outer to inner

        Returns
        -------
        res_its : List[Iterator]
            The splitted new Iterators
        """
        self.state_object, res = _ffi_api.StateFollowFusedSplit(self.state_object, stage_id, it,
                                                                src_step_ids, level,
                                                                factor_or_nparts)
        self.clear_cache()
        return res

    def fuse(self, stage_id, iters):
        """
        Parameters
        ----------
        stage_id : Int
            The index of the stage to fuse
        iters : List[Iterator]
            The iterators to be fused

        Returns
        -------
        res_it : Iterator
            The fused Iterator
        """
        self.state_object, res = _ffi_api.StateFuse(self.state_object, stage_id, iters)
        self.clear_cache()
        return res

    def vectorize(self, stage_id, it):
        """
        Parameters
        ----------
        stage_id : Int
            The index of the stage to vectorize
        it : Iterator
            The iterator to be vectorized

        Returns
        -------
        res_it : Iterator
            The vectorized Iterator
        """
        self.state_object, res = _ffi_api.StateVectorize(self.state_object, stage_id, it)
        self.clear_cache()
        return res

    def parallel(self, stage_id, it):
        """
        Parameters
        ----------
        stage_id : Int
            The index of the stage to parallel
        it : Iterator
            The iterator to be parallelized

        Returns
        -------
        res_it : Iterator
            The parallelized Iterator
        """
        self.state_object, res = _ffi_api.StateParallel(self.state_object, stage_id, it)
        self.clear_cache()
        return res

    def unroll(self, stage_id, it, max_unroll=-1):
        """
        Parameters
        ----------
        stage_id : Int
            The index of the stage to unroll
        it : Iterator
            The iterator to be unrolled
        max_unroll: Int
            The maximum length of the iterator that can be unrolled

        Returns
        -------
        res_it : Iterator
            The unrolled Iterator
        """
        self.state_object, res = _ffi_api.StateUnroll(self.state_object, stage_id, it, max_unroll)
        self.clear_cache()
        return res

    def bind_thread(self, stage_id, it, thread_name):
        """
        Parameters
        ----------
        stage_id : Int
            The index of the stage to bind
        it : Iterator
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

        self.state_object, res = _ffi_api.StateBindThread(self.state_object, stage_id, it, thread_id)
        self.clear_cache()
        return res

    def compute_at(self, stage_id, target_stage_id, target_iter):
        """
        Parameters
        ----------
        stage_id : Int
            The index of source stage
        target_stage_id : Int
            The index of the target stage of compute_at
        target_iter : Iterator
            The target Iterator of compute_at
        """
        self.state_object = _ffi_api.StateComputeAt(self.state_object, stage_id,
                                                     target_stage_id, target_iter)
        self.clear_cache()

    def compute_root(self, stage_id):
        """
        Parameters
        ----------
        stage_id : Int
            The index of the stage to compute root
        """
        self.state_object = _ffi_api.StateComputeRoot(self.state_object, stage_id)
        self.clear_cache()

    def compute_inline(self, stage_id):
        """
        Parameters
        ----------
        stage_id : Int
            The index of the stage to compute inline
        """
        self.state_object = _ffi_api.StateComputeInline(self.state_object, stage_id)
        self.clear_cache()

    def cache_read(self, stage_id, scope_name, reader_stage_ids, task_dag):
        """
        Parameters
        ----------
        stage_id : Int
            The index of the stage to do cache_read
        scope_name : Str
        reader_stage_ids : List[Int]
        task_dag : ComputeDAG

        Returns
        -------
        new_stage_id : Int
            The added staged id
        """
        self.state_object, new_stage_id = _ffi_api.StateCacheRead(self.state_object, stage_id,
                                                                  scope_name, reader_stage_ids,
                                                                  task_dag)
        self.clear_cache()
        return int(new_stage_id)

    def cache_write(self, stage_id, scope_name, task_dag):
        """
        Parameters
        ----------
        stage_id : Int
            The index of the stage to do cache read
        scope_name : Str
        task_dag : ComputeDAG

        Returns
        -------
        new_stage_id : Int
            The added staged id
        """
        self.state_object, new_stage_id = _ffi_api.StateCacheWrite(self.state_object, stage_id,
                                                                   scope_name, task_dag)
        self.clear_cache()
        return int(new_stage_id)

    def pragma(self, stage_id, it, pragma_type):
        """
        Parameters
        ----------
        stage_id : Int
            The index of the stage to add pragma
        it : Iterator
            The iterator to add pragma
        pragma_type : Str
        """
        self.state_object = _ffi_api.StatePragma(self.state_object, stage_id, it, pragma_type)
        self.clear_cache()

    def rfactor(self, stage_id, it, factor_iter_id, task_dag):
        """
        Parameters
        ----------
        stage_id : Int
            The index of the stage to do reduction factor
        it : Iterator
        factor_iter_id : Int
        task_dag : ComputeDAG

        Returns
        -------
        new_stage_id : Int
            The added staged id
        """
        self.state_object, new_stage_id = _ffi_api.StateRfactor(self.state_object, stage_id, it,
                                                                factor_iter_id, task_dag)
        self.clear_cache()
        return int(new_stage_id)

    def storage_align(self, stage_id, it, factor, offset):
        """
        Parameters
        ----------
        stage_id : Int
            The index of the stage to do storage align
        it : Iterator
        factor : Int
        offset : Int
        """
        self.state_object = _ffi_api.StateStorageAlign(self.state_object, stage_id, it, factor, offset)
        self.clear_cache()

    def tensorize(self, stage_id, it, ti_func_name):
        """ The `ti_func_name` corresponds to a global registered funcion
        that returns a TensorIntrin

        Parameters
        ----------
        stage_id : Int
            The index of the stage to do storage align
        it : Iterator
            The target iterator
        ti_func_name : Str
            Tensorize intrinsic function name

        Returns
        -------
        res_it : Iterator
            The tensorized Iterator
        """
        self.state_object, res = _ffi_api.StateTensorize(self.state_object,
                                                         stage_id, it,
                                                         ti_func_name)
        self.clear_cache()
        return res

    def __str__(self):
        return str(self.state_object)

    def __eq__(self, other):
        return _ffi_api.StateEqual(self.state_object, other.state_object)
