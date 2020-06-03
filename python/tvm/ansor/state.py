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
""" ... """

import tvm._ffi
from tvm.runtime import Object

from . import _ffi_api


@tvm._ffi.register_object("ansor.Iterator")
class Iterator(Object):
    pass


@tvm._ffi.register_object("ansor.Stage")
class Stage(Object):

    def iterator(self, index):
        return _ffi_api.StageGetIterator(self, index)

    def iterators(self):
        return _ffi_api.StageGetIterators(self)


@tvm._ffi.register_object("ansor.State")
class State(Object):

    def stage(self, index):
        """
        Parameters
        ----------
        index : Int

        Returns
        -------
        stage : Stage
        """
        return _ffi_api.StateGetStage(self, index)

    def transform_steps_size(self):
        """ Return the size of transform_steps
        """
        return _ffi_api.StateGetTransformStepsSize(self)

    def reorder(self, stage_id, order):
        """
        Parameters
        ----------
        stage_id : Int
            The index of target stage
        order : List[Iterator]
            Iterators in expected order

        Returns
        -------
        state : State
            The updated state
        """
        state = _ffi_api.StateReorder(self, stage_id, order)
        return state

    def split(self, stage_id, it, lengths, inner_to_outer=True):
        """
        Parameters
        ----------
        stage_id : Int
            The index of target stage
        it : Iterator
            The target Iterator
        lengths: List[Int]
            The split factor
        inner_to_outer: Bool
            True to use `factor` for split from inner to outer,
            False to use `nparts` for split from outer to inner

        Returns
        -------
        state : State
            The updated state
        """
        state = _ffi_api.StateSplit(self, stage_id, it, lengths,
                                    inner_to_outer)
        return state

    def follow_split(self, stage_id, it, src_step_id, n_split):
        """
        Parameters
        ----------
        stage_id : Int
            The index of target stage
        it : Iterator
            The target Iterator
        src_step_id : Int
            The index of target step that this split follows
        n_split : Int
            Indecate how many level needs to be split out

        Returns
        -------
        state : State
            The updated state
        """
        state = _ffi_api.StateFollowSplit(self, stage_id, it, src_step_id,
                                          n_split)
        return state

    def follow_fused_split(self, stage_id, it, src_step_ids, level,
                           factor_or_nparts):
        """
        Parameters
        ----------
        stage_id : Int
            The index of target stage
        it : Iterator
            The target Iterator
        src_step_ids : List[Int]
            The indexes of target step that this split follows
        level : Int
        factor_or_nparts : Bool
            True to use `factor` for split from inner to outer,
            False to use `nparts` for split from outer to inner

        Returns
        -------
        state : State
            The updated state
        """
        state = _ffi_api.StateFollowFusedSplit(self, stage_id, it, src_step_ids,
                                               level, factor_or_nparts)
        return state

    def fuse(self, stage_id, iters):
        """
        Parameters
        ----------
        stage_id : Int
            The index of target stage
        iters : List[Iterator]
            The target Iterators to be fused

        Returns
        -------
        state : State
            The updated state
        """
        state = _ffi_api.StateFuse(self, stage_id, iters)
        return state

    def vectorize(self, stage_id, it):
        """
        Parameters
        ----------
        stage_id : Int
            The index of target stage
        it : Iterator
            The target Iterator to be vectorized

        Returns
        -------
        state : State
            The updated state
        """
        state = _ffi_api.StateVectorize(self, stage_id, it)
        return state

    def parallel(self, stage_id, it):
        """
        Parameters
        ----------
        stage_id : Int
            The index of target stage
        it : Iterator
            The target Iterator to be paralleled

        Returns
        -------
        state : State
            The updated state
        """
        state = _ffi_api.StateParallel(self, stage_id, it)
        return state

    def unroll(self, stage_id, it, max_unroll=-1):
        """
        Parameters
        ----------
        stage_id : Int
            The index of target stage
        it : Iterator
            The target Iterator to be unrolled
        max_unroll : Int

        Returns
        -------
        state : State
            The updated state
        """
        state = _ffi_api.StateUnroll(self, stage_id, it, max_unroll)
        return state

    def bind_thread(self, stage_id, it, thread_type):
        """
        Parameters
        ----------
        stage_id : Int
            The index of target stage
        it : Iterator
            The target Iterator to be vectorized
        thread_type : ...
            Supported type: kVThread, kBlockX, kThreadX, kThreadY

        Returns
        -------
        state : State
            The updated state
        """
        state = _ffi_api.StateBindThread(self, stage_id, it, thread_type)
        return state

    def compute_at(self, stage_id, target_stage_id, target_iter):
        """
        Parameters
        ----------
        stage_id : Int
            The index of target stage
        target_stage_id : Int
            The index of compute at target stage
        target_iter : Iterator
            The target Iterator to be compute at

        Returns
        -------
        state : State
            The updated state
        """
        return _ffi_api.StateComputeAt(self, stage_id, target_stage_id,
                                       target_iter)

    def compute_root(self, stage_id):
        """
        Parameters
        ----------
        stage_id : Int
            The index of target stage

        Returns
        -------
        state : State
            The updated state
        """
        return _ffi_api.StateComputeRoot(self, stage_id)

    def compute_inline(self, stage_id):
        """
        Parameters
        ----------
        stage_id : Int
            The index of target stage

        Returns
        -------
        state : State
            The updated state
        """
        return _ffi_api.StateComputeInline(self, stage_id)

    def pack_for_vec(self, stage_id, target_iter, vec_size):
        """
        Parameters
        ----------
        stage_id : Int
            The index of target stage
        target_iter : Iterator
            The target Iterator
        vec_size : Int

        Returns
        -------
        state : State
            The updated state
        """
        return _ffi_api.StatePackForVec(self, stage_id, target_iter, vec_size)

    def cache_read(self, stage_id, scope_name, reader_stage_ids, task_dag):
        """
        Parameters
        ----------
        stage_id : Int
            The index of target stage
        scope_name : Str
        reader_stage_ids : List[Int]
        task_dag : ComputeDAG

        Returns
        -------
        state : State
            The updated state
        """
        state = _ffi_api.StateCacheRead(self, stage_id, scope_name,
                                        reader_stage_ids, task_dag)
        return state

    def cache_write(self, stage_id, scope_name, task_dag):
        """
        Parameters
        ----------
        stage_id : Int
            The index of target stage
        scope_name : Str
        task_dag : ComputeDAG

        Returns
        -------
        state : State
            The updated state
        """
        state = _ffi_api.StateCacheWrite(self, stage_id, scope_name, task_dag)
        return state

    def pragma(self, stage_id, it, pragma_type):
        """
        Parameters
        ----------
        stage_id : Int
            The index of target stage
        it : Iterator
            The target Iterator
        pragma_type : Str

        Returns
        -------
        state : State
            The updated state
        """
        return _ffi_api.StatePragma(self, stage_id, it, pragma_type)

    def rfactor(self, stage_id, it, factor_iter_id, task_dag):
        """
        Parameters
        ----------
        stage_id : Int
            The index of target stage
        it : Iterator
        factor_iter_id : Int
        task_dag : ComputeDAG

        Returns
        -------
        state : State
            The updated state
        """
        state = _ffi_api.StateRfactor(self, stage_id, it, factor_iter_id,
                                      task_dag)
        return state

    def storage_align(self, stage_id, it, factor, offset):
        """
        Parameters
        ----------
        stage_id : Int
            The index of target stage
        it : Iterator
        factor : Int
        offset : Int

        Returns
        -------
        state : State
            The updated state
        """
        return _ffi_api.StateStorageAlign(self, stage_id, it, factor, offset)
