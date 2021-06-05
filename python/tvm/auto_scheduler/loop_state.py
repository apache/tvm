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
The definition of the "state" in the search.
Each LoopState corresponds to a schedule for its ComputeDAG.
A LoopState consists of: 1. a current loop structure; 2. a list of transformation steps used to
construct the loop structure.
The loop structure keeps a preview of how the schedule will finally look like after lowering the
current state (e.g. number of iterators, the extent of each iterator, the compute_at locations
...).
During the schedule search process, the loop structure can provide search policy with necessary
information on how to manipulate the current state.
The transform history is a sequence of `TransformStep` which will finally be mapped to TVM
schedule primitives. The steps are also used for the serialization of a state.
The LoopState can be seen as a lightweight loop structure IR specifically for schedule search.
We don't use the existing TVM IR but to extend a new structure on it is because:
1. We want fast incremental change to the loop structures. The search policy needs to get the
immediate loop structures update rather than after TVM lowering;
2. We want serializable transform history for replay, backtracking, and mutation;
3. We may create some macro schedule primitives that represent the combination of several
TVM schedule primitives.
When the search is finished, we will lower the state to TVM IR with TVM's schedule primitives.
Since we share a lot of common objects during search, the transformation is implemented in
copy on write style. All objects are immutable, which is similar to TVM IR.
"""

import tvm._ffi
from tvm.te.tensor import Operation, Tensor
from tvm.runtime import Object
from . import _ffi_api


@tvm._ffi.register_object("auto_scheduler.Iterator")
class Iterator(Object):
    """A loop iterator structure."""


@tvm._ffi.register_object("auto_scheduler.Stage")
class Stage(Object):
    """A stage in the compute declaration. Similar to tvm.te.schedule.Stage."""

    # Static trans table for compute_at location
    # This is used to transform the compute_at location to C++ enum
    COMPUTE_AT_TRANS_TABLE = {"root": 0, "inlined": 1, "iter": 2}


@tvm._ffi.register_object("auto_scheduler.State")
class StateObject(Object):
    """The internal State object"""

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

    # Static trans table for thread bind and annotation
    # This is used to transform the annotation name to C++ enum
    ANNOTATION_TRANS_TABLE = {
        "none": 0,
        "unroll": 1,
        "vectorize": 2,
        "parallel": 3,
        "vthread": 4,
        "blockIdx.x": 5,
        "threadIdx.x": 6,
        "blockIdx.y": 7,
        "threadIdx.y": 8,
        "blockIdx.z": 9,
        "threadIdx.z": 10,
        "tensorize": 11,
    }

    def __init__(self, state_object, dag):
        self.state_object = state_object
        self.compute_dag = dag

        self.stage_id_map = {}  # A dict maps operation to stage id
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
    def transform_steps(self):
        """
        Returns
        -------
        transform_steps : List[transform_steps]
        """
        return self.state_object.transform_steps

    @property
    def stage_ops(self):
        """
        Returns
        -------
        ops: List[Operation]
        """
        return [stage.op for stage in self.stages]

    def bind(self, stage, iterator, thread_name):
        """Schedule primitive corresponding to `te.Stage.bind`.
        See also the `te.Stage` for more details.
        Parameters
        ----------
        stage : Union[int, Operation, Tensor]
            The Stage to be binded, which can be specified by the integer index, Operation,
            or output tensor of the stage.
        iterator : Iterator
            The iterator to be binded.
        thread_name : str
            The thread type to be binded. Candidates:
            - vthread
            - blockIdx.x
            - threadIdx.x
            - blockIdx.y
            - threadIdx.y
            - blockIdx.z
            - threadIdx.z
        Returns
        -------
        res_it : Iterator
            The binded Iterator.
        """
        if not thread_name in State.ANNOTATION_TRANS_TABLE.keys():
            raise ValueError("Invalid thread_name: ", thread_name)

        self.state_object, res = _ffi_api.StateBind(
            self.state_object,
            self._resolve_stage_id(stage),
            iterator,
            State.ANNOTATION_TRANS_TABLE[thread_name],
        )
        return res

    def parallel(self, stage, iterator):
        """Schedule primitive corresponding to `te.Stage.parallel`.
        See also the `te.Stage` for more details.
        Parameters
        ----------
        stage : Union[int, Operation, Tensor]
            The Stage to be paralleled, which can be specified by the integer index, Operation,
            or output tensor of the stage.
        iterator : Iterator
            The iterator to be paralleled.
        Returns
        -------
        res_it : Iterator
            The paralleled Iterator.
        """
        self.state_object, res = _ffi_api.StateParallel(
            self.state_object, self._resolve_stage_id(stage), iterator
        )
        return res

    def unroll(self, stage, iterator, max_unroll=None):
        """Schedule primitive corresponding to `te.Stage.unroll`.
        See also the `te.Stage` for more details.
        Parameters
        ----------
        stage : Union[int, Operation, Tensor]
            The Stage to be unrolled, which can be specified by the integer index, Operation,
            or output tensor of the stage.
        iterator : Iterator
            The iterator to be unrolled.
        max_unroll : Optional[int]
            The max unroll limit. Iterator with extent larger than this limit will be skipped.
        Returns
        -------
        res_it : Iterator
            The unrolled Iterator.
        """
        self.state_object, res = _ffi_api.StateUnroll(
            self.state_object,
            self._resolve_stage_id(stage),
            iterator,
            max_unroll if max_unroll else -1,
        )
        return res

    def vectorize(self, stage, iterator):
        """Schedule primitive corresponding to `te.Stage.vectorize`.
        See also the `te.Stage` for more details.
        Parameters
        ----------
        stage : Union[int, Operation, Tensor]
            The Stage to be vectorized, which can be specified by the integer index, Operation,
            or output tensor of the stage.
        iterator : Iterator
            The iterator to be vectorized.
        Returns
        -------
        res_it : Iterator
            The vectorized Iterator.
        """
        self.state_object, res = _ffi_api.StateVectorize(
            self.state_object, self._resolve_stage_id(stage), iterator
        )
        return res

    def fuse(self, stage, iters):
        """Schedule primitive corresponding to `te.Stage.fuse`.
        See also the `te.Stage` for more details.
        Parameters
        ----------
        stage : Union[int, Operation, Tensor]
            The Stage to be fused, which can be specified by the integer index, Operation,
            or output tensor of the stage.
        iters : List[Iterator]
            The iterators to be fused.
        Returns
        -------
        res_it : Iterator
            The fused Iterator.
        Notes
        -----
        If the iterators to be fused have stages attached at them(by compute_at), the fused
        result will become the new attach point.
        """
        self.state_object, res = _ffi_api.StateFuse(
            self.state_object, self._resolve_stage_id(stage), iters
        )
        return res

    def pragma(self, stage, iterator, pragma_type):
        """Schedule primitive corresponding to `te.Stage.pragma`.
        See also the `te.Stage` for more details.
        Parameters
        ----------
        stage : Union[int, Operation, Tensor]
            The Stage to add pragma, which can be specified by the integer index, Operation,
            or output tensor of the stage.
        iterator : Iterator
            The iterator to add pragma.
        pragma_type : str
            The pragma string.
        """
        self.state_object = _ffi_api.StatePragma(
            self.state_object, self._resolve_stage_id(stage), iterator, pragma_type
        )

    def reorder(self, stage, order):
        """Schedule primitive corresponding to `te.Stage.reorder`.
        See also the `te.Stage` for more details.
        Parameters
        ----------
        stage : Union[int, Operation, Tensor]
            The Stage to be reordered, which can be specified by the integer index, Operation,
            or output tensor of the stage.
        order : List[Iterator]
            Iterators in the expected order.
        """
        self.state_object = _ffi_api.StateReorder(
            self.state_object, self._resolve_stage_id(stage), order
        )

    def split(self, stage, iterator, lengths, inner_to_outer=True):
        """Schedule primitive corresponding to `te.Stage.split`.
        See also the `te.Stage` for more details.
        This API supports multiple split factors. (e.g. with 2 split factors, the original iterator
        will be split to 3 parts, use `inner_to_outer` to control the split order)
        Parameters
        ----------
        stage : Union[int, Operation, Tensor]
            The Stage to be split, which can be specified by the integer index, Operation,
            or output tensor of the stage.
        iterator : Iterator
            The iterator to be split.
        lengths: List[int]
            The multiple split factors. Can be None to be filled by search policy.
        inner_to_outer: boolean = True
            Whether the factor go from inner to outer, or from outer to inner.
        Returns
        -------
        res_its : List[Iterator]
            The splitted new Iterators.
        Notes
        -----
        If we do split on an iterator which has stages attached at it(by compute_at), the inner
        most iterator of split results will become the new attach point.
        """
        self.state_object, res = _ffi_api.StateSplit(
            self.state_object, self._resolve_stage_id(stage), iterator, lengths, inner_to_outer
        )
        return res

    def follow_split(self, stage, iterator, src_step_id, n_split):
        """The schedule primitive similar to split, but uses split factors from previous steps.
        This step splits the iterator by the same factors as the given SplitStep.
        Notes
        ------
            This step is useful in a scenario that we have subgraph Dense -> Relu,
            and we want to compute the Dense stage at ReLU. In this case, we need them to have
            the same tiling structure of common outer loops.
            The follow_split step could be used here to split the Dense stage and makes sure its
            splitting factors are the same as the given split step for the ReLU stage.
        Parameters
        ----------
        stage : Union[int, Operation, Tensor]
            The Stage to be split, which can be specified by the integer index, Operation,
            or output tensor of the stage.
        iterator : Iterator
            The iterator to split.
        src_step_id : int
            The index of the split step to be followed in the history.
        n_split : int
            The number of split level.
        Returns
        -------
        res_its : List[Iterator]
            The splitted new Iterators.
        """

        self.state_object, res = _ffi_api.StateFollowSplit(
            self.state_object, self._resolve_stage_id(stage), iterator, src_step_id, n_split
        )
        return res

    def follow_fused_split(self, stage, iterator, src_step_ids, level, factor_or_nparts):
        """Schedule primitive extends to split step.
        This step is used to split an iterator by the same factors
        as the given list of SplitSteps and FuseSteps.
        Notes
        ------
            This step is useful in a scenario that we have a subgraph
            in GPU schedule: Input -> Dense
            for i.0@j.0 = ... : Bind to blockIdx.x
                for i.1@j.1 = ... : Bind to threadIdx.x
                    for i.2@j.2 = ...
                        Input_shared = Input ...
                        for k = ...
                            Dense = ...
            We intend to apply cooperative fetching with the input stage, while the threadIdx.x
            axis is bound to an iterator generated by split & fuse step.
            The follow_fused_step is used split the iterator to 2 parts, while the split factor
            matches the final extent of the threadIdx.x bound iterator.
        Parameters
        ----------
        stage : Union[int, Operation, Tensor]
            The Stage to be split, which can be specified by the integer index, Operation,
            or output tensor of the stage.
        iterator : Iterator
            The iterator to split.
        src_step_ids : List[int]
            The indices of the split steps to be followed in the history.
        level : int
            Use the length in this split level.
        factor_or_nparts : bool
            True to use `factor` for split from inner to outer,
            False to use `nparts` for split from outer to inner.
        Returns
        -------
        res_its : List[Iterator]
            The splitted new Iterators.
        """

        self.state_object, res = _ffi_api.StateFollowFusedSplit(
            self.state_object,
            self._resolve_stage_id(stage),
            iterator,
            src_step_ids,
            level,
            factor_or_nparts,
        )
        return res

    def storage_align(self, stage, iterator, factor, offset):
        """Schedule primitive corresponding to `te.Stage.storage_align`.
        See also the `te.Stage` for  more details.
        Parameters
        ----------
        stage : Union[int, Operation, Tensor]
            The Stage to be storage aligned, which can be specified by the integer index,
            Operation, or output tensor of the stage.
        iterator : Iterator
            The iterator to be aligned.
        factor : int
            The factor in alignment specification.
        offset : int
            The offset in the alignment specification.
        """
        self.state_object = _ffi_api.StateStorageAlign(
            self.state_object, self._resolve_stage_id(stage), iterator, factor, offset
        )

    def compute_at(self, stage, target_stage, target_iter):
        """Schedule primitive corresponding to `te.Stage.compute_at`.
        See also the `te.Stage` for more details.
        Parameters
        ----------
        stage : Union[int, Operation, Tensor]
            The source Stage of computed at, which can be specified by the integer index,
            Operation, or output tensor of the stage.
        target_stage : Union[int, Operation, Tensor]
            The target stage of compute_at, which can be specified by the integer index, Operation,
            or output tensor of the stage.
        target_iter : Iterator
            The target Iterator of compute_at.
        Notes
        -----
        After compute_at, we need careful dependency analysis to compute the accurate bound
        information. However, it is relatively expensive and complicated, so we just fill "None"
        as bound for the newly created iterators.
        Call ComputeDAG::InferBound on the returned state to get the complete bound information.
        """
        self.state_object = _ffi_api.StateComputeAt(
            self.state_object,
            self._resolve_stage_id(stage),
            self._resolve_stage_id(target_stage),
            target_iter,
        )

    def compute_inline(self, stage):
        """Schedule primitive corresponding to `te.Stage.compute_inline`, see also the `te.Stage`
        for more details.
        Parameters
        ----------
        stage : Union[int, Operation, Tensor]
            The Stage to be marked compute inlined, which can be specified by the integer index,
            Operation, or output tensor of the stage.
        """
        self.state_object = _ffi_api.StateComputeInline(
            self.state_object, self._resolve_stage_id(stage)
        )

    def compute_root(self, stage):
        """Schedule primitive corresponding to `te.Stage.compute_root`.
        Ssee also the `te.Stage` for more details.
        Parameters
        ----------
        stage : Union[int, Operation, Tensor]
            The Stage to be marked compute at root, which can be specified by the integer index,
            Operation, or output tensor of the stage.
        Notes
        -----
        After compute_root, we need careful dependency analysis to compute the accurate bound
        information. However, it is relatively expensive and complicated, so we just fill "None"
        as bound for the newly created iterators.
        Call ComputeDAG::InferBound on the returned state to get the complete bound information.
        """
        self.state_object = _ffi_api.StateComputeRoot(
            self.state_object, self._resolve_stage_id(stage)
        )

    def cache_read(self, stage, scope_name, reader_stages):
        """Schedule primitive corresponding to `te.Schedule.cache_read`.
        See also the `te.Schedule` for more details.
        Parameters
        ----------
        stage : Union[int, Operation, Tensor]
            The Stage to be cache_read, which can be specified by the integer index, Operation,
            or output tensor of the stage.
        scope_name : str
            The scope name of the newly added read stage.
        reader_stages : List[Union[int, Operation, Tensor]]
            The reader stages. Each of the list can be specified by the integer index, Operation,
            or output tensor of the stage.
        Returns
        -------
        new_stage_op : Operator
            The Operator of the new added stage.
        Notes
        -----
        Cache read step will insert an extra stage to the original ComputeDAG (at the back of the
        target stage).
        """
        reader_stage_ids = [self._resolve_stage_id(i) for i in reader_stages]
        self.state_object, new_stage_id = _ffi_api.StateCacheRead(
            self.state_object,
            self._resolve_stage_id(stage),
            scope_name,
            reader_stage_ids,
            self.compute_dag,
        )
        # Add a new stage will change all ops behind the added stage. But we still want to keep the
        # original ops map, apply stage id offset to stage_id_map to make them work.
        self._apply_stage_id_offset(int(new_stage_id))
        self._update_stage_id_map()
        return self.stages[int(new_stage_id)].op

    def cache_write(self, stage, scope_name):
        """Schedule primitive corresponding to `te.Schedule.cache_write`.
        See also the `te.Schedule` for more details.
        Parameters
        ----------
        stage : Union[int, Operation, Tensor]
            The Stage to be cache_write, which can be specified by the integer index, Operation,
            or output tensor of the stage.
        scope_name : str
            The scope name of the newly added compute stage.
        Returns
        -------
        new_stage_op : Operator
            The Operator of the new added stage.
        Notes
        -----
        Cache write step will insert an extra stage to the original ComputeDAG (in the front of the
        target stage).
        This step will cache write all output tensors of the target stage.
        """
        self.state_object, new_stage_id = _ffi_api.StateCacheWrite(
            self.state_object, self._resolve_stage_id(stage), scope_name, self.compute_dag
        )
        # Add a new stage will change all ops behind the added stage. But we still want to keep the
        # original ops map, apply stage id offset to stage_id_map to make them work.
        self._apply_stage_id_offset(int(new_stage_id))
        self._update_stage_id_map()
        return self.stages[int(new_stage_id)].op

    def rfactor(self, stage, iterator, factor_iter_id):
        """Schedule primitive corresponding to `te.Schedule.rfactor`.
        See also the `te.Schedule` for more details.
        Parameters
        ----------
        stage : Union[int, Operation, Tensor]
            The Stage to be factored, which can be specified by the integer index, Operation,
            or output tensor of the stage.
        iterator : Iterator
            The reduction iterator to be factored.
        factor_iter_id : int
            The position where the new iterator is placed.
        Returns
        -------
        new_stage_op : Operator
            The Operator of the new added stage.
        Notes
        -----
        Rfactor step will insert an extra stage to the original ComputeDAG (in the front of the
        target stage).
        """
        self.state_object, new_stage_id = _ffi_api.StateRfactor(
            self.state_object,
            self._resolve_stage_id(stage),
            iterator,
            factor_iter_id,
            self.compute_dag,
        )
        # Add a new stage will change all ops behind the added stage. But we still want to keep the
        # original ops map, apply stage id offset to stage_id_map to make them work.
        self._apply_stage_id_offset(int(new_stage_id))
        self._update_stage_id_map()
        return self.stages[int(new_stage_id)].op

    def copy(self):
        """Do deep copy of this State."""
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
        raise ValueError(
            "Invalid stage: " + stage_id + " . Expect to be a int, Operation or Tensor"
        )

    def _update_stage_id_map(self):
        for index, stage in enumerate(self.stages):
            self.stage_id_map[stage.op] = index

    def _apply_stage_id_offset(self, start_id, offset=1):
        for key, value in self.stage_id_map.items():
            if value >= start_id:
                self.stage_id_map[key] = value + offset

    def __getitem__(self, key):
        if isinstance(key, Tensor):
            key = key.op
        if isinstance(key, Operation):
            return self.stages[self.stage_id_map[key]]
        raise ValueError("Invalid item: " + key + " . Expect to be a Operation or Tensor")

    def __str__(self):
        return str(self.state_object)

    def __eq__(self, other):
        return _ffi_api.StateEqual(self.state_object, other.state_object)
