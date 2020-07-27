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

    # Static trans table for thread bind
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
        "tensorize": 11
    }

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

    def bind(self, stage, iterator, thread_name):
        """ Schedule primitive corresponds to `te.Stage.bind`, see also the `te.Stage` for more
        details.

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

        self.state_object, res = _ffi_api.StateBind(self.state_object,
                                                    self._resolve_stage_id(stage), iterator,
                                                    State.ANNOTATION_TRANS_TABLE[thread_name])
        return res

    def parallel(self, stage, iterator):
        """ Schedule primitive corresponds to `te.Stage.parallel`, see also the `te.Stage` for more
        details.

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
        self.state_object, res = _ffi_api.StateParallel(self.state_object,
                                                        self._resolve_stage_id(stage), iterator)
        return res

    def unroll(self, stage, iterator, max_unroll=None):
        """ Schedule primitive corresponds to `te.Stage.unroll`, see also the `te.Stage` for more
        details.

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
        self.state_object, res = _ffi_api.StateUnroll(self.state_object,
                                                      self._resolve_stage_id(stage), iterator,
                                                      max_unroll if max_unroll else -1)
        return res

    def vectorize(self, stage, iterator):
        """ Schedule primitive corresponds to `te.Stage.vectorize`, see also the `te.Stage` for
        more details.

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
        self.state_object, res = _ffi_api.StateVectorize(self.state_object,
                                                         self._resolve_stage_id(stage), iterator)
        return res

    def fuse(self, stage, iters):
        """ Schedule primitive corresponds to `te.Stage.fuse`, see also the `te.Stage` for more
        details.

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
        self.state_object, res = _ffi_api.StateFuse(self.state_object,
                                                    self._resolve_stage_id(stage), iters)
        return res

    def reorder(self, stage, order):
        """ Schedule primitive corresponds to `te.Stage.reorder`, see also the `te.Stage` for more
        details.

        Parameters
        ----------
        stage : Union[int, Operation, Tensor]
            The Stage to be reordered, which can be specified by the integer index, Operation,
            or output tensor of the stage.
        order : List[Iterator]
            Iterators in the expected order.
        """
        self.state_object = _ffi_api.StateReorder(self.state_object, self._resolve_stage_id(stage),
                                                  order)

    def split(self, stage, iterator, lengths, inner_to_outer=True):
        """ Schedule primitive corresponds to `te.Stage.split`, see also the `te.Stage` for more
        details.

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
        self.state_object, res = _ffi_api.StateSplit(self.state_object,
                                                     self._resolve_stage_id(stage),
                                                     iterator, lengths, inner_to_outer)
        return res

    def compute_at(self, stage, target_stage, target_iter):
        """ Schedule primitive corresponds to `te.Stage.compute_at`, see also the `te.Stage` for
        more details.

        Parameters
        ----------
        stage : Union[int, Operation, Tensor]
            The Stage to be computed at, which can be specified by the integer index, Operation,
            or output tensor of the stage.
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
        self.state_object = _ffi_api.StateComputeAt(self.state_object,
                                                    self._resolve_stage_id(stage),
                                                    self._resolve_stage_id(target_stage),
                                                    target_iter)

    def compute_inline(self, stage):
        """ Schedule primitive corresponds to `te.Stage.compute_inline`, see also the `te.Stage`
        for more details.

        Parameters
        ----------
        stage : Union[int, Operation, Tensor]
            The Stage to be marked compute inlined, which can be specified by the integer index,
            Operation, or output tensor of the stage.
        """
        self.state_object = _ffi_api.StateComputeInline(self.state_object,
                                                        self._resolve_stage_id(stage))

    def compute_root(self, stage):
        """ Schedule primitive corresponds to `te.Stage.compute_root`, see also the `te.Stage` for
        more details.

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
        self.state_object = _ffi_api.StateComputeRoot(self.state_object,
                                                      self._resolve_stage_id(stage))

    def cache_read(self, stage, scope_name, reader_stages):
        """ Schedule primitive corresponds to `te.Schedule.cache_read`, see also the `te.Schedule`
        for more details.

        Parameters
        ----------
        stage : Union[int, Operation, Tensor]
            The Stage to be cache read, which can be specified by the integer index, Operation,
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
        self.state_object, new_stage_id = _ffi_api.StateCacheRead(self.state_object,
                                                                  self._resolve_stage_id(stage),
                                                                  scope_name, reader_stage_ids,
                                                                  self.compute_dag)
        # Add a new stage will change all ops behind the added stage. But we still want to keep the
        # original ops map, apply stage id offset to stage_id_map to make them work.
        self._apply_stage_id_offset(int(new_stage_id))
        self._update_stage_id_map()
        return self.stages[int(new_stage_id)].op

    def cache_write(self, stage, scope_name):
        """ Schedule primitive corresponds to `te.Schedule.cache_write`, see also the `te.Schedule`
        for more details.

        Parameters
        ----------
        stage : Union[int, Operation, Tensor]
            The Stage to be cache write, which can be specified by the integer index, Operation,
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
        self.state_object, new_stage_id = _ffi_api.StateCacheWrite(self.state_object,
                                                                   self._resolve_stage_id(stage),
                                                                   scope_name, self.compute_dag)
        # Add a new stage will change all ops behind the added stage. But we still want to keep the
        # original ops map, apply stage id offset to stage_id_map to make them work.
        self._apply_stage_id_offset(int(new_stage_id))
        self._update_stage_id_map()
        return self.stages[int(new_stage_id)].op

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

    def _apply_stage_id_offset(self, start_id, offset=1):
        for key, value in self.stage_id_map.items():
            if value >= start_id:
                self.stage_id_map[key] = value + offset

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
