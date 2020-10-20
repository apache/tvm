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
"""The computation schedule api of TVM."""
import tvm._ffi
from tvm._ffi.base import string_types

from tvm.runtime import Object, convert
from tvm.ir import container as _container
from tvm.tir import IterVar, Buffer

from . import tensor as _tensor
from . import _ffi_api


@tvm._ffi.register_object
class Split(Object):
    """Split operation on axis."""


@tvm._ffi.register_object
class Fuse(Object):
    """Fuse operation on axis."""


@tvm._ffi.register_object
class Singleton(Object):
    """Singleton axis."""


def create_schedule(ops):
    """Create a schedule for list of ops

    Parameters
    ----------
    ops : list of Operations
        The source expression.

    Returns
    -------
    sch : schedule.Schedule
        The created schedule.
    """
    if not isinstance(ops, (list, _container.Array)):
        ops = [ops]
    return _ffi_api.CreateSchedule(ops)


@tvm._ffi.register_object
class Schedule(Object):
    """Schedule for all the stages."""

    def __getitem__(self, k):
        if isinstance(k, _tensor.Tensor):
            k = k.op
        if not isinstance(k, _tensor.Operation):
            raise ValueError("Expect schedule key to be Tensor or Operation")
        if k not in self.stage_map:
            raise ValueError("Cannot find the operation %s in schedule" % (str(k)))
        return self.stage_map[k]

    def normalize(self):
        """Build a normalized schedule from the current schedule.

        Insert necessary rebase to make certain iter var to start from 0.
        This is needed before bound inference and followup step.

        Returns
        -------
        sch : Schedule
            The normalized schedule.
        """
        return _ffi_api.ScheduleNormalize(self)

    def create_group(self, outputs, inputs, include_inputs=False):
        """Create stage group by giving output and input boundary.

        The operators between outputs and inputs are placed as member of group.
        outputs are include in the group, while inputs are not included.

        Parameters
        ----------
        outputs : list of Tensors
            The outputs of the group.

        inputs : list of Tensors
            The inputs of the group.

        include_inputs : boolean, optional
            Whether include input operations in the group if they are used by outputs.

        Returns
        -------
        group : Stage
            A virtual stage represents the group, user can use compute_at to move
            the attachment point of the group.
        """
        if isinstance(outputs, _tensor.Tensor):
            outputs = [outputs]
        if isinstance(inputs, _tensor.Tensor):
            inputs = [inputs]
        return _ffi_api.ScheduleCreateGroup(self, outputs, inputs, include_inputs)

    def cache_read(self, tensor, scope, readers):
        """Create a cache read of original tensor for readers.

        This will mutate the body of the readers.
        A new cache stage will be created for the tensor.
        Call this before doing any split/fuse schedule.

        Parameters
        ----------
        tensor : Tensor
            The tensor to be cached.
        scope : str
            The scope of cached
        readers : list of Tensor or Operation
            The readers to read the cache.

        Returns
        -------
        cache : Tensor
            The created cache tensor.
        """
        if isinstance(readers, (_tensor.Tensor, _tensor.Operation)):
            readers = [readers]
        readers = [t.op if isinstance(t, _tensor.Tensor) else t for t in readers]
        return _ffi_api.ScheduleCacheRead(self, tensor, scope, readers)

    def cache_write(self, tensor, scope):
        """Create a cache write of original tensor, before storing into tensor.

        This will mutate the body of the tensor.
        A new cache stage will created before feed into the tensor.

        This function can be used to support data layout transformation.
        If there is a split/fuse/reorder on the data parallel axis of tensor
        before cache_write is called. The intermediate cache stores
        the data in the layout as the iteration order of leave axis.
        The data will be transformed back to the original layout in the original tensor.
        User can further call compute_inline to inline the original layout and keep
        the data stored in the transformed layout.

        Parameters
        ----------
        tensor : Tensor, list or tuple
            The tensors to be feed to. All the tensors must be produced by one computeOp
        scope : str
            The scope of cached

        Returns
        -------
        cache : Tensor
            The created cache tensor.
        """
        return _ffi_api.ScheduleCacheWrite(self, tensor, scope)

    def rfactor(self, tensor, axis, factor_axis=0):
        """Factor a reduction axis in tensor's schedule to be an explicit axis.

        This will create a new stage that generated the new tensor with axis
        as the first dimension. The tensor's body will be rewritten as a reduction
        over the factored tensor.

        Parameters
        ----------
        tensor : Tensor
            The tensor to be factored.
        axis : IterVar
            The reduction axis in the schedule to be factored.
        factor_axis : int
            The position where the new axis is placed.

        Returns
        -------
        tfactor : Tensor or Array of Tensor
            The created factored tensor.
        """
        factored = _ffi_api.ScheduleRFactor(self, tensor, axis, factor_axis)
        return factored[0] if len(factored) == 1 else factored


@tvm._ffi.register_object
class Stage(Object):
    """A Stage represents schedule for one operation."""

    def split(self, parent, factor=None, nparts=None):
        """Split the stage either by factor providing outer scope, or both

        Parameters
        ----------
        parent : IterVar
             The parent iter var.

        factor : Expr, optional
             The splitting factor

        nparts : Expr, optional
             The number of outer parts.

        Returns
        -------
        outer : IterVar
            The outer variable of iteration.

        inner : IterVar
            The inner variable of iteration.
        """
        if nparts is not None:
            if factor is not None:
                raise ValueError("Do not need to provide both outer and nparts")
            outer, inner = _ffi_api.StageSplitByNParts(self, parent, nparts)
        else:
            if factor is None:
                raise ValueError("Either nparts or factor need to be provided")
            outer, inner = _ffi_api.StageSplitByFactor(self, parent, factor)
        return outer, inner

    def fuse(self, *args):
        """Fuse multiple consecutive iteration variables into a single iteration variable.

        fused = fuse(...fuse(fuse(args[0], args[1]), args[2]),..., args[-1])
        The order is from outer to inner.

        Parameters
        ----------
        args : list of IterVars
            Itervars that proceeds each other

        Returns
        -------
        fused : IterVar
            The fused variable of iteration.
        """
        fused = _ffi_api.StageFuse(self, args)
        return fused

    def set_scope(self, scope):
        """Set the thread scope of this stage

        Parameters
        ----------
        scope : str
            The thread scope of this stage
        """
        return _ffi_api.StageSetScope(self, scope)

    def bind(self, ivar, thread_ivar):
        """Bind ivar to thread index thread_ivar

        Parameters
        ----------
        ivar : IterVar
            The iteration to be binded to thread.

        thread_ivar : IterVar
            The thread to be binded.
        """
        _ffi_api.StageBind(self, ivar, thread_ivar)

    def env_threads(self, threads):
        """Mark threads to be launched at the outer scope of composed op.

        Parameters
        ----------
        threads : list of threads
            The threads to be launched.
        """
        if isinstance(threads, IterVar):
            threads = [threads]
        _ffi_api.StageEnvThreads(self, threads)

    def set_store_predicate(self, predicate):
        """Set predicate under which store to the array can be performed.

        Use this when there are duplicated threads doing the same store and we only
        need one of them to do the store.

        Parameters
        ----------
        predicate : Expr
            The guard condition fo store.
        """
        _ffi_api.StageSetStorePredicate(self, predicate)

    def compute_at(self, parent, scope):
        """Attach the stage at parent's scope

        Parameters
        ----------
        parent : Stage
            The parent stage

        scope : IterVar
            The loop scope t be attached to.
        """
        _ffi_api.StageComputeAt(self, parent, scope)

    def compute_inline(self):
        """Mark stage as inline

        Parameters
        ----------
        parent : Stage
            The parent stage
        """
        _ffi_api.StageComputeInline(self)

    def compute_root(self):
        """Attach the stage at parent, and mark it as root

        Parameters
        ----------
        parent : Stage
            The parent stage
        """
        _ffi_api.StageComputeRoot(self)

    def reorder(self, *args):
        """reorder the arguments in the specified order.

        Parameters
        ----------
        args : list of IterVar
            The order to be ordered
        """
        _ffi_api.StageReorder(self, args)

    def tile(self, x_parent, y_parent, x_factor, y_factor):
        """Perform tiling on two dimensions

        The final loop order from outmost to inner most are
        [x_outer, y_outer, x_inner, y_inner]

        Parameters
        ----------
        x_parent : IterVar
            The original x dimension
        y_parent : IterVar
            The original y dimension
        x_factor : Expr
            The stride factor on x axis
        y_factor : Expr
            The stride factor on y axis

        Returns
        -------
        x_outer : IterVar
            Outer axis of x dimension
        y_outer : IterVar
            Outer axis of y dimension
        x_inner : IterVar
            Inner axis of x dimension
        p_y_inner : IterVar
            Inner axis of y dimension
        """
        x_outer, y_outer, x_inner, y_inner = _ffi_api.StageTile(
            self, x_parent, y_parent, x_factor, y_factor
        )
        return x_outer, y_outer, x_inner, y_inner

    def vectorize(self, var):
        """Vectorize the iteration.

        Parameters
        ----------
        var : IterVar
            The iteration to be vectorize
        """
        _ffi_api.StageVectorize(self, var)

    def tensorize(self, var, tensor_intrin):
        """Tensorize the computation enclosed by var with tensor_intrin

        Parameters
        ----------
        var : IterVar
            The iteration boundary of tensorization.

        tensor_intrin : TensorIntrin
            The tensor intrinsic used for computation.
        """
        _ffi_api.StageTensorize(self, var, tensor_intrin)

    def unroll(self, var):
        """Unroll the iteration.

        Parameters
        ----------
        var : IterVar
            The iteration to be unrolled.
        """
        _ffi_api.StageUnroll(self, var)

    def parallel(self, var):
        """Parallelize the iteration.

        Parameters
        ----------
        var : IterVar
            The iteration to be parallelized.
        """
        _ffi_api.StageParallel(self, var)

    def pragma(self, var, pragma_type, pragma_value=None):
        """Annotate the iteration with pragma

        This will translate to a pragma_scope surrounding
        the corresponding loop generated.
        Useful to support experimental features and extensions.

        Parameters
        ----------
        var : IterVar
            The iteration to be anotated

        pragma_type : str
             The pragma string to be annotated

        pragma_value : Expr, optional
             The pragma value to pass along the pragma

        Note
        ----
        Most pragmas are advanced/experimental features
        and may subject to change. List of supported pragmas:

        - **debug_skip_region**

          Force skip the region marked by the axis and turn it into no-op.
          This is useful for debug purposes.

        - **parallel_launch_point**

          Specify to launch parallel threads outside the
          specified iteration loop. By default the threads
          launch at the point of parallel construct.
          This pragma moves the launching point to even outer scope.
          The threads are launched once and reused across multiple
          parallel constructs as BSP style program.

        - **parallel_barrier_when_finish**

          Insert a synchronization barrier between working threads
          after the specified loop iteration finishes.

        - **parallel_stride_pattern**

          Hint parallel loop to execute in strided pattern.
          :code:`for (int i = task_id; i < end; i += num_task)`

        """
        if isinstance(pragma_value, string_types):
            pragma_value = convert(pragma_value)
        _ffi_api.StagePragma(self, var, pragma_type, pragma_value)

    def prefetch(self, tensor, var, offset):
        """Prefetch the specified variable

        Parameters
        ----------
        tensor : Tensor
            The tensor to be prefetched
        var : IterVar
            The loop point at which the prefetching is applied
        offset : Expr
            The number of iterations to be prefetched before actual execution
        """
        _ffi_api.StagePrefetch(self, tensor, var, offset)

    def storage_align(self, axis, factor, offset):
        """Set alignment requirement for specific axis

        This ensures that stride[axis] == k * factor + offset for some k.
        This is useful to set memory layout to for more friendly memory
        access pattern. For example, we can set alignment to be
        factor=2, offset=1 to avoid bank conflict for thread access on
        higher dimension in GPU shared memory.

        Parameters
        ----------
        axis : IterVar
            The axis dimension to be aligned.
        factor : int
            The factor in alignment specification.
        offset : int
            The offset in the alignment specification.
        """
        _ffi_api.StageStorageAlign(self, axis, factor, offset)

    def double_buffer(self):
        """Compute the current stage via double buffering.

        This can only be applied to intermediate stage.
        This will double the storage cost of the current stage.
        Can be useful to hide load latency.
        """
        _ffi_api.StageDoubleBuffer(self)


@tvm._ffi.register_object
class SpecializedCondition(Object):
    """Specialized condition to enable op specialization."""

    def __init__(self, conditions):
        """Create a specialized condition.

        .. note::
            Conditions are represented in conjunctive joint form (CNF).
            Each condition should be a simple expression, e.g., n > 16,
            m % 8 == 0, etc., where n, m are tvm.Var that represents a
            dimension in the tensor shape.

        Parameters
        ----------
        conditions : List of tvm.Expr
            List of conditions in conjunctive joint form (CNF).
        """
        if not isinstance(conditions, (list, _container.Array)):
            conditions = [conditions]
        self.__init_handle_by_constructor__(_ffi_api.CreateSpecializedCondition, conditions)

    @staticmethod
    def current():
        """Returns the current specialized condition"""
        return _ffi_api.GetCurrentSpecialization()

    def __enter__(self):
        _ffi_api.EnterSpecializationScope(self)
        return self

    def __exit__(self, ptype, value, trace):
        _ffi_api.ExitSpecializationScope(self)


tvm._ffi._init_api("schedule", __name__)
