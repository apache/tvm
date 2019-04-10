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
"""The computation schedule api of TVM."""
from __future__ import absolute_import as _abs
from ._ffi.base import string_types
from ._ffi.node import NodeBase, register_node
from ._ffi.node import convert_to_node as _convert_to_node
from ._ffi.function import _init_api, Function
from ._ffi.function import convert_to_tvm_func as _convert_tvm_func
from . import _api_internal
from . import tensor as _tensor
from . import expr as _expr
from . import container as _container

def convert(value):
    """Convert value to TVM node or function.

    Parameters
    ----------
    value : python value

    Returns
    -------
    tvm_val : Node or Function
        Converted value in TVM
    """
    if isinstance(value, (Function, NodeBase)):
        return value

    if callable(value):
        return _convert_tvm_func(value)

    return _convert_to_node(value)

@register_node
class Buffer(NodeBase):
    """Symbolic data buffer in TVM.

    Buffer provide a way to represent data layout
    specialization of data structure in TVM.

    Do not construct directly, use :any:`decl_buffer` instead.
    See the documentation of :any:`decl_buffer` for more details.

    See Also
    --------
    decl_buffer : Declare a buffer
    """
    READ = 1
    WRITE = 2

    def access_ptr(self, access_mask, ptr_type="handle", content_lanes=1, offset=0):
        """Get an access pointer to the head of buffer.

        This is the recommended method to get buffer data
        ptress when interacting with external functions.

        Parameters
        ----------
        access_mask : int
            The access pattern MASK. Indicate whether the
            access will read or write to the data content.

        ptr_type : str, optional
            The data type of the result pointer. Do not specify
            unless we want to cast pointer to specific type.

        content_lanes: int, optional
            The number of lanes for the data type. This value
            is greater than one for vector types.

        offset: Expr, optional
            The offset of pointer. We can use it to offset by
            the number of elements from the address of ptr.

        Examples
        --------
        .. code-block:: python

          import tvm.schedule.Buffer
          # Get access ptr for read
          buffer.access_ptr("r")
          # Get access ptr for read/write with bitmask
          buffer.access_ptr(Buffer.READ | Buffer.WRITE)
          # Get access ptr for read/write with str flag
          buffer.access_ptr("rw")
          # Get access ptr for read with offset
          buffer.access_ptr("r", offset = 100)
        """
        if isinstance(access_mask, string_types):
            mask = 0
            for value in access_mask:
                if value == "r":
                    mask = mask | Buffer.READ
                elif value == "w":
                    mask = mask | Buffer.WRITE
                else:
                    raise ValueError("Unknown access_mask %s" % access_mask)
            access_mask = mask
        offset = convert(offset)
        return _api_internal._BufferAccessPtr(self, access_mask, ptr_type,
                                              content_lanes, offset)

    def vload(self, begin, dtype=None):
        """Generate an Expr that loads dtype from begin index.

        Parameters
        ----------
        begin : Array of Expr
            The beginning index in unit of Buffer.dtype

        dtype : str
            The data type to be loaded,
            can be vector type which have lanes that is multiple of Buffer.dtype

        Returns
        -------
        load : Expr
            The corresponding load expression.
        """
        begin = (begin,) if isinstance(begin, (int, _expr.Expr)) else begin
        dtype = dtype if dtype else self.dtype
        return _api_internal._BufferVLoad(self, begin, dtype)

    def vstore(self, begin, value):
        """Generate a Stmt that store value into begin index.

        Parameters
        ----------
        begin : Array of Expr
            The beginning index in unit of Buffer.dtype

        value : Expr
            The value to be stored.

        Returns
        -------
        store : Stmt
            The corresponding store stmt.
        """
        begin = (begin,) if isinstance(begin, (int, _expr.Expr)) else begin
        return _api_internal._BufferVStore(self, begin, value)


@register_node
class Split(NodeBase):
    """Split operation on axis."""


@register_node
class Fuse(NodeBase):
    """Fuse operation on axis."""


@register_node
class Singleton(NodeBase):
    """Singleton axis."""


@register_node
class IterVar(NodeBase, _expr.ExprOp):
    """Represent iteration variable.

    IterVar is normally created by Operation, to represent
    axis iterations in the computation.
    It can also created by schedule primitives like :any:`tvm.schedule.Stage.split`.

    See Also
    --------
    tvm.thread_axis: Create thread axis IterVar.
    tvm.reduce_axis: Create reduce axis IterVar.
    """
    DataPar = 0
    ThreadIndex = 1
    CommReduce = 2
    Ordered = 3
    DimInfo = 4
    Unrolled = 5
    Vectorized = 6
    Parallelized = 7
    Tensorized = 8

_tensor.iter_var_cls = IterVar

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
    return _api_internal._CreateSchedule(ops)


@register_node
class Schedule(NodeBase):
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
        return _api_internal._ScheduleNormalize(self)

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
        return _api_internal._ScheduleCreateGroup(
            self, outputs, inputs, include_inputs)

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
        return _api_internal._ScheduleCacheRead(self, tensor, scope, readers)

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
        return _api_internal._ScheduleCacheWrite(self, tensor, scope)

    def rfactor(self, tensor, axis, factor_axis=0):
        """ Factor a reduction axis in tensor's schedule to be an explicit axis.

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
        factored = _api_internal._ScheduleRFactor(self, tensor, axis, factor_axis)
        return factored[0] if len(factored) == 1 else factored


@register_node
class Stage(NodeBase):
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
            outer, inner = _api_internal._StageSplitByNParts(self, parent, nparts)
        else:
            if factor is None:
                raise ValueError("Either nparts or factor need to be provided")
            outer, inner = _api_internal._StageSplitByFactor(self, parent, factor)
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
        fused = _api_internal._StageFuse(self, args)
        return fused

    def set_scope(self, scope):
        """Set the thread scope of this stage

        Parameters
        ----------
        scope : str
            The thread scope of this stage
        """
        return _api_internal._StageSetScope(self, scope)

    def bind(self, ivar, thread_ivar):
        """Bind ivar to thread index thread_ivar

        Parameters
        ----------
        ivar : IterVar
            The iteration to be binded to thread.

        thread_ivar : IterVar
            The thread to be binded.
        """
        _api_internal._StageBind(self, ivar, thread_ivar)

    def env_threads(self, threads):
        """Mark threads to be launched at the outer scope of composed op.

        Parameters
        ----------
        threads : list of threads
            The threads to be launched.
        """
        if isinstance(threads, IterVar):
            threads = [threads]
        _api_internal._StageEnvThreads(self, threads)

    def set_store_predicate(self, predicate):
        """Set predicate under which store to the array can be performed.

        Use this when there are duplicated threads doing the same store and we only
        need one of them to do the store.

        Parameters
        ----------
        predicate : Expr
            The guard condition fo store.
        """
        _api_internal._StageSetStorePredicate(self, predicate)

    def compute_at(self, parent, scope):
        """Attach the stage at parent's scope

        Parameters
        ----------
        parent : Stage
            The parent stage

        scope : IterVar
            The loop scope t be attached to.
        """
        _api_internal._StageComputeAt(self, parent, scope)

    def compute_inline(self):
        """Mark stage as inline

        Parameters
        ----------
        parent : Stage
            The parent stage
        """
        _api_internal._StageComputeInline(self)

    def compute_root(self):
        """Attach the stage at parent, and mark it as root

        Parameters
        ----------
        parent : Stage
            The parent stage
        """
        _api_internal._StageComputeRoot(self)

    def reorder(self, *args):
        """reorder the arguments in the specified order.

        Parameters
        ----------
        args : list of IterVar
            The order to be ordered
        """
        _api_internal._StageReorder(self, args)

    def tile(self, x_parent, y_parent, x_factor, y_factor):
        """ Perform tiling on two dimensions

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
        x_outer, y_outer, x_inner, y_inner = _api_internal._StageTile(
            self, x_parent, y_parent, x_factor, y_factor)
        return x_outer, y_outer, x_inner, y_inner

    def vectorize(self, var):
        """Vectorize the iteration.

        Parameters
        ----------
        var : IterVar
            The iteration to be vectorize
        """
        _api_internal._StageVectorize(self, var)

    def tensorize(self, var, tensor_intrin):
        """Tensorize the computation enclosed by var with tensor_intrin

        Parameters
        ----------
        var : IterVar
            The iteration boundary of tensorization.

        tensor_intrin : TensorIntrin
            The tensor intrinsic used for computation.
        """
        _api_internal._StageTensorize(self, var, tensor_intrin)

    def unroll(self, var):
        """Unroll the iteration.

        Parameters
        ----------
        var : IterVar
            The iteration to be unrolled.
        """
        _api_internal._StageUnroll(self, var)

    def parallel(self, var):
        """Parallelize the iteration.

        Parameters
        ----------
        var : IterVar
            The iteration to be parallelized.
        """
        _api_internal._StageParallel(self, var)

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
        _api_internal._StagePragma(self, var, pragma_type, pragma_value)

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
        _api_internal._StagePrefetch(self, tensor, var, offset)

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
        _api_internal._StageStorageAlign(self, axis, factor, offset)

    def double_buffer(self):
        """Compute the current stage via double buffering.

        This can only be applied to intermediate stage.
        This will double the storage cost of the current stage.
        Can be useful to hide load latency.
        """
        _api_internal._StageDoubleBuffer(self)

    def opengl(self):
        """The special OpenGL schedule

        Maps each output element to a pixel.
        """
        _api_internal._StageOpenGL(self)

_init_api("tvm.schedule")
