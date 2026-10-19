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
"""Wrapping existing transformations."""
# pylint: disable=invalid-name, unsupported-binary-operation

from collections.abc import Callable

from . import _ffi_api
from .function_pass import function_pass


def DecorateDeviceScope():
    """Wrap each function body in a ``tirx.device_scope`` region.

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.DecorateDeviceScope()(before)

    TVMScript body fragments:

    .. code-block:: python

        # Before
        T.evaluate(x)
        # After
        with T.region("tirx.device_scope", []):
            T.evaluate(x)
    """
    return _ffi_api.DecorateDeviceScope()


def Apply(ftransform):
    """Apply ftransform to each function in the Module.

    This function is a thin wrapper around :py:func:`function_pass`

    Parameters
    ----------
    ftransform: tvm_tirx.tirx.Function -> tvm_tirx.tirx.Function
       The transformation pass.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.Apply(lambda f: f.with_attr("example", 1))(before)

    Here the callback adds an attribute; TVMScript body fragments:

    .. code-block:: python

        # Before
        T.evaluate(0)
        # After
        T.func_attr({"example": 1})
        T.evaluate(0)
    """

    # pylint: disable=unused-argument
    def _transform(func, mod, ctx):
        return ftransform(func)

    return function_pass(_transform, opt_level=0, name="Apply")  # type: ignore


def VectorizeLoop(enable_vectorize: bool = True):
    """Lower vectorization loops.

    Target-dependent vectorization uses the Function's ``target`` attribute,
    not an ambient target context or nested target attributes. Target-independent
    fixed-width loops do not require a target. Code needing different vectorization
    targets must be separated into functions before this pass.

    Parameters
    ----------
    enable_vectorize : bool
        Whether vectorization is enabled.
        Will lower to scalar loop when it is turned off.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.VectorizeLoop()(before)

    TVMScript body fragments for contiguous float32 tensors:

    .. code-block:: python

        # Before
        for i in T.vectorized(4):
            B[i] = A[i]
        # After
        B[T.Ramp(0, 1, 4)] = A[T.Ramp(0, 1, 4)]
    """
    return _ffi_api.VectorizeLoop(enable_vectorize)  # type: ignore


def StorageRewrite():
    """Plan storage by moving allocations and reusing compatible storage.

    Hoists allocations where valid and shares storage when lifetimes and
    memory-scope allocation properties permit it.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.StorageRewrite()(before)

    TVMScript allocation fragments when equally sized allocations have
    non-overlapping lifetimes and the memory scope permits reuse. The omitted
    uses of ``A`` finish before ``B`` is allocated:

    .. code-block:: python

        # Before
        A = T.alloc_tensor(ty_args=[T.Tensor((128,), "float32")])
        # ... all uses of A ...
        B = T.alloc_tensor(ty_args=[T.Tensor((128,), "float32")])
        # After
        A = T.alloc_tensor(ty_args=[T.Tensor((128,), "float32")])
        # ... all uses of A ...
        B = T.decl_tensor(A.data, ty_args=[T.Tensor((128,), "float32")])
    """
    return _ffi_api.StorageRewrite()  # type: ignore


def PointerValueTypeRewrite():
    """Rewrite pointer and tensor element types to match their accesses.

    Adjusts argument and allocation types, shapes and accesses where possible
    to reduce backend casts, including packing scalar storage into vectors.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.PointerValueTypeRewrite()(before)

    TVMScript fragments for an allocation accessed only through aligned
    four-lane loads/stores. Other accesses and uses of ``A`` are omitted:

    .. code-block:: python

        # Before
        A = T.alloc_local((16,), "float32")
        A[T.Ramp(0, 1, 4)] = T.Broadcast(T.float32(1), 4)
        # After
        A = T.alloc_local((4,), "float32x4", layout=None)
        A[0] = T.Broadcast(T.float32(1), 4)
    """
    return _ffi_api.PointerValueTypeRewrite()  # type: ignore


def UnrollLoop():
    """Unroll loops with constant extents marked ``T.unroll``.

    The ``tirx.UnrollLoop`` PassContext configuration also controls automatic
    unrolling and whether to expand loops or retain unroll annotations.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.UnrollLoop()(before)

    TVMScript body fragments with the default explicit unrolling enabled:

    .. code-block:: python

        # Before
        for i in T.unroll(2):
            A[i] = 1
        # After
        A[0] = 1
        A[1] = 1
    """
    return _ffi_api.UnrollLoop()  # type: ignore


def RemoveNoOp():
    """Remove effect-free statements and provably empty control flow.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.RemoveNoOp()(before)

    TVMScript body fragments; the store remains because it has an effect:

    .. code-block:: python

        # Before
        T.evaluate(1)
        A[0] = 1
        # After
        A[0] = 1
    """
    return _ffi_api.RemoveNoOp()  # type: ignore


def HoistIf():
    """Hoist loop-invariant If statements outside eligible loops.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass.

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.HoistIf()(before)

    TVMScript body fragments where ``n`` is invariant in the loop:

    .. code-block:: python

        # Before
        for i in T.serial(4):
            if n > 0:
                A[i] = 1
        # After
        if 0 < n:
            for i in T.serial(4):
                A[i] = 1
    """
    return _ffi_api.HoistIf()


def BF16ComputeLegalize():
    """Promote unsupported BF16 computations and internal temporaries to float32.

    Inserts conversions at BF16 boundaries. Leaves functions unchanged when
    their CUDA target reports native BF16 support.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.BF16ComputeLegalize()(before)

    TVMScript fragments of an internal temporary used for computation,
    when the target lacks native BF16 support. Loads from BF16 inputs and
    stores to BF16 outputs also gain conversion expressions (omitted here):

    .. code-block:: python

        # Before
        B = T.alloc_tensor(ty_args=[T.Tensor((100,), "bfloat16")])
        # After
        B = T.alloc_tensor(ty_args=[T.Tensor((100,), "float32")])
    """
    return _ffi_api.BF16ComputeLegalize()  # type: ignore


def FP8ComputeLegalize(promote_dtype: str = "float32"):
    """Promote unsupported FP8 computations and internal temporaries.

    Inserts conversions at FP8 boundaries. Leaves functions unchanged when
    their CUDA target reports native FP8 support.

    Parameters
    ----------
    promote_dtype : str
        The data type we promote fp8 to, options: float16/float32.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.FP8ComputeLegalize("float16")(before)

    TVMScript fragments of an internal temporary used for computation,
    when the target lacks native FP8 support. Input/output conversion
    expressions are omitted:

    .. code-block:: python

        # Before
        C = T.alloc_tensor(ty_args=[T.Tensor((100,), "float8_e5m2")])
        # After
        C = T.alloc_tensor(ty_args=[T.Tensor((100,), "float16")])
    """
    return _ffi_api.FP8ComputeLegalize(promote_dtype)  # type: ignore


def BF16StorageLegalize():
    """Represent BF16 storage as uint16 while preserving its bit patterns.

    Run after compute legalization. Functions whose CUDA target reports
    native BF16 support are unchanged.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.BF16StorageLegalize()(before)

    TVMScript signature fragments after compute legalization, on a target
    without native BF16 support. The unchanged body is omitted:

    .. code-block:: python

        # Before
        @T.function
        def main(A: T.handle("bfloat16")):
            ...
        # After
        @T.function
        def main(A: T.handle("uint16")):
            ...
    """
    return _ffi_api.BF16StorageLegalize()  # type: ignore


def FP8StorageLegalize():
    """Represent FP8 storage as uint8 while preserving its bit patterns.

    Run after compute legalization. Functions whose CUDA target reports
    native FP8 support are unchanged.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.FP8StorageLegalize()(before)

    TVMScript signature fragments after compute legalization, on a target
    without native FP8 support. The unchanged body is omitted:

    .. code-block:: python

        # Before
        @T.function
        def main(A: T.handle("float8_e5m2")):
            ...
        # After
        @T.function
        def main(A: T.handle("uint8")):
            ...
    """
    return _ffi_api.FP8StorageLegalize()  # type: ignore


def CommonSubexprElim():
    """Replace redundant computations by new variables.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.CommonSubexprElim()(before)

    TVMScript body fragments; generated variable names may differ:

    .. code-block:: python

        # Before
        B[0] = x + y
        B[1] = x + y
        # After
        cse_v1 = T.bind(x + y)
        B[0] = cse_v1
        B[1] = cse_v1
    """
    return _ffi_api.CommonSubexprElim()  # type: ignore


def RenormalizeSplitPattern():
    """Renormalize the split pattern from floordiv(floormod()) to floormod(floordiv())

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.RenormalizeSplitPattern()(before)

    TVMScript body fragments for integer ``i``; 128 is divisible by 32:

    .. code-block:: python

        # Before
        A[0] = (i % 128) // 32
        # After
        A[0] = (i // 32) % 4
    """
    return _ffi_api.RenormalizeSplitPattern()  # type: ignore


def StmtSimplify():
    """Run statement-level arithmetic simplifications on the TIR Function.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.StmtSimplify()(before)

    TVMScript body fragments; arithmetic is simplified using loop bounds:

    .. code-block:: python

        # Before
        for i in T.serial(4):
            A[i] = i // 4
        # After
        for i in T.serial(4):
            A[i] = 0
    """
    return _ffi_api.StmtSimplify()  # type: ignore


def LowerThreadAllreduce():
    """Lower ``gpu_thread_allreduce`` to target-specific communication.

    Requires a function ``target`` attribute and explicit thread extents.
    Selects warp shuffles or shared-memory reductions as supported.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass.

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.LowerThreadAllreduce()(before)

    TVMScript fragments for the degenerate one-thread reduction. Here
    ``combine = T.Lambda([T.float32, T.float32], lambda x, y: (x + y,))``
    and ``R`` is a one-element local float32 tensor; the function has a CUDA
    target. Larger reductions generate shuffles or shared-memory communication:

    .. code-block:: python

        # Before
        with T.launch_thread("threadIdx.x", 1) as tx:
            T.gpu_thread_allreduce(combine, (T.float32(0),), (A[tx],), True, (R[0],), (tx,))
        # After
        with T.launch_thread("threadIdx.x", 1) as tx:
            R[0] = A[tx]
    """
    return _ffi_api.LowerThreadAllreduce()


def ConvertSSA():
    """Give every variable definition in an IRModule a distinct identity.

    This pass handles cases where the same `tirx.Var` appears in
    multiple functions within the same module.  For example, after
    extracting a fragment from one function into another, where the
    same `tirx.Var` may be defined both as within the body of the
    original function, and as a parameter within the hoisted function.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.ConvertSSA()(before)

    TVMScript body fragments with an explicitly reused variable
    ``v = T.dynamic("v", "int32")``. Construct the input with
    ``@I.ir_module(check_well_formed=False)`` because it is deliberately
    not SSA. Fresh parsed bindings normally need no change:

    .. code-block:: python

        # Before
        T.bind(16, var=v)
        T.evaluate(v)
        T.bind(32, var=v)
        T.evaluate(v)
        # After
        v = T.bind(16)
        T.evaluate(v)
        v_1 = T.bind(32)
        T.evaluate(v_1)
    """
    return _ffi_api.ConvertSSA()  # type: ignore


def MakePackedAPI():
    """Expose exported functions through the TVM FFI packed-call ABI.

    Requires a ``target`` attribute with a host target on functions with a
    ``global_symbol`` and the default calling convention. Private functions,
    functions with other calling conventions, and functions whose target has
    no host are left unchanged.

    Tensor parameter types generate runtime checks for shape, dtype and
    layout. Symbolic dimensions not defined by other arguments are bound from
    the incoming DLTensor metadata. For example, a ``T.Tensor((16, 32),
    "float32")`` argument is checked against that shape, while symbolic
    dimensions are read from the argument's shape. The resulting function
    uses the host target and an exported ``__tvm_ffi_`` symbol.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.MakePackedAPI()(before)

    TVMScript signature fragments for an exported function with a host
    target. Argument decoding, validation, the compute body and return
    handling are omitted:

    .. code-block:: python

        # Before
        @T.function
        def main(A: T.Tensor((16,), "float32")):
            T.func_attr({"target": T.target("llvm", host="llvm")})
            ...
        # After
        @T.function
        def main(self_handle: T.Ptr(T.void), args: T.Ptr(T.void),
                 num_args: T.int32, result: T.Ptr(T.void)) -> T.int32:
            T.func_attr({"global_symbol": "__tvm_ffi_main", "calling_conv": 1,
                         "target": T.target("llvm")})
            ...
    """
    return _ffi_api.MakePackedAPI()  # type: ignore


def SplitHostDevice():
    """Annotate, split, and lower host/device functions.

    This pass first annotates device regions within host functions,
    then splits them into host and device-side Functions, and finally
    lowers host-to-device calls into the device kernel launch ABI.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.SplitHostDevice()(before)

    TVMScript fragments for a function named ``main`` with target
    ``T.target("cuda", host="llvm")``. Generated function attributes are
    omitted; the device function receives the region's free variable ``n``:

    .. code-block:: python

        # Before: body of main
        with T.region("tirx.device_scope", []):
            T.evaluate(n)
        # After: host body of main
        T.call_ffi_kernel("main_kernel", n)
        # After: extracted device function
        @T.function
        def main_kernel(n: T.int32):
            T.evaluate(n)
    """
    return _ffi_api.SplitHostDevice()  # type: ignore


def SkipAssert():
    """Replace assertion statements with no-ops.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.SkipAssert()(before)

    TVMScript body fragments; no-op cleanup is a separate pass:

    .. code-block:: python

        # Before
        assert n > 0, "positive extent required"
        A[0] = 1
        # After
        T.evaluate(0)
        A[0] = 1
    """
    return _ffi_api.SkipAssert()  # type: ignore


def LowerTVMBuiltin():
    """Lower runtime builtins to explicit stack storage and ABI operations.

    For example, packed calls gain argument/result slots. Requires the
    function target attribute for target-dependent lowering.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.LowerTVMBuiltin()(before)

    TVMScript call fragments in an LLVM-targeted function. The generated
    ``stack_ffi_any`` allocation and argument/result slot initialization
    are omitted:

    .. code-block:: python

        # Before
        T.call_packed("callback", 1)
        # After
        T.call_packed_lowered("callback", stack_ffi_any, 0, 1)
    """
    return _ffi_api.LowerTVMBuiltin()  # type: ignore


def LowerIntrin():
    """Lower intrinsic calls using rules registered for the function target.

    Requires a function ``target`` attribute. Also lowers logical tensor
    pointer operations to address operations.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.LowerIntrin()(before)

    TVMScript body fragments with the function target set to LLVM:

    .. code-block:: python

        # Before
        A[0] = x // 8
        # After
        A[0] = T.shift_right(x, 3)
    """
    return _ffi_api.LowerIntrin()  # type: ignore


def NarrowDataType(target_bits: int):
    """Narrow integer indices and loop variables when bounds prove it safe.

    Keeps wider types when the analyzed range does not fit ``target_bits``.

    Parameters
    ----------
    target_bits : int
        The target bit configuration.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Note
    ----
    Run this pass after FlattenTensor.

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.NarrowDataType(32)(before)

    TVMScript body fragments for a flat tensor; the bounded loop index
    and index arithmetic fit in int32:

    .. code-block:: python

        # Before
        for i in T.serial(T.int64(4)):
            A[i] = T.float32(1)
        # After
        for i in T.serial(4):
            A[i] = T.float32(1)
    """
    return _ffi_api.NarrowDataType(target_bits)  # type: ignore


def ForceNarrowIndexToInt32():
    """Force index expressions to int32 without range proofs.

    Use only when the caller can guarantee representability. Unlike
    :py:func:`NarrowDataType`, this pass may change overflowing computations.
    Rejects signed integer tensor parameters and ordinary allocations wider
    than int32. Explicitly typed local scalars retain their storage type and
    are cast at narrowed index uses.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Note
    ----
    This pass should not be used in default cases.

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.ForceNarrowIndexToInt32()(before)

    TVMScript body fragments. The caller must ensure that the narrowed
    indices fit in int32:

    .. code-block:: python

        # Before
        for i in T.serial(T.int64(4)):
            A[i] = T.float32(1)
        # After
        for i in T.serial(4):
            A[i] = T.float32(1)
    """
    return _ffi_api.ForceNarrowIndexToInt32()  # type: ignore


def VerifyMemory():
    """Reject illegal host-side access to device memory.

    Returns the module unchanged on success and raises an error on invalid
    access. Functions without a target attribute are skipped, as are
    functions with a non-default calling convention.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.VerifyMemory()(before)

    This is a validator, so valid IR is unchanged. TVMScript body
    fragments in a CUDA-targeted function with tensor argument ``A``:

    .. code-block:: python

        # Before
        with T.launch_thread("threadIdx.x", 4) as tx:
            A[tx] = T.float32(1)
        # After: identical
        with T.launch_thread("threadIdx.x", 4) as tx:
            A[tx] = T.float32(1)
    """
    return _ffi_api.VerifyMemory()  # type: ignore


def FlattenTensor():
    """Flatten tensor allocations and accesses to one-dimensional storage.

    Uses shapes, strides and offsets to compute linear indices. Tensor
    parameters retain their interface types and gain flat views as needed.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.FlattenTensor()(before)

    TVMScript body fragments for a compact row-major local tensor:

    .. code-block:: python

        # Before
        A = T.alloc_local((2, 4), "float32")
        A[1, 2] = T.float32(1)
        # After
        A = T.alloc_local((8,), "float32", layout=None)
        A[6] = T.float32(1)
    """
    return _ffi_api.FlattenTensor()  # type: ignore


def BindTarget(target):
    """Bind targets and classify host/device helper functions.

    Preserves existing device targets and updates host information on exported
    functions. Helpers used on both sides may be cloned for the host.

    Device helper calls must be inside explicit launch_thread or device_entry
    regions. Lower S-TIR thread bindings before classifying host/device calls.

    Parameters
    ----------
    target : tvm_tirx.target.Target
        target

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.BindTarget(tvm_tirx.target.Target("cuda"))(before)

    TVMScript body fragments for a function without a target attribute:

    .. code-block:: python

        # Before
        T.evaluate(0)
        # After
        T.func_attr({"target": T.target("cuda")})
        T.evaluate(0)
    """
    return _ffi_api.BindTarget(target)  # type: ignore


def AnnotateEntryFunc():
    """Infer and mark an unambiguous TIRx entry function.

    Sets ``tirx.is_entry_func`` when the module contains only one function
    and it is TIRx, or when exactly one function is externally exposed and
    it is a TIRx function.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.AnnotateEntryFunc()(before)

    TVMScript body fragments when the module contains only this function:

    .. code-block:: python

        # Before
        T.evaluate(0)
        # After
        T.func_attr({"tirx.is_entry_func": True})
        T.evaluate(0)
    """
    return _ffi_api.AnnotateEntryFunc()  # type: ignore


def Filter(fcond: Callable):
    """Remove TIRx functions for which ``fcond(function)`` is false.

    The predicate receives a Function and returns a boolean.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Given an IRModule ``before``::

        after = tvm_tirx.tirx.transform.Filter(lambda f: False)(before)

    TVMScript module fragments; this predicate removes every TIRx function:

    .. code-block:: python

        # Before
        @I.ir_module
        class Before:
            @T.function(private=True)
            def main():
                T.evaluate(0)
        # After
        @I.ir_module
        class After:
            pass
    """
    return _ffi_api.Filter(fcond)  # type: ignore


def TileDispatch():
    """Lower TIRx tile calls using the active backend dispatch table.

    Resolves tile instructions and device launch/execution scopes. Select
    the backend with a target context; follow with :py:func:`LowerTensorLayouts`
    or the backend's own layout-lowering pass.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Given an IRModule ``before``::

        with tvm_tirx.target.Target("cuda"):
            after = tvm_tirx.tirx.transform.TileDispatch()(before)

    TVMScript body fragments showing launch normalization, one part of
    dispatch. The CUDA backend records the six launch-field names in the
    region attributes:

    .. code-block:: python

        # Before
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        T.evaluate(0)
        # After
        with T.device_scope(1, 1, 1, 32, 1, 1, attrs={
            "cuda.kernel_attrs": {},
            "cuda.launch_fields": ["grid.x", "grid.y", "grid.z", "block.x", "block.y", "block.z"],
        }):
            T.evaluate(0)
    """
    return _ffi_api.TileDispatch()  # type: ignore


def LowerTensorLayouts():
    """Apply tensor layouts to physical storage and accesses.

    Run after :py:func:`TileDispatch` for generic tile compilation, or
    independently on already dispatched IR. Backends may pair TileDispatch
    with their own layout-lowering pass.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass

    Examples
    --------
    Given an IRModule ``before``::

        with tvm_tirx.target.Target("cuda"):
            after = tvm_tirx.tirx.transform.LowerTensorLayouts()(before)

    TVMScript body fragments for a layout with gaps. The physical
    allocation spans offsets 0 through 38, rather than just 12 elements:

    .. code-block:: python

        # Before
        A = T.alloc_tensor(ty_args=[T.Tensor(
            (3, 4), "uint32", scope="local", layout=T.TileLayout(T.S[(3, 4):(16, 2)]))])
        A[2, 3] = T.uint32(7)
        # After
        A = T.alloc_local((39,), "uint32", layout=None)
        A[38] = T.uint32(7)
    """
    return _ffi_api.LowerTensorLayouts()  # type: ignore
