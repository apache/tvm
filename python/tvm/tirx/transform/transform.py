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

import tvm_ffi as _ffi

from . import _ffi_api
from .function_pass import function_pass


def Apply(ftransform):
    """Apply ftransform to each function in the Module.

    This function is a thin wrapper around tvm.tirx.transform.function_pass

    Parameters
    ----------
    ftransform: tvm.tirx.Function -> tvm.tirx.Function
       The transformation pass.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
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
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.VectorizeLoop(enable_vectorize)  # type: ignore


def StorageRewrite():
    """Rewrite storage allocation pattern.

    Moves the allocation to outer most possible scope.
    Trying to share space between allocations to make
    a static allocation plan when possible.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.StorageRewrite()  # type: ignore


def InlinePrivateFunctions():
    """Inline calls to private functions

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InlinePrivateFunctions()  # type: ignore


def PointerValueTypeRewrite():
    """
    Rewrite the pointer content type of arguments, as well as Alloc internal to the function to use
    the most frequently accessed type for load/store to avoid pointer casting in backend when
    possible.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.PointerValueTypeRewrite()  # type: ignore


@_ffi.register_object("tirx.transform.UnrollLoopConfig")
class UnrollLoopConfig(_ffi.Object):
    """Config for unroll loop pass"""


def UnrollLoop():
    """Unroll the constant loop marked by unroll.

    This pass also automatically attach pragma unroll tag to loops which meets the standard.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.UnrollLoop()  # type: ignore


@_ffi.register_object("tirx.transform.RemoveNoOpConfig")
class RemoveNoOpConfig(_ffi.Object):
    """Config for remove no op pass"""


def RemoveNoOp():
    """Remove No Op from the Stmt.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.RemoveNoOp()  # type: ignore


def RemoveAssume():
    """Remove all instances of prim::assume_op

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.RemoveAssume()  # type: ignore


def BF16ComputeLegalize():
    """Legalize bf16 compute Ops.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.BF16ComputeLegalize()  # type: ignore


def FP8ComputeLegalize(promote_dtype: str = "float32"):
    """Legalize fp8 compute Ops.

    Parameters
    ----------
    promote_dtype : str
        The data type we promote fp8 to, options: float16/float32.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.FP8ComputeLegalize(promote_dtype)  # type: ignore


def BF16StorageLegalize():
    """Legalize bf16 storage types to u16.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.BF16StorageLegalize()  # type: ignore


def FP8StorageLegalize():
    """Legalize fp8 storage types to u8.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.FP8StorageLegalize()  # type: ignore


def CommonSubexprElim():
    """Replace redundant computations by new variables.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.CommonSubexprElim()  # type: ignore


def RenormalizeSplitPattern():
    """Renormalize the split pattern from floordiv(floormod()) to floormod(floordiv())

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.RenormalizeSplitPattern()  # type: ignore


@_ffi.register_object("tirx.transform.StmtSimplifyConfig")
class StmtSimplifyConfig(_ffi.Object):
    """Config for stmt simplify pass"""


def StmtSimplify():
    """Run statement-level arithmetic simplifications on the TIR Function.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.StmtSimplify()  # type: ignore


def LowerThreadAllreduce():
    """Lower cross-thread reductions to target-specific communication.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass.
    """
    return _ffi_api.LowerThreadAllreduce()


def ConvertSSA():
    """Convert an IRModule to be SSA form.

    This pass handles cases where the same `tirx.Var` appears in
    multiple functions within the same module.  For example, after
    extracting a fragment from one function into another, where the
    same `tirx.Var` may be defined both as within the body of the
    original function, and as a parameter within the hoisted function.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass

    """
    return _ffi_api.ConvertSSA()  # type: ignore


def MakePackedAPI():
    """Transform the Functions in the module to a packed func API.

    Prior to this pass, the Function may have parameters annotated with
    `TensorType`.  This pass consumes those annotations to generate
    arguments that implement the packed based TVM FFI API.

    For static shapes, the `TensorType::shape`, `TensorType::strides`,
    and `TensorType::elem_offset` fields are used to
    generate runtime checks on the corresponding member variables in
    the user-provided `DLTensor*` or `tvm.runtime.tensor` argument.  (e.g. A
    Function that accepts a buffer of shape `[16,32]` validates that
    the `DLTensor::shape` array is `[16,32]`.)

    For dynamic Buffers, in which one or more of these `TensorType` fields
    use `tirx.Var` that are not defined by other Function
    parameters, these are instead used to define the variables based on
    the corresponding `DLTensor` members.  (e.g. A Function that accepts a
    buffer of shape `[tirx.Var("n", "int64"), tirx.Var("m", "int64")]`,
    when passed a `DLTensor` of shape `[16, 32]`, will define `n = 16` and
    `m = 32`, based on the argument's shape.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MakePackedAPI()  # type: ignore


def SplitHostDevice():
    """Annotate, split, and lower host/device functions.

    This pass first annotates device regions within host functions,
    then splits them into host and device-side Functions, and finally
    lowers host-to-device calls into the device kernel launch ABI.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.SplitHostDevice()  # type: ignore


def SkipAssert():
    """Skip assert stmt.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.SkipAssert()  # type: ignore


def LowerWarpMemory():
    """Lower warp memory access to low-level device related function calls.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerWarpMemory()  # type: ignore


def LowerTVMBuiltin():
    """Lower tvm builtin intrinsics.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerTVMBuiltin()  # type: ignore


def LowerIntrin():
    """Lower target specific intrinsic calls.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerIntrin()  # type: ignore


def NarrowDataType(target_bits: int):
    """Narrow down Expr datatype in stmt to target_bits.

    Parameters
    ----------
    target_bits : int
        The target bit configuration.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass

    Note
    ----
    Run this pass after FlattenBuffer.
    """
    return _ffi_api.NarrowDataType(target_bits)  # type: ignore


def ForceNarrowIndexToInt32():
    """Force narrow down indexing expressions and integer buffers to int32 dtype.

    The function must not contain S-TIR blocks. Use
    :py:func:`tvm.s_tir.transform.ForceNarrowIndexToInt32` before block lowering.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass

    Note
    ----
    This pass should not be used in default cases.
    """
    return _ffi_api.ForceNarrowIndexToInt32()  # type: ignore


def VerifyMemory():
    """Verify if func contains illegal host side direct memory access.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.VerifyMemory()  # type: ignore


def FlattenBuffer():
    """Flatten the multi-dimensional TensorLoad and TensorStore to single dimensional
    TensorLoad/TensorStore for the TIR not contains opaque block.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.FlattenBuffer()  # type: ignore


def BindTarget(target):
    """Annotate a Function with a given target.
    Parameters
    -------
    target : tvm.target.Target
        target

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.BindTarget(target)  # type: ignore


def AnnotateEntryFunc():
    """Set a Function as the entry point if it is only function in IRModule.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.AnnotateEntryFunc()  # type: ignore


def Filter(fcond: Callable):
    """Filter out Functions that does not satisfy the given condition.
    `fcond` should be a function that takes a function and returns boolean.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.Filter(fcond)  # type: ignore


def TilePrimitiveDispatch():
    """Lower TIRx tile primitive calls through the active backend dispatch table.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.TilePrimitiveDispatch()  # type: ignore


def LowerTIRx():
    """Lower TIR to a lower-level IR.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerTIRx()  # type: ignore


def LowerTIRxOpaque():
    """Lower opaque constructs in TIRX programs.

    Handles allocation call lowering, For(thread_binding) to RegionStmt(launch_thread)
    conversion, unit loop elimination, and pragma annotation handling.
    This is the tirx-specific counterpart of s_tir.LowerOpaqueBlock and
    s_tir.LowerThreadBinding,
    without any SBlock/SBlockRealize handling.

    Run s_tir.transform.LoopPartition first when thread-binding loops carry
    loop_partition_hint.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerTIRxOpaque()  # type: ignore
