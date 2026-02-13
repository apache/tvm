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


import enum
from typing import Callable

from . import _ffi_api
from . import function_pass as _fpass
from ... import ir as _ir
from ... import ffi as _ffi


def Apply(ftransform):
    """Apply ftransform to each function in the Module.

    This function is a thin wrapper around tvm.tir.transform.prim_func_pass

    Parameters
    ----------
    ftransform: tvm.tir.PrimFunc -> tvm.tir.PrimFunc
       The transformation pass.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """

    # pylint: disable=unused-argument
    def _transform(func, mod, ctx):
        return ftransform(func)

    return _fpass.prim_func_pass(_transform, opt_level=0, name="Apply")  # type: ignore


def VectorizeLoop(enable_vectorize: bool = True):
    """Lower vectorization loops.

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


@_ffi.register_object("tir.transform.UnrollLoopConfig")
class UnrollLoopConfig(_ir.Attrs):
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


@_ffi.register_object("tir.transform.ReduceBranchingThroughOvercomputeConfig")
class ReduceBranchingThroughOvercomputeConfig(_ir.Attrs):
    """Config for reduce branching through overcompute pass"""


def ReduceBranchingThroughOvercompute():
    """Reduce branching by introducing overcompute

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.ReduceBranchingThroughOvercompute()  # type: ignore


@_ffi.register_object("tir.transform.RemoveNoOpConfig")
class RemoveNoOpConfig(_ir.Attrs):
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
    """Remove all instances of builtin::assume

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


def CommonSubexprElimTIR(enable_cse_tir: bool = True, identify_equiv_terms: bool = False):
    """Replace redundant computations by new variables.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.CommonSubexprElimTIR(enable_cse_tir, identify_equiv_terms)  # type: ignore


@_ffi.register_object("tir.transform.SimplifyConfig")
class SimplifyConfig(_ir.Attrs):
    """Config for simplify pass"""


def Simplify():
    """Run arithmetic simplifications on the statements and expressions.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.Simplify()  # type: ignore


def ConvertSSA():
    """Convert an IRModule to be SSA form.

    This pass handles cases where the same `tir.Var` appears in
    multiple functions within the same module.  For example, after
    extracting a fragment from one function into another, where the
    same `tir.Var` may be defined both as within the body of the
    original function, and as a parameter within the hoisted function.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass

    """
    return _ffi_api.ConvertSSA()  # type: ignore


def LowerCustomDatatypes():
    """Lower custom datatypes.

    See tvm::datatypes::Registry for more information on adding custom datatypes.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerCustomDatatypes()  # type: ignore


def MakePackedAPI():
    """Transform the PrimFuncs in the module to a packed func API.

    Prior to this pass, the PrimFunc may have Buffer arguments defined
    in the `PrimFuncNode::buffer_map`.  This pass consumes the
    `buffer_map`, using it to generate arguments that implement
    the packed based TVM FFI API.

    For static shapes, the `BufferNode::shape`, `BufferNode::strides`,
    and `BufferNode::elem_offset` member variables are used to
    generate runtime checks on the corresponding member variables in
    the user-provided `DLTensor*` or `tvm.runtime.tensor` argument.  (e.g. A
    PrimFunc that accepts a buffer of shape `[16,32]` validates that
    the `DLTensor::shape` array is `[16,32]`.)

    For dynamic Buffers, in which one or more of these `BufferNode` member
    variables use `tir.Var` that are not defined by other PrimFunc
    parameters, these are instead used to define the variables based on
    the corresponding `DLTensor` members.  (e.g. A PrimFunc that accepts a
    buffer of shape `[tir.Var("n"), tir.Var("m")]`, when passed a
    `DLTensor` of shape `[16,32]`, will define `n = 16` and `n=32`, based
    on the argument's shape.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MakePackedAPI()  # type: ignore


def MakeUnpackedAPI():
    """Transform the PrimFuncs in the module to a C API compatible with internal calls.

    Prior to this pass, the PrimFunc may have Buffer arguments defined in
    the `PrimFuncNode::buffer_map`.  This pass consumes the `buffer_map`,
    using it to generate `T*` arguments (e.g. `float32*`) that can be
    directly called by a C API.

    For static shapes, no runtime validation is performed to confirm that
    the argument buffer's shape matches the expected shape.  For dynamic
    shapes, `MakeUnpackedAPI` requires that the dynamic parameters be
    passed as separate `tir.Var` parameters.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MakeUnpackedAPI()  # type: ignore


def AnnotateDeviceRegions():
    """Annotate locations that should be run on the device

    Insert `AttrStmt` nodes specifying a target on which regions
    within the PrimFunc should be executed.  Only modifies functions
    that have a `tvm::attr::kTarget` attribute, and where that target
    defines a host.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.AnnotateDeviceRegions()  # type: ignore


def SplitHostDevice():
    """Split the function into a host function and device functions.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.SplitHostDevice()  # type: ignore


def LowerDeviceKernelLaunch():
    """Lower cross-device function calls.

    Prior to this pass, host to device calls are represented as
    subroutine calls, with environment parameters (e.g. env_thread)
    specified internally.  The device function is an internal
    function, without a `tvm::attr::kGlobalSymbol` attribute.

    After this pass, host to device calls are represented as
    tvm_call_packed built-in.  The device function is an
    externally-exposed function, with a non-empty
    `tvm::attr::kGlobalSymbol` attribute.


    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerDeviceKernelLaunch()  # type: ignore


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


def LowerDeviceStorageAccessInfo():
    """Lower attached storage access information on device.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass

    Note
    ----
    Run this pass after all storage access analysis finish.
    """
    return _ffi_api.LowerDeviceStorageAccessInfo()  # type: ignore


def CombineContextCall():
    """Combine context calls in the host function.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.CombineContextCall()  # type: ignore


def NarrowDataType(target_bits: int):
    """Narrow down PrimExpr datatype in stmt to target_bits.

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


@_ffi.register_object("s_tir.transform.HoistIfThenElseConfig")
class HoistIfThenElseConfig(_ir.Attrs):
    """Config for hoist if then else pass"""


class HoistedConditionals(enum.Flag):
    """Flags for use in HoistExpressionConfig.conditional_types

    Each bitflag represents a type of expression that should be
    hoisted to the outermost loop possible.
    """

    Never = 0
    """ No hoisting of conditionals """

    IfElseStmt = 1
    """ If set, look for hoist candidates in IfElseStmt """

    IfElseExpr = 2
    """ If set, look for hoist candidates in tir.if_then_else """

    BooleanExpression = 4
    """ If set, look for hoist candidates in all boolean expressions """

    UsingBlockVar = 8
    """ If set, allow hoisting of conditionals that use a block variable (e.g. threadIdx.x)  """

    All = IfElseStmt | IfElseExpr | BooleanExpression | UsingBlockVar
    """ Enable all hoisting of conditionals"""


class HoistedLetBindings(enum.Flag):
    """Flags for use in HoistExpressionConfig.let_binding_types

    Each bitflag represents a type of let binding expression that should be
    hoisted to the outermost loop possible.
    """

    Never = 0
    """ No hoisting of let bindings """

    RequiredByConditional = 1
    """ Bindings that are used by a hoisted conditional """

    LetStmt = 2
    """ Bindings occurring in LetStmt """

    LetExpr = 4
    """ Bindings occurring in Let expressions """

    All = RequiredByConditional | LetStmt | LetExpr
    """ Enable all hoisting of let bindings """


@_ffi.register_object("s_tir.transform.HoistExpressionConfig")
class HoistExpressionConfig(_ir.Attrs):
    """Config for hoist expression pass"""


def FlattenBuffer():
    """Flatten the multi-dimensional BufferLoad and BufferStore to single dimensional
    BufferLoad/BufferStore for the TIR not contains opaque block.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.FlattenBuffer()  # type: ignore


def ConvertForLoopsToSerial():
    """Convert Parallel For Loops to Serial For Loops.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.ConvertForLoopsToSerial()  # type: ignore


def BindTarget(target):
    """Annotate a PrimFunc with a given target.
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
    """Set a PrimFunc as the entry point if it is only function in IRModule.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.AnnotateEntryFunc()  # type: ignore


def Filter(fcond: Callable):
    """Filter out PrimFuncs that does not satisfy the given condition.
    `fcond` should be a function that takes a primfunc and returns boolean.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.Filter(fcond)  # type: ignore
