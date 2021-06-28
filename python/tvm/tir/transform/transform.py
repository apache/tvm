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
# pylint: disable=invalid-name
from . import _ffi_api
from . import function_pass as _fpass


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

    return _fpass.prim_func_pass(_transform, opt_level=0, name="Apply")


def Filter(fcond):
    """Filter functions by the calling convention attribute.

    Parameters
    ----------
    fcond : tvm.tir.PrimFunc -> bool
        The condition of the filtering.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    # pylint: disable=unused-argument
    def _transform(func, mod, ctx):
        return func if fcond(func) else None

    return _fpass.prim_func_pass(_transform, opt_level=0, name="Filter")


def InjectPrefetch():
    """Inject prefetch instructions into stmt.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectPrefetch()


def StorageFlatten(cache_line_size, create_bound_attribute=False):
    """Flatten the multi-dimensional read/write to 1D.


    Parameters
    ----------
    cache_line_size: int
        The size of CPU cache line.

    create_bound_attribute:
        Whether to create bound attributes.


    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.StorageFlatten(cache_line_size, create_bound_attribute)


def InjectCopyIntrin(pragma_key, fintrin):
    """Inject virtual thread loops.

    Parameters
    ----------
    pragma_key : str
        The pragma key for hint of copy.

    fintrin : function
        The function with signature copyintrin(src, dst, pad_before, pad_after, pad_value)

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectCopyIntrin(pragma_key, fintrin)


def CoProcSync():
    """Detect and insert sync points to co-processor.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.CoProcSync()


def LiftAttrScope(attr_key):
    """Lift common attrs with attr_key to outer scope.

    Parameters
    ----------
    attr_key : str
        The attribute key to be checked.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LiftAttrScope(attr_key)


def LoopPartition():
    """Inject virtual thread loops.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LoopPartition()


def VectorizeLoop(enable_vectorize=True):
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
    return _ffi_api.VectorizeLoop(enable_vectorize)


def InjectVirtualThread():
    """Inject virtual thread loops.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectVirtualThread()


def InjectDoubleBuffer():
    """Inject double buffer statements.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectDoubleBuffer()


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
    return _ffi_api.StorageRewrite()


def UnrollLoop():
    """Unroll the constant loop marked by unroll.

    This pass also automatically attach pragma unroll tag to loops which meets the standard.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.UnrollLoop()


def RemoveNoOp():
    """Remove No Op from the Stmt.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.RemoveNoOp()


def BF16Legalize():
    """Legalize bf16 typed Ops.
    Runs BF16Promote, BF16CastElimination and BF16TypeLowering

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.BF16Legalize()


def BF16Promote():
    """Promote bf16 to fp32. Add a cast to fp32
    before Ops, then add a cast back to bf16.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.BF16Promote()


def BF16CastElimination():
    """Eliminate verbose casting between fp32 and bf16
    Checks if the AST has the pattern:
    castto32(castto16(some_fp32_op(...)))
    The verbose casting is generated by BF16Promote for multiple
    bf16 Ops in a row. e.g.:
    X[i] + Y[i] + T[i] =>
    bf16((float32(bf16((float32(X[i]) + float32(Y[i])))) + float32(T[i])))
    After this pass:
    bf16(float32(X[i]) + float32(Y[i]) + float32(T[i]))

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.BF16CastElimination()


def BF16TypeLowering():
    """Replace all bf16 type with uint16. Also lower the casting
    between fp32 and bf16

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.BF16TypeLowering()


def RewriteUnsafeSelect():
    """Detect and rewrite unsafe select that contains memory access.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.RewriteUnsafeSelect()


def Simplify():
    """Run arithmetic simplifications on the statements and expressions.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.Simplify()


def InstrumentBoundCheckers():
    """Instruments bound checkers.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InstrumentBoundCheckers()


def LowerCustomDatatypes():
    """Lower custom datatypes.

    See tvm::datatypes::Registry for more information on adding custom datatypes.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerCustomDatatypes()


def MakePackedAPI(num_unpacked_params=0):
    """Transform the PrimFuncs in the module to a packed func API.

    Parameters
    ----------
    num_unpacked_params : int
        Number of parameters that we hope to directly pass via normal arguments
        following the PackedFunc input signature.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MakePackedAPI(num_unpacked_params)


def MakeUnpackedAPI():
    """Transform the PrimFuncs in the module to a C API compatible with internal calls.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MakeUnpackedAPI()


def SplitHostDevice():
    """Split the function into a host function and device functions.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.SplitHostDevice()


def DecorateDeviceScope():
    """Decorate all the function's body as device function.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.DecorateDeviceScope()


def SkipAssert():
    """Skip assert stmt.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.SkipAssert()


def ThreadSync(storage_scope):
    """Insert sync between parallel read/write of shared buffers.

    Parameters
    ----------
    storage_scope: str
        The target storage scope.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.ThreadSync(storage_scope)


def LowerThreadAllreduce():
    """Lower cross thread alleduce.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerThreadAllreduce()


def InferFragment():
    """Infer the TensorCore fragment infomation using tensor intrinsics.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InferFragment()


def LowerWarpMemory():
    """Lower warp memory access to low-level device related function calls.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerWarpMemory()


def LowerTVMBuiltin():
    """Lower tvm builtin intrinsics.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerTVMBuiltin()


def LowerIntrin():
    """Lower target specific intrinsic calls.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerIntrin()


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
    return _ffi_api.LowerDeviceStorageAccessInfo()


def CombineContextCall():
    """Combine context calls in the host function.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.CombineContextCall()


def NarrowDataType(target_bits):
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
    Run this pass after StorageFlatten.
    """
    return _ffi_api.NarrowDataType(target_bits)


def VerifyMemory():
    """Verify if func contains illegal host side direct memory access.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.VerifyMemory()


# pylint: disable=no-else-return,inconsistent-return-statements
def HoistIfThenElse(variant=None):
    """Hoist loop-invariant IfThenElse nodes to outside the elligible loops.

    Parameters
    ----------
    variant : Optional[String]
        The variant of the pass.
        variant can have any one of following values ["basic", None(Default)].

        The basic variant supports basic hoisting scenarios where it exepects
        the For & If Nodes are in place consecutively and does not involve
        global scope variables or more advanced scenarios.

        Default variant supports all hoisting scenarios,i.e., {"Basic" + "Advanced"}
        supported with control with PassContext configs like below:

            config={"tir.HoistIfThenElse": {"support_block_scope_hosting": True}}

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    if variant == "basic":
        return _ffi_api.HoistIfThenElseBasic()
    elif variant is None:
        return _ffi_api.HoistIfThenElse()


def LowerInitBlock():
    """Lower block init stmt into IfThenElse stmts

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerInitBlock()


def PlanAndUpdateBufferAllocationLocation():
    """Locate the buffer allocation to the exact position (usually is
    the lca of buffer access). This pass will inject opaque block
    with alloc_buffers at the allocation site.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.PlanAndUpdateBufferAllocationLocation()


def ConvertBlocksToOpaque():
    """Substitute all the block vars with the PrimExprs they are bound to, indicated by
    the corresponding iter_values in BlockRealize, and then convert the blocks into
    opaque ones by removing all the iter_values in BlockRealize and iter_vars in Block.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.ConvertBlocksToOpaque()


def CompactBufferAllocation():
    """Compact the buffer access region. by removing the buffer regions
    that are not accessed, i.e. narrowing the buffer shape and adjust
    the access region if necessary.

    Example
    -------

    Before narrowing, ``B`` is a ``[16, 16]`` buffer, but only a
    skinny vector ``B[i, 0:16]`` is accessed.

    .. code-block:: python

        for i in range(0, 16):
            with tir.block([]):
                B = tir.alloc_buffer(16, 16)
                for j in range(0, 16):
                    B[i, j] = A[i, j] + 1
                for j in range(0, 16):
                    C[i, j] = B[i, j] + 1

    This pass narrows the buffer shape and adjust its accessed region
    accordingly.  In this particular case, because only a ``1 * 16``
    vector of ``B`` is accessed, the pass narrows ``B`` to shape ``[1,
    16]``, and changes the access to ``B[i, j]`` to ``B[0, j]``.

    .. code-block:: python

        for i in range(0, 16):
            with tir.block([]):
                B = tir.alloc_buffer(1, 16)
                for j in range(0, 16):
                    B[0, j] = A[i, j] + 1
                for j in range(0, 16):
                    C[i, j] = B[0, j] + 1

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass

    """
    return _ffi_api.CompactBufferAllocation()


def FlattenBuffer():
    """Flatten the multi-dimensional BufferLoad and BufferStore
    to single dimensional Load/Store. Also remove Block to
    ensure that the flattened TIR can not be scheduled again.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.FlattenBuffer()
