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


def LoopPartition(split_const_loop):
    """Inject virtual thread loops.

    Parameters
    ----------
    split_const_loop : bool
        Flag to enable partition for const loop.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LoopPartition(split_const_loop)


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


def InjectDoubleBuffer(split_loop_factor):
    """Inject double buffer statements.

    Parameters
    ----------
    split_loop_factor : int
        Loop splitting factor.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectDoubleBuffer(split_loop_factor)


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


def UnrollLoop(auto_max_step,
               auto_max_depth,
               auto_max_extent,
               explicit_unroll):
    """Unroll the constant loop marked by unroll.

    This pass also automatically attach pragma unroll tag to loops which meets the standard.

    Parameters
    ----------
    auto_max_step : int
        The maximum step before stop attach automatic unroll

    auto_max_depth : int
        The maximum depth before stop attach automatic unroll

     auto_max_extent : int
        The maximum extent of the loop we can unroll.
        This is an legacy option that do not take the loop total steps into account.

    explicit_unroll : bool
        Whether explicitly unroll the loop, or leave unroll annotation to codegen.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.UnrollLoop(
        auto_max_step, auto_max_depth, auto_max_extent, explicit_unroll)


def RemoveNoOp():
    """Remove No Op from the Stmt.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.RemoveNoOp()


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
    """ Insert sync between parallel read/write of shared buffers.

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
    """ Infer the TensorCore fragment infomation using tensor intrinsics.

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
