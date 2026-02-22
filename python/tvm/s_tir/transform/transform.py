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
"""S-TIR specific transformations."""
# pylint: disable=invalid-name, unsupported-binary-operation

from ... import ffi as _ffi
from ... import ir as _ir
from . import _ffi_api


def CanonicalizeLoop():
    """Canonicalize the loop to start from zero and use trivial step

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.CanonicalizeLoop()  # type: ignore


def LowerCrossThreadReduction():
    """Lower cross-thread reduction from thread bindings to
    intrinsic function calls.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerCrossThreadReduction()  # type: ignore


def LowerInitBlock():
    """Lower block init stmt into IfThenElse statements.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerInitBlock()  # type: ignore


def PlanAndUpdateBufferAllocationLocation():
    """Locate the buffer allocation to the exact position (usually is
    the lca of buffer access). This pass will inject opaque block
    with alloc_buffers at the allocation site.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.PlanAndUpdateBufferAllocationLocation()  # type: ignore


def ConvertBlocksToOpaque():
    """Substitute all the block vars with the PrimExprs they are bound to, indicated by
    the corresponding iter_values in BlockRealize, and then convert the blocks into
    opaque ones by removing all the iter_values in BlockRealize and iter_vars in Block.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.ConvertBlocksToOpaque()  # type: ignore


def LiftThreadBinding():
    """Lift the same thread bindings to their LCA loops.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LiftThreadBinding()  # type: ignore


def CompactBufferAllocation(is_strict: bool = True):
    """Compact the buffer access region by removing the buffer regions
    that are not accessed, i.e. narrowing the buffer shape and adjust
    the access region if necessary.

    Parameters
    ----------
    is_strict : bool
        Ensure the compacted shape to be always smaller than the original shape.
        Otherwise it allows to grow the shape to match actual accessed buffer regions.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.CompactBufferAllocation(is_strict)  # type: ignore


def LowerMatchBuffer():
    """Remove match buffers inside the block. Also, it will validate the binding.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerMatchBuffer()  # type: ignore


def LowerOpaqueBlock():
    """Remove the block to ensure that the TIR can not be scheduled again.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerOpaqueBlock()  # type: ignore


def TransformMmaBufferLayout():
    """Transform mma buffer layout

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.TransformMmaBufferLayout()  # type: ignore


def InjectPermutedLayout():
    """Inject permuted layout in mma

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectPermutedLayout()  # type: ignore


def UnifyThreadBinding():
    """Unify all the thread bindings for "blockIdx.x/y/z",
    "threadIdx.x/y/z", and "vthread.x/y/z".

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.UnifyThreadBinding()  # type: ignore


def InjectSoftwarePipeline():
    """Transform annotated loops into pipelined one that parallelize producers and consumers

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectSoftwarePipeline()  # type: ignore


def LowerAutoCopy():
    """Automatically do memory optimizations for auto copy blocks

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerAutoCopy()  # type: ignore


def ManifestSharedMemoryLocalStage():
    """Add the explicit local stage for the shared memory access on GPU.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.ManifestSharedMemoryLocalStage()  # type: ignore


def AnnotateIrregularLoop():
    """Annotate irregular loop mark.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.AnnotateIrregularLoop()  # type: ignore


@_ffi.register_object("s_tir.transform.LoopPartitionConfig")
class LoopPartitionConfig(_ir.Attrs):
    """Config for loop partition pass"""


def LoopPartition():
    """Partition loops in the stmt.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LoopPartition()  # type: ignore


def InjectVirtualThread():
    """Inject virtual thread loops.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectVirtualThread()  # type: ignore


@_ffi.register_object("s_tir.transform.InjectDoubleBufferConfig")
class InjectDoubleBufferConfig(_ir.Attrs):
    """Config for inject double buffer pass"""


def InjectDoubleBuffer():
    """Inject double buffer statements.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectDoubleBuffer()  # type: ignore


def HoistIfThenElse(variant=None):
    """Hoist loop-invariant IfThenElse nodes to outside the eligible loops.

    Parameters
    ----------
    variant : Optional[String]
        The variant of the pass.
        variant can have any one of following values ["basic", None(Default)].

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    if variant == "basic":
        return _ffi_api.HoistIfThenElseBasic()  # type: ignore
    elif variant is None:
        return _ffi_api.HoistIfThenElse()  # type: ignore
    else:
        raise ValueError("wrong variant of HoistIfThenElse, " + variant)


def HoistExpression():
    """Hoist loop-invariant expressions to outside the eligible loops.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.HoistExpression()  # type: ignore


def RenormalizeSplitPattern():
    """Renormalize the split pattern from floordiv(floormod()) to floormod(floordiv())

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.RenormalizeSplitPattern()  # type: ignore


def RewriteUnsafeSelect():
    """Detect and rewrite unsafe select that contains memory access.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.RewriteUnsafeSelect()  # type: ignore


def InstrumentBoundCheckers():
    """Instruments bound checkers.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InstrumentBoundCheckers()  # type: ignore


def InjectPTXLDG32(enable_inject_ptx_intrin=True):
    """Inject ptx.ldg.32 intrinsics.

    Parameters
    ----------
    enable_inject_ptx_intrin : bool
        If True, inject ptx.ldg.32 intrinsics.
    """
    return _ffi_api.InjectPTXLDG32(enable_inject_ptx_intrin)  # type: ignore


def InstrumentProfileIntrinsics():
    """Insert intrinsic calls to instrument function and loop level profiling.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InstrumentProfileIntrinsics()  # type: ignore


def VerifyVTCMLimit(default_target=None):
    """Verify if the size of the allocated vtcm memory satisfies the limit.

    The limit is determined from the "vtcm-capacity" attribute of the target.

    Parameters
    ----------
    default_target : Optional[tvm.target.Target]
        The default target to use if a PrimFunc does not have a target attribute.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.VerifyVTCMLimit(default_target)  # type: ignore


def LowerVtcmAlloc():
    """Lower vtcm allocation.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerVtcmAlloc()  # type: ignore


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
    return _ffi_api.ThreadSync(storage_scope)  # type: ignore


def InferFragment():
    """Infer the TensorCore fragment information using tensor intrinsics.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InferFragment()  # type: ignore


def LowerThreadAllreduce():
    """Lower cross thread allreduce.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerThreadAllreduce()  # type: ignore


def LowerAsyncDMA():
    """Lower async DMA to DMA.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerAsyncDMA()  # type: ignore


def InjectPTXAsyncCopy():
    """Rewrite global to shared memory copy on CUDA with asynchronous copy.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectPTXAsyncCopy()  # type: ignore


def MergeSharedMemoryAllocations():
    """This pass merges multiple TIR-level shared memory allocations
    into one allocation.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MergeSharedMemoryAllocations()  # type: ignore


def DefaultGPUSchedule():
    """Set default thread bindings for GPU PrimFuncs.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.DefaultGPUSchedule()  # type: ignore


def RemoveWeightLayoutRewriteBlock(skip_tensor_rewrite=False):
    """Remove weight layout rewrite block before benchmarking during tuning stage.

    Parameters
    ----------
    skip_tensor_rewrite : bool
        If True, exact rewrite of Tensor, according to the given index map, will be skipped.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.RemoveWeightLayoutRewriteBlock(skip_tensor_rewrite)  # type: ignore


def RemoveStoreUndef():
    """Remove stores of undefined values from the Stmt.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.RemoveStoreUndef()  # type: ignore


def DecorateDeviceScope():
    """Decorate all the function's body as device function.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.DecorateDeviceScope()  # type: ignore


def UseAssumeToReduceBranches():
    """Eliminate layout specific pad branch by overcomputing values for padded region.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.UseAssumeToReduceBranches()  # type: ignore
