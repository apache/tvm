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
"""Trainium-specific TIRX transformations."""
# pylint: disable=invalid-name

from tvm_ffi import get_global_func

import tvm_tirx
from tvm_tirx import tirx

_LAZY_TRANSFORMS = {
    "TrnNaiveAllocator": ".naive_allocator",
    "TrnPrivateTensorAlloc": ".private_tensor_alloc",
}


def LowerTrainiumLayout():
    """Lower logical tensor accesses to Trainium physical shapes and indices.

    SBUF layouts become partition/free dimensions; PSUM layouts additionally
    have a bank dimension. Ordinary global memory is flattened. Parameter
    declarations retain their logical ABI shape while physical views serve
    lowered accesses. Run after tile dispatch.

    Examples
    --------
    Apply to an IRModule ``mod``::

        from tvm_tirx.backend.trn.transform import LowerTrainiumLayout

        after = LowerTrainiumLayout()(mod)

    A local allocation and store change as follows (``T`` is
    ``tvm_tirx.script.tirx``; ``P`` and ``F`` are from
    ``tvm_tirx.backend.trn.axis``)::

        # Before
        A = T.alloc_tensor(ty_args=[T.Tensor(
            (128, 2, 2), "float32", scope="trn.sbuf",
            layout=T.TileLayout(T.S[(128, 4):(1 @ P, 1 @ F)]))])
        A[0, 1, 0] = T.float32(1)
        # After
        A = T.alloc_tensor(ty_args=[T.Tensor((128, 4), "float32", scope="trn.sbuf")])
        A[0, 2] = T.float32(1)
    """
    return get_global_func("tirx.backend.trn.transform.LowerTrainiumLayout")()


def LowerTIRx():
    """Run tile dispatch followed by Trainium physical-layout lowering.

    This two-pass sequence does not allocate private workspaces or assign
    allocation addresses. When tile instructions need those preparations, run
    ``TrnPrivateTensorAlloc`` and ``TrnNaiveAllocator`` first, as in the full
    Trainium pipeline. Tile dispatch needs a Trainium target on the function
    or in the current target scope.

    Examples
    --------
    Apply to a prepared IRModule ``mod``::

        from tvm_tirx.backend.trn.transform import LowerTIRx
        from tvm_tirx.target import Target

        with Target("aws/trn1/trn1.2xlarge"):
            after = LowerTIRx()(mod)

    This scalar TVMScript example exercises the layout phase; there are no
    tile calls to dispatch. ``T`` is ``tvm_tirx.script.tirx``, and ``P``/``F``
    are from ``tvm_tirx.backend.trn.axis``::

        # Before
        A = T.alloc_tensor(ty_args=[T.Tensor(
            (128, 2, 2), "float32", scope="trn.sbuf",
            layout=T.TileLayout(T.S[(128, 4):(1 @ P, 1 @ F)]))])
        A[0, 1, 0] = T.float32(1)
        # After
        A = T.alloc_tensor(ty_args=[T.Tensor((128, 4), "float32", scope="trn.sbuf")])
        A[0, 2] = T.float32(1)
    """
    return tvm_tirx.ir.transform.Sequential(
        [tirx.transform.TileDispatch(), LowerTrainiumLayout()],
        name="tirx.backend.trn.LowerTIRx",
    )


def __getattr__(name):
    target = _LAZY_TRANSFORMS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module  # pylint: disable=import-outside-toplevel

    return getattr(import_module(target, __name__), name)


__all__ = ["LowerTIRx", "LowerTrainiumLayout", "TrnNaiveAllocator", "TrnPrivateTensorAlloc"]
