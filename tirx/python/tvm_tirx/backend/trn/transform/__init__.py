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
    "AnnotateKernelArguments": ".annotate_kernel_arguments",
    "TrnNaiveAllocator": ".naive_allocator",
    "TrnPrivateTensorAlloc": ".private_tensor_alloc",
}


def LowerTrainiumLayout():
    """Lower logical tensor accesses to Trainium physical shapes and indices.

    SBUF uses partition/free dimensions; PSUM adds a bank dimension. Global
    memory is flattened. Parameter ABI shapes stay logical. Run after tile
    dispatch.

    Examples
    --------
    .. code:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import tirx as T
        from tvm_tirx.backend.trn.axis import P, F
        from tvm_tirx.backend.trn.transform import LowerTrainiumLayout

        @T.ir_module
        class Before:
            @T.function
            def main():
                T.device_entry()
                A = T.alloc_tensor(ty_args=[T.Tensor(
                    (128, 2, 2), "float32", scope="trn.sbuf",
                    layout=T.TileLayout(T.S[(128, 4):(1 @ P, 1 @ F)]))])
                A[0, 1, 0] = T.float32(1)

        @T.ir_module
        class Expected:
            @T.function
            def main():
                T.device_entry()
                A = T.alloc_tensor(ty_args=[T.Tensor((128, 4), "float32", scope="trn.sbuf")])
                A[0, 2] = T.float32(1)

        after = LowerTrainiumLayout()(Before)
        tvm.ir.assert_structural_equal(after, Expected)
    """
    return get_global_func("tirx.backend.trn.transform.LowerTrainiumLayout")()


def LowerTIRx():
    """Run ``TileDispatch`` followed by ``LowerTrainiumLayout``.

    Run ``TrnPrivateTensorAlloc`` and ``TrnNaiveAllocator`` first when tiles
    need workspaces or allocation addresses. Tile dispatch requires a
    Trainium target on the function or in the current target scope.
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


__all__ = [
    "AnnotateKernelArguments",
    "LowerTIRx",
    "LowerTrainiumLayout",
    "TrnNaiveAllocator",
    "TrnPrivateTensorAlloc",
]
