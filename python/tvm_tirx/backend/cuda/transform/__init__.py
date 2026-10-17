# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""CUDA-specific TIRx transformations."""

# pylint: disable=invalid-name

from tvm_ffi import get_global_func


def LowerIket():
    """Lower frontend-only NVIDIA IKET annotations.

    Run after ``SplitHostDevice`` and before ``MakePackedAPI``. Strip
    annotations for regular compilation; emit metadata and NativeDump
    placeholders for IKET-enabled modules. The external ``run-iket`` process
    owns trace collection and postprocessing.

    Examples
    --------
    .. code:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import tirx as T
        from tvm_tirx.backend.cuda.transform import LowerIket

        @T.ir_module
        class Before:
            @T.function
            def main(A: T.Tensor((1,), "int32")):
                T.cuda.iket_mark("store")
                A[0] = 1

        @T.ir_module
        class Expected:
            @T.function
            def main(A: T.Tensor((1,), "int32")):
                A[0] = 1

        after = LowerIket()(Before)
        tvm.ir.assert_structural_equal(after, Expected)
    """
    return get_global_func("tirx.backend.cuda.transform.LowerIket")()


def BindBackendConfig():
    """Resolve CUDA function and device-entry compilation settings.

    Run after ``BindTarget`` and before tile dispatch. Function defaults and
    entry overrides become ``target`` and JSON ``backend_config`` attributes;
    non-CUDA functions are unchanged.

    Examples
    --------
    .. code:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import tirx as T
        from tvm_tirx.backend.config import backend_config_json
        from tvm_tirx.backend.cuda.backend_config import resolve_backend_config
        from tvm_tirx.backend.cuda.transform import BindBackendConfig

        target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})
        config = backend_config_json({"cuda": resolve_backend_config(target=target)})

        @T.ir_module
        class Before:
            @T.function
            def main(A: T.Tensor((1,), "int32")):
                T.func_attr({"target": target})
                with T.device_entry():
                    A[0] = 1

        @T.ir_module
        class Expected:
            @T.function
            def main(A: T.Tensor((1,), "int32")):
                T.func_attr({"target": target, "backend_config": config})
                with T.device_entry(attrs={"target": target, "backend_config": config}):
                    A[0] = 1

        after = BindBackendConfig()(Before)
        tvm.ir.assert_structural_equal(after, Expected)
    """
    return get_global_func("tirx.backend.cuda.transform.BindBackendConfig")("")


def SpecializeDeviceHelpers():
    """Specialize private device helpers for each CUDA compilation group.

    Run after ``SplitHostDevice``. Kernel target and backend configuration define
    each group. Called helpers receive those attributes and are cloned when
    their settings differ; helpers already in the same group can be reused.
    Unreferenced device-only helpers are omitted from the output module.
    """
    return get_global_func("tirx.backend.cuda.transform.SpecializeDeviceHelpers")()


def SpecializeEntryHelpers():
    """Specialize helpers for each CUDA device entry's resolved settings.

    Run after ``BindBackendConfig`` and before ``TileDispatch`` and
    ``SplitHostDevice``. Clone helpers when entry settings differ, propagating
    them through the call graph; remove originals that become unreachable.

    Examples
    --------
    .. code:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import tirx as T
        from tvm_tirx.backend.cuda.transform import SpecializeEntryHelpers

        def settings(arch):
            return {"target": tvm.target.Target({"kind": "cuda", "arch": arch}),
                    "backend_config": '{"cuda":{"arch":"' + arch + '"}}'}

        host_attrs, entry_attrs = settings("sm_100a"), settings("sm_90a")

        @T.ir_module
        class Before:
            @T.function(private=True)
            def helper(x: T.int32) -> T.int32:
                T.func_attr(host_attrs)
                return x + 1

            @T.function
            def main(A: T.Tensor((1,), "int32")):
                T.func_attr(host_attrs)
                with T.device_entry(attrs=entry_attrs):
                    A[0] = Before.helper(A[0])

        @T.ir_module
        class Expected:
            @T.function(private=True)
            def helper_cuda(x: T.int32) -> T.int32:
                T.func_attr(entry_attrs)
                return x + 1

            @T.function
            def main(A: T.Tensor((1,), "int32")):
                T.func_attr(host_attrs)
                with T.device_entry(attrs=entry_attrs):
                    A[0] = Expected.helper_cuda(A[0])

        after = SpecializeEntryHelpers()(Before)
        tvm.ir.assert_structural_equal(after, Expected)
    """
    return get_global_func("tirx.backend.cuda.transform.SpecializeEntryHelpers")()


__all__ = ["BindBackendConfig", "LowerIket", "SpecializeDeviceHelpers", "SpecializeEntryHelpers"]
