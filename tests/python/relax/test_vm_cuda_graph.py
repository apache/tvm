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

import tvm
import tvm.testing

from tvm import relax
from tvm.script import tir as T, relax as R, ir as I

import numpy as np
import pytest


# fmt: off


@I.ir_module
class Module:
    @R.function(pure=False)
    def main(x: R.Tensor((16, 16), dtype="float32")) -> R.Tensor((16, 16), dtype="float32"):
        cls = Module
        R.func_attr({"global_symbol": "main"})
        gv: R.Tuple(R.Object, R.Object) = R.call_builtin_with_ctx("vm.builtin.cuda_graph.get_cached_alloc", (cls.cuda_graph_alloc, R.prim_value(0)), sinfo_args=(R.Tuple(R.Object, R.Object),))
        storage: R.Object = gv[0]
        alloc = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape((16, 16)), R.dtype("float32"))
        _: R.Tuple = cls.add(x, alloc)
        storage1: R.Object = gv[1]
        gv1: R.Tuple(R.Tensor(dtype="float32"), R.Object, R.Object) = (alloc, storage1, storage)
        gv2: R.Tuple(R.Tensor((16, 16), dtype="float32")) = R.call_builtin_with_ctx("vm.builtin.cuda_graph.run_or_capture", (cls.cuda_graph_capture, gv1, R.prim_value(0)), sinfo_args=(R.Tuple(R.Tensor((16, 16), dtype="float32")),))
        storage2: R.Object = R.vm.alloc_storage(R.shape((1024,)), R.prim_value(0), R.dtype("uint8"))
        alloc3 = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape((16, 16)), R.dtype("float32"))
        lv4: R.Tensor((16, 16), dtype="float32") = gv2[0]
        _3: R.Tuple = cls.add(lv4, alloc3)
        lv5: R.Tensor(dtype="float32") = alloc3
        return lv5

    @T.prim_func
    def add(A: T.Buffer((16, 16), "float32"), B: T.Buffer((16, 16), "float32")):
        T.func_attr({"global_symbol": "add"})
        with T.block("root"):
            for i in T.thread_binding(16, thread="threadIdx.x"):
                for j in range(16):
                    with T.block("update"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] + T.float32(1)

    @R.function
    def cuda_graph_alloc() -> R.Tuple(R.Object, R.Object):
        R.func_attr({"global_symbol": "cuda_graph_alloc"})
        storage: R.Object = R.vm.alloc_storage(R.shape((1024,)), R.prim_value(0), R.dtype("uint8"))
        storage1: R.Object = R.vm.alloc_storage(R.shape((1024,)), R.prim_value(0), R.dtype("uint8"))
        gv: R.Tuple(R.Object, R.Object) = (storage, storage1)
        return gv

    @R.function(pure=False)
    def cuda_graph_capture(alloc: R.Tensor((16, 16), dtype="float32"), storage1: R.Object, storage: R.Object) -> R.Tuple(R.Tensor((16, 16), dtype="float32")):
        cls = Module
        R.func_attr({"global_symbol": "cuda_graph_capture"})
        lv0: R.Tensor((16, 16), dtype="float32") = alloc
        alloc1 = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape((16, 16)), R.dtype("float32"))
        _1: R.Tuple = cls.add(lv0, alloc1)
        lv1: R.Tensor(dtype="float32") = alloc1
        lv2: R.Tuple(R.Tensor(dtype="float32")) = (lv1,)
        lv3: R.Tensor(dtype="float32") = lv2[0]
        alloc2 = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape((16, 16)), R.dtype("float32"))
        _2: R.Tuple = cls.add(lv3, alloc2)
        lv4: R.Tensor(dtype="float32") = alloc2
        gv: R.Tuple(R.Tensor(dtype="float32")) = (lv4,)
        return gv


# fmt: on


def codegen(mod, target, exec_mode="bytecode"):
    builder = relax.ExecBuilder()
    leftover_mod = relax.vm_build._vmcodegen(builder, mod, exec_mode=exec_mode)
    tir_mod = relax.vm_build._filter_tir(leftover_mod)
    return relax.vm_build._vmlink(builder, target, tir_mod)


@tvm.testing.requires_cuda
def test_vm_run():
    mod = Module
    target = tvm.target.Target("cuda", host="llvm")
    ex = codegen(mod, target)
    dev = tvm.cuda(0)
    vm = relax.VirtualMachine(ex, dev)
    x_np = np.random.uniform(size=(16, 16)).astype("float32")
    x = tvm.nd.array(x_np, dev)
    y = vm["main"](x)
    y_np = x_np + 1.0 + 1.0 + 1.0 + 1.0
    tvm.testing.assert_allclose(y.asnumpy(), y_np, rtol=1e-5, atol=1e-5)


@tvm.testing.requires_cudagraph
def test_capture_error_is_recoverable():
    """Function calls while capturing cudagraph may throw exceptions

    Calls to PackedFuncs may occur within a captured cudaGraph.  If a
    call to that PackedFunc raises an exception while capturing the
    cudaGraph, throwing exception should cleanly unwind the stack, and
    the exception may be caught in the calling scope.

    This is a regression test.  In previous implementations, an
    exception thrown while capturing a cudaGraph would skip the call
    to `cudaStreamEndCapture`, causing additional exceptions to be
    thrown while freeing memory in TVM destructors.  Since C++ does
    not support stack unwinding from multiple simultaneous exceptions,
    this would result in immediate `std::terminate`, making it
    difficult to debug the original error.

    """

    target = tvm.target.Target("cuda")
    dev = tvm.cuda()

    @tvm.register_func("test_vm_cuda_graph.invalid_impl_for_cudagraph", override=True)
    def invalid_impl_for_cudagraph(arg_tensor):
        # Memory allocation/deallocation may not be performed while
        # capturing a cudaGraph.  This passes the warm-up run
        # performed by "vm.builtin.cuda_graph.run_or_capture", but
        # throws an exception when the cudaGraph is being captured.
        _dummy_workspace = tvm.nd.empty([16], "float16", dev)
        return arg_tensor

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16], "float16")):
            B = R.add(A, A)
            C = R.call_pure_packed(
                "test_vm_cuda_graph.invalid_impl_for_cudagraph",
                B,
                sinfo_args=R.Tensor([16], "float16"),
            )
            D = R.add(C, C)
            return D

    with target, tvm.ir.transform.PassContext(config={"relax.backend.use_cuda_graph": True}):
        Module = tvm.ir.transform.Sequential(
            [
                tvm.relax.transform.LegalizeOps(),
                tvm.tir.transform.DefaultGPUSchedule(),
                tvm.relax.transform.RemovePurityChecking(),
                tvm.relax.transform.CallTIRRewrite(),
                tvm.relax.transform.StaticPlanBlockMemory(),
                tvm.relax.transform.RewriteCUDAGraph(),
            ]
        )(Module)

    assert "cuda_graph_alloc" in Module, (
        "Validity of unit test requires the call to `invalid_impl_for_cudagraph` "
        "to have been captured by RewriteCUDAGraph."
    )

    built = tvm.relax.build(Module, target=target)
    vm = tvm.relax.VirtualMachine(built, dev)

    arg = tvm.nd.array(np.arange(16).astype("float16"), dev)

    with pytest.raises(tvm.TVMError):
        vm["main"](arg)


if __name__ == "__main__":
    tvm.testing.main()
