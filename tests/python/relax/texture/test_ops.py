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
import numpy as np
from tvm import relax
import tvm.testing
from tvm.relax.transform import ConvertLayout, Normalize
from tvm.script.parser import ir as I, relax as R, tir as T
from tvm.relax.transform.legalize_ops import adreno as legalize_adreno
from tvm.ir.module import IRModule
from tvm.relax.expr_functor import PyExprMutator, PyExprVisitor, mutator, visitor
from tvm import dlight as dl
from tvm.contrib import utils, ndk

import os
from tvm import rpc as _rpc


def get_rpc():
    rpc_target = os.getenv("RPC_TARGET", None)
    if rpc_target:
        connection_type = "tracker"
        host = os.getenv("TVM_TRACKER_HOST", "localhost")
        port = int(os.getenv("TVM_TRACKER_PORT", 9090))
        target = "opencl"
        target_host = "llvm -mtriple=aarch64-linux-gnu"
        device_key = os.getenv("RPC_DEVICE_KEY", "android")
        cross_compile = os.getenv("TVM_NDK_CC", "aarch64-linux-android-g++")
        tracker = _rpc.connect_tracker(host, port)
        return tracker.request(device_key, priority=1, session_timeout=1000)
    else:
        return None


def build_run(mod, inputs, is_adreno):
    tgt = tvm.target.Target("opencl --device=adreno", host="llvm -mtriple=aarch64-linux-gnu")
    skip_ops = [
        "relax.nn.conv2d",
        "relax.nn.max_pool2d",
        "relax.nn.adaptive_avg_pool2d",
        # "relax.nn.layer_norm",
    ]
    with tgt:
        mod = tvm.tir.transform.BindTarget(tvm.target.Target.current(allow_none=False))(mod)
        mod = tvm.relax.transform.DecomposeOpsForInference()(mod)
        mod = tvm.relax.transform.FoldConstant()(mod)
        desired_layouts = {"relax.nn.conv2d": ["NCHW4c", "OIHW4o", "NCHW4c"]}
        if is_adreno:
            mod = tvm.relax.transform.ConvertLayout(desired_layouts)(mod)
            mod = tvm.relax.transform.Normalize()(mod)
            mod = tvm.relax.transform.FoldConstant()(mod)
            mod = tvm.relax.transform.LegalizeOps(skip_ops=skip_ops)(mod)
            mod = tvm.relax.transform.AnnotateTIROpPattern()(mod)
            mod = tvm.relax.backend.adreno.transform.AnnotateCustomMemoryScope(tgt)(mod)
        mod = tvm.relax.transform.LegalizeOps()(mod)
        if is_adreno:
            mod = tvm.relax.transform.LegalizeOps(
                {"relax.nn.conv2d": legalize_adreno.conv2d_NCHWc_OIHWo},
            )(mod)
        mod = tvm.relax.transform.AnnotateTIROpPattern()(mod)
        mod = tvm.relax.transform.FoldConstant()(mod)
        mod = tvm.relax.transform.FuseOps()(mod)
        mod = tvm.relax.transform.FuseTIR()(mod)
        mod = tvm.relax.transform.DeadCodeElimination()(mod)
        if is_adreno:
            mod = tvm.relax.backend.adreno.transform.FoldVDeviceScopeChange()(mod)
            mod = tvm.relax.transform.DeadCodeElimination()(mod)
            mod = tvm.relax.transform.SpecializePrimFuncBasedOnCallSite()(mod)
        mod = tvm.relax.transform.Normalize()(mod)

        if is_adreno:
            mod = dl.ApplyDefaultSchedule(
                dl.adreno.Conv2d(),
                dl.adreno.LayoutTransform(),
                dl.adreno.Pool2D(),
            )(mod)

        mod = dl.ApplyDefaultSchedule(
            dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(),
            dl.gpu.Fallback(),
        )(mod)

        mod = tvm.relax.transform.ToNonDataflow()(mod)
        mod = tvm.relax.transform.RemovePurityChecking()(mod)
        # print(mod)
        mod = tvm.relax.transform.CallTIRRewrite()(mod)
        mod = tvm.relax.transform.Normalize()(mod)
        mod = tvm.relax.transform.StaticPlanBlockMemory()(mod)
        mod = tvm.relax.transform.LowerAllocTensor()(mod)
        mod = tvm.relax.transform.KillAfterLastUse()(mod)
        mod = tvm.relax.transform.VMBuiltinLower()(mod)
        mod = tvm.relax.transform.VMShapeLower()(mod)
        mod = tvm.relax.transform.AttachGlobalSymbol()(mod)

    # print("Mod relax.build:", mod)
    # exit(0)
    ex = relax.build(mod, tgt)
    # for smod in ex.mod.imported_modules:
    #    print("Mod:", smod.type_key)
    #    for cmod in smod.imported_modules:
    #       print(cmod.get_source())
    load_path = "vm_library.so"
    temp = utils.tempdir()
    path = temp.relpath(load_path)
    path = "./" + load_path
    ex.export_library(path, fcompile=ndk.create_shared, options=["-shared", "-fPIC", "-lm"])

    rpc = get_rpc()
    rpc.upload(path)
    rexec = rpc.load_module(load_path)
    dev = rpc.cl(0)

    if "vdevice" in mod.global_infos:
        device_arr = [dev for ii in range(len(mod.global_infos["vdevice"]))]
    else:
        device_arr = [dev]

    vm = relax.VirtualMachine(rexec, device_arr)
    inputs = [tvm.runtime.tensor(inp, dev) for inp in inputs]
    vm.set_input("main", *inputs)
    vm.invoke_stateful("main")
    tvm_output = vm.get_outputs("main")
    if isinstance(tvm_output, tuple):
        tvm_output = (out.numpy() for out in tvm_output)
    else:
        tvm_output = tvm_output.numpy()

    rpc.get_function("CloseRPCConnection")()
    return tvm_output


def verify(mod):
    inputs = []
    for arg in mod["main"].params:
        shape = tuple(shape_val.value for shape_val in arg.struct_info.shape.values)
        inputs.append(np.random.uniform(-1, 1, size=shape).astype(arg.struct_info.dtype))

    ret1 = build_run(mod, inputs, True)
    ret2 = build_run(mod, inputs, False)

    if isinstance(ret1, tuple):
        for val1, val2 in zip(ret1, ret2):
            tvm.testing.assert_allclose(val1, ret2, rtol=1e-3, atol=1e-3)
    else:
        tvm.testing.assert_allclose(ret1, ret2, rtol=1e-3, atol=1e-3)


@tvm.testing.requires_opencl
def test_conv2d():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 64, 56, 56), "float32"), w: R.Tensor((32, 64, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 32, 54, 54), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                R.output(gv)
            return gv

    verify(Input)


@tvm.testing.requires_opencl
def test_conv2d_relu():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                R.output(gv2)
            return gv2

    verify(Input)


@tvm.testing.requires_opencl
def test_relu_conv2d_relu():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                x0: R.Tensor((2, 16, 28, 28), "float32") = R.nn.relu(x)
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x0, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                R.output(gv2)
            return gv2

    verify(Input)


@tvm.testing.requires_opencl
def test_conv2d_relu_tanh():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                gv3: R.Tensor((2, 4, 26, 26), "float32") = R.tanh(gv2)
                R.output(gv3)
            return gv3

    verify(Input)


@tvm.testing.requires_opencl
def test_conv2d_add():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"),
            w: R.Tensor((4, 16, 3, 3), "float32"),
            bias: R.Tensor((2, 4, 26, 26), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.add(gv, bias)
                R.output(gv2)
            return gv2

    verify(Input)


@tvm.testing.requires_opencl
def test_conv2d_sum():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4), "float32") = R.sum(gv, axis=[2, 3])
                R.output(gv2)
            return gv2

    verify(Input)


@tvm.testing.requires_opencl
def test_conv2d_sum_keepdims():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 1, 1), "float32") = R.sum(gv, axis=[2, 3], keepdims=True)
                R.output(gv2)
            return gv2

    verify(Input)


@tvm.testing.requires_opencl
def test_conv2d_sum_reduce():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 26), "float32") = R.sum(gv, axis=[1, 2])
                R.output(gv2)
            return gv2

    verify(Input)


@tvm.testing.requires_opencl
def test_conv2d_transpose():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((26, 26, 4, 2), "float32") = R.permute_dims(gv, axes=[3, 2, 1, 0])
                R.output(gv2)
            return gv2

    verify(Input)


@tvm.testing.requires_opencl
def test_conv2d_expand_dims():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=6):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 1, 4, 1, 26, 26), "float32") = R.expand_dims(gv, axis=(-3, 1))
                R.output(gv2)
            return gv2

    verify(Input)


@tvm.testing.requires_opencl
def test_conv2d_squeeze():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((1, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=3):
            with R.dataflow():
                gv: R.Tensor((1, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((4, 26, 26), "float32") = R.squeeze(gv, axis=[0])
                R.output(gv2)
            return gv2

    verify(Input)


@tvm.testing.requires_opencl
def test_conv2d_strided_slice():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 2, 9, 7), dtype="float32") = R.strided_slice(
                    gv, begin=[0, 0, 0], end=[4, 26, 26], strides=[2, 3, 4], axes=[1, 2, 3]
                )
                R.output(gv2)
            return gv2

    verify(Input)


@tvm.testing.requires_opencl
def test_conv2d_relu_concat():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                gv3: R.Tensor((2, 8, 26, 26), "float32") = R.concat((gv, gv2), axis=1)
                R.output(gv3)
            return gv3

    verify(Input)


@tvm.testing.requires_opencl
def test_conv2d_relu_concat_split():
    @I.ir_module
    class Input:
        @R.function
        def main(x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                gv3: R.Tensor((2, 8, 26, 26), "float32") = R.concat((gv, gv2), axis=1)
                gv4 = R.split(gv3, indices_or_sections=2, axis=1)
                # TODO @Siva: Multi value return have an issue at runtime.
                gv5 = gv4[0]
                R.output(gv5)
            return gv5

    verify(Input)


@tvm.testing.requires_opencl
def test_conv2d_relu_concat_split_transpose_concat():
    @I.ir_module
    class Input:
        @R.function
        def main(x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                gv3: R.Tensor((2, 8, 26, 26), "float32") = R.concat((gv, gv2), axis=1)
                gv4 = R.split(gv3, indices_or_sections=2, axis=1)
                gv5: R.Tensor((26, 26, 4, 2), "float32") = R.permute_dims(gv4[0], axes=[3, 2, 1, 0])
                gv6: R.Tensor((26, 26, 4, 2), "float32") = R.permute_dims(gv4[1], axes=[3, 2, 1, 0])
                gv7: R.Tensor((26, 26, 8, 2), "float32") = R.concat((gv5, gv6), axis=2)
                R.output(gv7)
            return gv7

    verify(Input)


@tvm.testing.requires_opencl
def test_conv2d_maxpool2d():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2 = R.nn.max_pool2d(
                    gv,
                    pool_size=[2, 2],
                    strides=[2, 2],
                    padding=[0, 0],
                    layout="NCHW",
                    out_layout="NCHW",
                )
                R.output(gv2)
            return gv2

    verify(Input)


@tvm.testing.requires_opencl
def test_conv2d_avgpool2d():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2 = R.nn.adaptive_avg_pool2d(gv, output_size=[13, 13], layout="NCHW")
                R.output(gv2)
            return gv2

    verify(Input)


@tvm.testing.requires_opencl
def test_conv2d_softmax():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2 = R.nn.softmax(gv, axis=1)
                R.output(gv2)
            return gv2

    verify(Input)


@tvm.testing.requires_opencl
def test_conv2d_layernorm():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"),
            w: R.Tensor((4, 16, 3, 3), "float32"),
            gamma: R.Tensor((26, 26), dtype="float32"),
            beta: R.Tensor((26, 26), dtype="float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.layer_norm(
                    gv, gamma, beta, axes=[-2, -1]
                )
                R.output(gv2)
            return gv2

    verify(Input)


@tvm.testing.requires_opencl
def test_binary_broadcast():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"),
            w: R.Tensor((4, 16, 3, 3), "float32"),
            bias: R.Tensor((26, 26), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.add(gv, bias)
                R.output(gv2)
            return gv2

    verify(Input)


@tvm.testing.requires_opencl
def test_binary_ewise_scalar():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.add(gv, R.const(1, "float32"))
                R.output(gv2)
            return gv2

    verify(Input)


@tvm.testing.requires_opencl
def test_residual_block():
    """
    - some kind of residual block followed by convolution to have texture after residual block
    - scalar data type verification which should be mapped to global memory scope
        layout_transform (NCHW->NCHW4c)
                  |                      <- buffer
                conv2d (1)                  <- to get textures as output
               /         \
            conv2d (2)    |
                 \       /
                    add                     <- add should be fused into conv2d (2)
                multiply to scalar          <- buffer to the input of multiply scalar value
                    relu
                     |                      <- texture in intermediate tensor
                  conv2d (3)
                   relu
                     |                      <- buffer
               layout_transform (NCHW4c->NCHW)
    """

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 32, 40, 40), "float32"),
            w1: R.Tensor((32, 32, 2, 2), "float32"),
            w2: R.Tensor((32, 32, 1, 1), "float32"),
            w3: R.Tensor((32, 32, 2, 2), "float32"),
            bias: R.Tensor((1, 32, 1, 1), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv = R.nn.conv2d(x, w1, strides=[2, 2], out_dtype="float32")
                gv1 = R.add(gv, bias)
                gv2 = R.nn.relu(gv1)
                gv3 = R.nn.conv2d(gv2, w2, strides=[1, 1], out_dtype="float32")
                bias_1 = R.multiply(bias, R.const(0.15, "float32"))
                gv4 = R.add(gv3, bias_1)
                gv5 = R.nn.relu(gv4)
                gv6 = R.nn.conv2d(gv5, w3, strides=[2, 2], out_dtype="float32")
                gv7 = R.nn.relu(gv6)
                R.output(gv7)
            return gv7

    verify(Input)


@tvm.testing.requires_opencl
def test_conv2d_conv2d_fallback_to_buffer_conv2d():
    """
        layout_transform (NCHW->NCHW4c)
                  |                      <- texture
                conv2d (1)               <- textures as output
               /         \
            conv2d (2)    conv2d (3)     <- conv2d (2) emits texture, conv2d (3) emits buffer
                 \       /               <- concat shouldn't support textures here
                concatenation
                     |                   <- buffer
               layout_transform (NCHW4c->NCHW)
    """

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 32, 40, 40), "float32"),
            w1: R.Tensor((96, 32, 2, 2), "float32"),
            w2: R.Tensor((32, 96, 2, 2), "float32"),
            w3: R.Tensor((5, 96, 2, 2), "float32"),
            bias1: R.Tensor((1, 96, 1, 1), "float32"),
            bias2: R.Tensor((1, 32, 1, 1), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv = R.nn.conv2d(x, w1, strides=[2, 2], out_dtype="float32")
                gv1 = R.add(gv, bias1)
                gv2 = R.nn.relu(gv1)
                gv3 = R.nn.conv2d(gv2, w2, strides=[2, 2], out_dtype="float32")
                gv4 = R.add(gv3, bias2)
                gv5 = R.nn.relu(gv4)
                gv6 = R.nn.conv2d(gv2, w3, strides=[2, 2], out_dtype="float32")
                gv7 = R.concat((gv3, gv6), axis=1)
                R.output(gv7)
            return gv7

    verify(Input)


@tvm.testing.requires_opencl
def test_conv2d_conv2d_conv2d_concat():
    """
        layout_transform (NCHW->NCHW4c)
                  |                      <- texture
                conv2d (1)               <- textures as output
               /         \
            conv2d (2)    conv2d (3)
                 \       /               <- concat does support textures here
                concatenation
                     |                   <- buffer
               layout_transform (NCHW4c->NCHW)
    """

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 32, 40, 40), "float32"),
            w1: R.Tensor((96, 32, 2, 2), "float32"),
            w2: R.Tensor((32, 96, 2, 2), "float32"),
            w3: R.Tensor((8, 96, 2, 2), "float32"),
            bias1: R.Tensor((1, 96, 1, 1), "float32"),
            bias2: R.Tensor((1, 32, 1, 1), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv = R.nn.conv2d(x, w1, strides=[2, 2], out_dtype="float32")
                gv1 = R.add(gv, bias1)
                gv2 = R.nn.relu(gv1)
                gv3 = R.nn.conv2d(gv2, w2, strides=[2, 2], out_dtype="float32")
                gv4 = R.add(gv3, bias2)
                gv5 = R.nn.relu(gv4)
                gv6 = R.nn.conv2d(gv2, w3, strides=[2, 2], out_dtype="float32")
                gv7 = R.concat((gv3, gv6), axis=1)
                R.output(gv7)
            return gv7

    verify(Input)


@tvm.testing.requires_opencl
def _test_pooling_branching_texture_params():
    """
    Verification of the pooling and many branches having textures
                layout_transform (NCHW->NCHW4c)
                         |                        <- texture
                      conv2d (0)                  <- to get textures
                         |                        <- textures
                     pooling
               /           \           \          <- textures
            conv2d (1)    conv2d (2)    conv2d (3)
                \             /           |
                     add                  |       <- to have  the only one output, will be fused
                      \                  /
                            add                  <- to have  the only one output, will be fused
                             |                   <- buffer
                    layout_transform (NCHW4c->NCHW)
    """

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 32, 40, 40), "float32"),
            w1: R.Tensor((32, 32, 1, 1), "float32"),
            w2: R.Tensor((32, 32, 2, 2), "float32"),
            w3: R.Tensor((32, 32, 1, 1), "float32"),
            w4: R.Tensor((32, 32, 2, 2), "float32"),
            bias1: R.Tensor((1, 32, 1, 1), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv = R.nn.conv2d(x, w1, strides=[1, 1], out_dtype="float32")
                gv1 = R.nn.max_pool2d(gv, pool_size=[2, 2], strides=[2, 2])
                gv2 = R.nn.conv2d(
                    gv1, w2, padding=[0, 0, 1, 1], strides=[1, 1], out_dtype="float32"
                )
                gv3 = R.add(gv2, bias1)
                gv4 = R.nn.relu(gv3)
                gv5 = R.nn.conv2d(
                    gv1, w3, padding=[0, 0, 0, 0], strides=[1, 1], out_dtype="float32"
                )
                gv6 = R.nn.conv2d(
                    gv1, w4, padding=[0, 1, 1, 0], strides=[1, 1], out_dtype="float32"
                )
                gv7 = R.nn.relu(gv6)
                gv8 = R.add(gv2, gv5)
                gv9 = R.add(gv8, gv6)
                R.output(gv9)
            return gv9

    verify(Input)


@tvm.testing.requires_opencl
def _test_injective_inputs1():
    """
                                     Input
                               /                   \
                            /                      |
                         |                        /
                      conv2d (1)                 /
                         |                      /
                      conv2d (2)              mean
                  /         \                 /
                 |           |      \        /
                 |           |       (3) add
                 |           |         |
                 |             \    /
                 \                mul
                  \            /
                        add

    """

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((1, 4, 40, 40), "float32"),
            w1: R.Tensor((4, 4, 3, 3), "float32"),
            w2: R.Tensor((4, 4, 3, 3), "float32"),
            w3: R.Tensor((4, 4, 3, 3), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                mean = R.mean(x, axis=1, keepdims=True)
                conv1 = R.nn.conv2d(
                    x, w1, padding=[1, 1, 1, 1], strides=[1, 1], out_dtype="float32"
                )
                conv2 = R.nn.conv2d(
                    conv1, w2, padding=[1, 1, 1, 1], strides=[1, 1], out_dtype="float32"
                )
                ad3 = R.add(conv1, conv2)
                ad1 = R.add(mean, conv1)
                ad2 = R.multiply(ad1, conv2)
                gv = R.add(ad3, ad2)
                R.output(gv)
            return gv

    verify(Input)


@tvm.testing.requires_opencl
def _test_injective_nwo_inputs2():
    """
                                     Input
                               /             \
                         |                    \
                      conv2d                   \
                         |                     /
                      conv2d           mean    /
                  /         \                 /
                add         |   \             |
                 |           |    \           |
                 |           |      \        /
                 |           |       (3) add
                 |           |         |
                 |            \       /
                 |             \    /
                 \                mul
                  \            /
                        add

    """

    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((1, 4, 40, 40), "float32"),
            w1: R.Tensor((4, 4, 3, 3), "float32"),
            w2: R.Tensor((4, 4, 3, 3), "float32"),
            w3: R.Tensor((4, 4, 3, 3), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                mean = R.mean(x, axis=1, keepdims=True)
                conv1 = R.nn.conv2d(
                    x, w1, padding=[1, 1, 1, 1], strides=[1, 1], out_dtype="float32"
                )
                conv2 = R.nn.conv2d(
                    conv1, w2, padding=[1, 1, 1, 1], strides=[1, 1], out_dtype="float32"
                )
                ad3 = R.add(conv1, conv2)
                ad1 = R.add(mean, conv1)
                ad2 = R.multiply(ad1, conv2)
                gv = R.add(ad2, ad3)
                R.output(gv)
            return gv

    verify(Input)


if __name__ == "__main__":
    tvm.testing.main()
