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
""" Support level2 operator test cases.
"""
import sys

import numpy as np
import pytest
import tvm
import tvm.testing
import tvm.topi.testing
from tvm import autotvm, relay, te
from tvm.contrib import utils, cudnn
from tvm.ir.module import IRModule
from tvm.relay import transform
from tvm.relay.testing import run_infer_type
from tvm.topi.cuda.conv3d_winograd import _infer_tile_size

executor_kind = tvm.testing.parameter("graph", "vm")


@tvm.testing.uses_gpu
def test_conv1d_infer_type():
    # symbolic in batch dimension
    n, c, w = te.var("n"), 10, 224
    x = relay.var("x", relay.ty.TensorType((n, c, w), "float32"))
    w = relay.var("w")
    y = relay.nn.conv1d(x, w, kernel_size=3, padding=(1, 1), channels=2)
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, 2, 224), "float32")
    assert yy.args[1].checked_type == relay.TensorType((2, 10, 3), "float32")

    # infer by shape of w, mixed precision
    n, c, w = te.var("n"), 10, 224
    x = relay.var("x", relay.TensorType((n, c, w), "int8"))
    w = relay.var("w", relay.TensorType((2, 10, 3), "int8"))
    y = relay.nn.conv1d(x, w, out_dtype="int32")
    assert 'out_dtype="int32"' in y.astext()
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, 2, 222), "int32")

    # infer shape in case of different dtypes for input and weight.
    n, c, w = te.var("n"), 10, 224
    x = relay.var("x", relay.TensorType((n, c, w), "uint8"))
    w = relay.var("w", relay.TensorType((2, 10, 3), "int8"))
    y = relay.nn.conv1d(x, w, out_dtype="int32")
    assert 'out_dtype="int32"' in y.astext()
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, 2, 222), "int32")

    # Infer with NWC
    n, c, w = 4, 32, 224
    x = relay.var("x", relay.TensorType((n, w, c), "int8"))
    wt = relay.var("w")
    y = relay.nn.conv1d(
        x, wt, kernel_size=3, padding=(1, 1), channels=16, data_layout="NWC", out_dtype="int32"
    )
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, w, 16), "int32")


@tvm.testing.uses_gpu
def test_conv1d_run():
    def run_test_conv1d(
        dtype,
        out_dtype,
        scale,
        dshape,
        kshape,
        padding=(1, 1),
        fref=None,
        dilation=1,
        except_targets=None,
        **attrs,
    ):
        if except_targets is None:
            except_targets = []

        x = relay.var("x", shape=dshape, dtype=dtype)
        w = relay.var("w", dtype=dtype)
        y = relay.nn.conv1d(x, w, padding=padding, dilation=dilation, **attrs)
        func = relay.Function([x, w], y)
        data = np.random.uniform(-scale, scale, size=dshape).astype(dtype)
        kernel = np.random.uniform(-scale, scale, size=kshape).astype(dtype)
        ref_res = tvm.topi.testing.conv1d_ncw_python(
            data.astype(out_dtype), kernel.astype(out_dtype), 1, padding, dilation
        )

        for target, dev in tvm.testing.enabled_targets():
            if target in except_targets:
                continue
            dev = tvm.device(target, 0)
            op_res1 = relay.create_executor("graph", device=dev, target=target).evaluate(func)(
                data, kernel
            )
            tvm.testing.assert_allclose(op_res1.numpy(), ref_res, rtol=1e-5, atol=1e-5)

    # normal conv1d
    dshape = (1, 3, 224)
    kshape = (10, 3, 3)
    run_test_conv1d(
        "float32", "float32", 1, dshape, kshape, padding=(1, 1), channels=10, kernel_size=3
    )
    # mixed precision
    run_test_conv1d("int8", "int32", 1, dshape, kshape, padding=(1, 1), channels=10, kernel_size=3)
    # dilated conv2d
    dshape = (1, 3, 18)
    kshape = (10, 3, 3)
    run_test_conv1d(
        "float32",
        "float32",
        1,
        dshape,
        kshape,
        padding=(1, 1),
        channels=10,
        kernel_size=3,
        dilation=3,
    )


@tvm.testing.uses_gpu
def test_conv2d_infer_type():
    # symbolic in batch dimension
    n, c, h, w = te.size_var("n"), 10, 224, 224
    x = relay.var("x", relay.ty.TensorType((n, c, h, w), "float32"))
    w = relay.var("w")
    y = relay.nn.conv2d(x, w, kernel_size=(3, 3), padding=(1, 1), channels=2)
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, 2, 224, 224), "float32")
    assert yy.args[1].checked_type == relay.TensorType((2, 10, 3, 3), "float32")

    # infer by shape of w, mixed precision
    n, c, h, w = te.size_var("n"), 10, 224, 224
    x = relay.var("x", relay.TensorType((n, c, h, w), "int8"))
    w = relay.var("w", relay.TensorType((2, 10, 3, 3), "int8"))
    y = relay.nn.conv2d(x, w, out_dtype="int32")
    assert 'out_dtype="int32"' in y.astext()
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, 2, 222, 222), "int32")

    # infer shape in case of different dtypes for input and weight.
    n, c, h, w = te.size_var("n"), 10, 224, 224
    x = relay.var("x", relay.TensorType((n, c, h, w), "uint8"))
    w = relay.var("w", relay.TensorType((2, 10, 3, 3), "int8"))
    y = relay.nn.conv2d(x, w, out_dtype="int32")
    assert 'out_dtype="int32"' in y.astext()
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, 2, 222, 222), "int32")

    # Infer with a different layout
    n, c, h, w = 4, 32, 224, 224
    x = relay.var("x", relay.TensorType((n // 4, c // 4, h, w, 4, 4), "int8"))
    wt = relay.var("w")
    y = relay.nn.conv2d(
        x,
        wt,
        kernel_size=(3, 3),
        padding=(1, 1),
        channels=16,
        data_layout="NCHW4n4c",
        kernel_layout="OIHW4o4i",
        out_dtype="int32",
    )
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((1, 4, 224, 224, 4, 4), "int32")
    assert yy.args[1].checked_type == relay.TensorType((4, 8, 3, 3, 4, 4), "int8")

    # Infer with NHWC
    n, c, h, w = 4, 32, 224, 224
    x = relay.var("x", relay.TensorType((n, h, w, c), "int8"))
    wt = relay.var("w")
    y = relay.nn.conv2d(
        x,
        wt,
        kernel_size=(3, 3),
        padding=(1, 1),
        channels=16,
        data_layout="NHWC",
        out_dtype="int32",
    )
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, h, w, 16), "int32")


class TestConv2D:
    config = {
        "group1": dict(
            dtype="float32",
            out_dtype="float32",
            scale=1,
            dshape=(1, 32, 18, 18),
            kshape=(32, 4, 3, 3),
            padding=(1, 1),
            channels=32,
            groups=8,
            kernel_size=(3, 3),
            dilation=(1, 1),
        ),
        "group2": dict(
            dtype="float32",
            out_dtype="float32",
            scale=1,
            dshape=(1, 32, 18, 18),
            kshape=(64, 1, 3, 3),
            padding=(1, 1),
            channels=64,
            groups=32,
            kernel_size=(3, 3),
            dilation=(1, 1),
        ),
        "normal": dict(
            dtype="float32",
            out_dtype="float32",
            scale=1,
            dshape=(1, 3, 224, 224),
            kshape=(10, 3, 3, 3),
            padding=(1, 1),
            channels=10,
            groups=1,
            kernel_size=(3, 3),
            dilation=(1, 1),
        ),
        "mixed_precision_int8_int32_case1": dict(
            dtype="int8",
            out_dtype="int32",
            scale=1,
            dshape=(1, 3, 224, 224),
            kshape=(10, 3, 3, 3),
            padding=(1, 1),
            channels=10,
            groups=1,
            kernel_size=(3, 3),
            dilation=(1, 1),
        ),
        "mixed_precision_int8_int32_case2": dict(
            dtype="int8",
            out_dtype="int32",
            scale=1,
            dshape=(1, 3, 224, 224),
            kshape=(10, 3, 1, 3),
            padding=(0, 1),
            channels=10,
            groups=1,
            kernel_size=(1, 3),
            dilation=(1, 1),
        ),
        "dilated": dict(
            dtype="float32",
            out_dtype="float32",
            scale=1,
            dshape=(1, 3, 18, 18),
            kshape=(10, 3, 3, 3),
            padding=(1, 1),
            channels=10,
            groups=1,
            kernel_size=(3, 3),
            dilation=(3, 3),
        ),
    }

    # TODO(Lunderberg): Make a cleaner utility for this type of
    # parametrization.  It would be much nicer to have the fixture
    # name come from the dictionaries themselves, rather than needing
    # to be re-packed into tuples.
    (
        dtype,
        out_dtype,
        scale,
        dshape,
        kshape,
        padding,
        channels,
        groups,
        kernel_size,
        dilation,
    ) = tvm.testing.parameters(
        *[
            [
                d[p]
                for p in [
                    "dtype",
                    "out_dtype",
                    "scale",
                    "dshape",
                    "kshape",
                    "padding",
                    "channels",
                    "groups",
                    "kernel_size",
                    "dilation",
                ]
            ]
            for d in config.values()
        ],
        ids=config.keys(),
    )

    def test_run(
        self,
        target,
        dev,
        dtype,
        out_dtype,
        scale,
        dshape,
        kshape,
        padding,
        groups,
        dilation,
        channels,
        kernel_size,
    ):
        target = tvm.target.Target(target)

        x = relay.var("x", shape=dshape, dtype=dtype)
        w = relay.var("w", shape=kshape, dtype=dtype)
        y = relay.nn.conv2d(
            x,
            w,
            padding=padding,
            dilation=dilation,
            groups=groups,
            channels=channels,
            kernel_size=kernel_size,
        )
        func = relay.Function([x, w], y)

        kernel = np.random.uniform(-scale, scale, size=kshape).astype(dtype)
        dkernel = tvm.topi.testing.dilate_python(kernel, (1, 1) + dilation)

        data = np.random.uniform(-scale, scale, size=dshape).astype(dtype)
        ref_res = tvm.topi.testing.conv2d_nchw_python(
            data.astype(out_dtype), dkernel.astype(out_dtype), 1, padding, groups=groups
        )

        op_res1 = relay.create_executor("graph", device=dev, target=target).evaluate(func)(
            data, kernel
        )
        tvm.testing.assert_allclose(op_res1.numpy(), ref_res, rtol=1e-4, atol=1e-4)


def test_compile_depthwise_conv2d_arm_cpu():
    dtype = "float32"
    out_dtype = "float32"
    scale = 1
    dshape = (1, 512, 32, 32)
    kshape = (512, 1, 3, 3)
    padding = (1, 1)
    channels = 512
    groups = 512
    kernel_size = (3, 3)
    dilation = (1, 1)

    x = relay.var("x", shape=dshape, dtype=dtype)
    w = relay.var("w", shape=kshape, dtype=dtype)
    y = relay.nn.conv2d(
        x,
        w,
        padding=padding,
        dilation=dilation,
        groups=groups,
        channels=channels,
        kernel_size=kernel_size,
    )
    func = relay.Function([x, w], y)
    mod = tvm.IRModule()
    mod["main"] = func

    test_schedule = '{"i": ["llvm -device=arm_cpu", "depthwise_conv2d_nchw_spatial_pack.arm_cpu", \
                    [["TENSOR", [1, 512, 32, 32], "float32"], \
                    ["TENSOR", [512, 1, 3, 3], "float32"], \
                    [1, 1], [1, 1], [1, 1], "float32"], {}, \
                    ["depthwise_conv2d_nchw_spatial_pack.arm_cpu", [1, 512, 32, 32, "float32"], \
                    [512, 1, 3, 3, "float32"], [1, 1], [1, 1], [1, 1], "float32"], \
                    {"i": 743640, "t": "", "c": null, \
                    "e": [["tile_co", "sp", [32, 16]], ["tile_oh", "sp", [8, 1]], \
                    ["tile_ow", "sp", [1, 8]], \
                    ["reorder_0", "re", [0, 1, 2, 3, 4, 5, 8, 6, 7]], \
                    ["reorder_1", "re", [0, 1, 2, 3, 6, 4, 5]], \
                    ["ann_reduce", "an", ["unroll", "none"]], \
                    ["ann_spatial", "an", ["unroll", "unroll", "vec"]], \
                    ["data_pad_inline", "ot", 4], ["data_vec_inline", "ot", 1], \
                    ["conv_inline", "ot", 0]]}], "r": [[0.0002933163], \
                    0, 3.1976189613342285, 1570811630.6058347], "v": 0.1}'
    temp = utils.tempdir()
    with open(temp.relpath("temp.log"), "w") as log_file:
        log_file.write(test_schedule)
    with autotvm.apply_history_best(temp.relpath("temp.log")):
        with tvm.transform.PassContext(opt_level=3):
            print("Compiling...")
            graph_json, mod, params = tvm.relay.build(mod, target="llvm -device=arm_cpu")


@tvm.testing.uses_gpu
def test_conv2d_winograd():
    class WinogradFallback(autotvm.FallbackContext):
        def _query_inside(self, target, workload):
            key = (target, workload)
            if key in self.memory:
                return self.memory[key]
            cfg = autotvm.task.space.FallbackConfigEntity()
            cfg.is_fallback = False
            cfg.cost = 0.1 if "winograd" in workload[0] else 1
            cfg["tile_b"] = autotvm.task.space.SplitEntity([-1, 1, 1, 1])
            cfg["tile_y"] = autotvm.task.space.SplitEntity([-1, 1, 1, 1])
            cfg["tile_x"] = autotvm.task.space.SplitEntity([-1, 1, 1, 1])
            cfg["tile_rc"] = autotvm.task.space.SplitEntity([-1, 1])
            cfg["auto_unroll_max_step"] = autotvm.task.space.OtherOptionEntity(1500)
            cfg["unroll_explicit"] = autotvm.task.space.OtherOptionEntity(1)
            self.memory[key] = cfg
            return cfg

    def run_test_conv2d_cuda(
        dtype, out_dtype, scale, dshape, kshape, padding=(1, 1), groups=1, dilation=(1, 1), **attrs
    ):

        x = relay.var("x", shape=dshape, dtype=dtype)
        w = relay.var("w", shape=kshape, dtype=dtype)
        y = relay.nn.conv2d(x, w, padding=padding, dilation=dilation, groups=groups, **attrs)
        func = relay.Function([x, w], y)
        mod = tvm.IRModule()
        mod["main"] = func
        mod = relay.transform.InferType()(mod)

        data = np.random.uniform(-scale, scale, size=dshape).astype(dtype)
        kernel = np.random.uniform(-scale, scale, size=kshape).astype(dtype)
        ref_res = tvm.topi.testing.conv2d_nchw_python(
            data.astype(out_dtype), kernel.astype(out_dtype), 1, padding, groups=groups
        )

        with WinogradFallback(), tvm.transform.PassContext(opt_level=3):
            for target, dev in tvm.testing.enabled_targets():
                if target != "cuda":
                    continue
                dev = tvm.device(target, 0)
                params = {"w": tvm.nd.array(kernel)}
                graph, lib, params = relay.build_module.build(mod, target=target, params=params)
                module = tvm.contrib.graph_executor.create(graph, lib, dev)
                module.set_input("x", tvm.nd.array(data))
                module.set_input(**params)
                module.run()
                op_res1 = module.get_output(0)
                tvm.testing.assert_allclose(op_res1.numpy(), ref_res, rtol=1e-3, atol=1e-3)

    # normal winograd: stride 1, padding 1, kernel 3x3
    dshape = (1, 80, 73, 73)
    kshape = (192, 80, 3, 3)
    run_test_conv2d_cuda(
        "float32", "float32", 1, dshape, kshape, padding=(1, 1), channels=192, kernel_size=(3, 3)
    )
    # extended winograd: stride 1, padding N, kernel 3x3
    run_test_conv2d_cuda(
        "float32", "float32", 1, dshape, kshape, padding=(0, 0), channels=192, kernel_size=(3, 3)
    )
    run_test_conv2d_cuda(
        "float32", "float32", 1, dshape, kshape, padding=(2, 2), channels=192, kernel_size=(3, 3)
    )
    # extended winograd: stride 1, padding N, kernel NxN
    kshape = (192, 80, 7, 7)
    run_test_conv2d_cuda(
        "float32", "float32", 1, dshape, kshape, padding=(2, 2), channels=192, kernel_size=(7, 7)
    )


@tvm.testing.uses_gpu
def test_conv3d_infer_type():
    # symbolic in batch dimension
    n, c, d, h, w = te.size_var("n"), 10, 224, 224, 224
    x = relay.var("x", relay.ty.TensorType((n, c, d, h, w), "float32"))
    w = relay.var("w")
    y = relay.nn.conv3d(x, w, kernel_size=(3, 3, 3), padding=(1, 1, 1), channels=2)
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, 2, 224, 224, 224), "float32")
    assert yy.args[1].checked_type == relay.TensorType((2, 10, 3, 3, 3), "float32")

    # infer by shape of w, mixed precision
    n, c, d, h, w = te.size_var("n"), 10, 224, 224, 224
    x = relay.var("x", relay.TensorType((n, c, d, h, w), "int8"))
    w = relay.var("w", relay.TensorType((2, 10, 3, 3, 3), "int8"))
    y = relay.nn.conv3d(x, w, out_dtype="int32")
    assert 'out_dtype="int32"' in y.astext()
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, 2, 222, 222, 222), "int32")

    # infer shape in case of different dtypes for input and weight.
    n, c, d, h, w = te.size_var("n"), 10, 224, 224, 224
    x = relay.var("x", relay.TensorType((n, c, d, h, w), "uint8"))
    w = relay.var("w", relay.TensorType((2, 10, 3, 3, 3), "int8"))
    y = relay.nn.conv3d(x, w, out_dtype="int32")
    assert 'out_dtype="int32"' in y.astext()
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, 2, 222, 222, 222), "int32")

    # Infer with NDHWC
    n, c, d, h, w = 4, 32, 224, 224, 224
    x = relay.var("x", relay.TensorType((n, d, h, w, c), "int8"))
    wt = relay.var("w")
    y = relay.nn.conv3d(
        x,
        wt,
        kernel_size=(3, 3, 3),
        padding=(1, 1, 1),
        channels=16,
        data_layout="NDHWC",
        out_dtype="int32",
    )
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, d, h, w, 16), "int32")

    # Infer with groups
    x = relay.var("x", relay.TensorType((1, 16, 224, 224, 224), "float32"))
    w = relay.var("w", relay.TensorType((4, 4, 1, 1, 1), "float32"))
    y = relay.nn.conv3d(x, w, groups=4, kernel_size=(1, 1, 1), channels=4)
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((1, 4, 224, 224, 224), "float32")


@tvm.testing.uses_gpu
def test_conv3d_run():
    def run_test_conv3d(
        dtype,
        out_dtype,
        scale,
        dshape,
        kshape,
        padding=(1, 1, 1),
        fref=None,
        groups=1,
        dilation=(1, 1, 1),
        except_targets=None,
        **attrs,
    ):
        if except_targets is None:
            except_targets = []

        x = relay.var("x", shape=dshape, dtype=dtype)
        w = relay.var("w", dtype=dtype)
        y = relay.nn.conv3d(x, w, padding=padding, dilation=dilation, groups=groups, **attrs)
        func = relay.Function([x, w], y)
        data = np.random.uniform(-scale, scale, size=dshape).astype(dtype)
        kernel = np.random.uniform(-scale, scale, size=kshape).astype(dtype)
        dkernel = tvm.topi.testing.dilate_python(kernel, (1, 1) + dilation)
        if fref is None:
            ref_res = tvm.topi.testing.conv3d_ncdhw_python(
                data.astype(out_dtype), dkernel.astype(out_dtype), 1, padding, groups=groups
            )
        else:
            ref_res = fref(data.astype(out_dtype), dkernel.astype(out_dtype))

        for target, dev in tvm.testing.enabled_targets():
            if target in except_targets:
                continue
            dev = tvm.device(target, 0)

            op_res1 = relay.create_executor("graph", device=dev, target=target).evaluate(func)(
                data, kernel
            )
            tvm.testing.assert_allclose(op_res1.numpy(), ref_res, rtol=1e-5, atol=1e-5)

    # normal conv3d
    dshape = (1, 3, 5, 224, 224)
    kshape = (10, 3, 3, 3, 3)
    run_test_conv3d(
        "float32",
        "float32",
        1,
        dshape,
        kshape,
        padding=(1, 1, 1),
        channels=10,
        kernel_size=(3, 3, 3),
    )


@tvm.testing.uses_gpu
def test_conv3d_ndhwc_run():
    def run_test_conv3d(
        dtype,
        out_dtype,
        scale,
        dshape,
        kshape,
        padding=(1, 1, 1),
        fref=None,
        groups=1,
        dilation=(1, 1, 1),
        except_targets=None,
        **attrs,
    ):
        if except_targets is None:
            except_targets = []

        x = relay.var("x", shape=dshape, dtype=dtype)
        w = relay.var("w", dtype=dtype)
        y = relay.nn.conv3d(
            x,
            w,
            padding=padding,
            dilation=dilation,
            groups=groups,
            data_layout="NDHWC",
            kernel_layout="DHWIO",
            **attrs,
        )
        func = relay.Function([x, w], y)
        data = np.random.uniform(-scale, scale, size=dshape).astype(dtype)
        kernel = np.random.uniform(-scale, scale, size=kshape).astype(dtype)
        dkernel = tvm.topi.testing.dilate_python(kernel, (1, 1) + dilation)
        if fref is None:
            ref_res = tvm.topi.testing.conv3d_ndhwc_python(
                data.astype(out_dtype), dkernel.astype(out_dtype), 1, padding
            )
        else:
            ref_res = fref(data.astype(out_dtype), dkernel.astype(out_dtype))

        for target, dev in tvm.testing.enabled_targets():
            if target in except_targets:
                continue
            dev = tvm.device(target, 0)

            op_res1 = relay.create_executor("graph", device=dev, target=target).evaluate(func)(
                data, kernel
            )
            tvm.testing.assert_allclose(op_res1.numpy(), ref_res, rtol=1e-5, atol=1e-5)

    # normal conv3d
    dshape = (1, 5, 224, 224, 6)
    kshape = (3, 3, 3, 6, 10)
    run_test_conv3d(
        "float32",
        "float32",
        1,
        dshape,
        kshape,
        padding=(1, 1, 1),
        channels=10,
        kernel_size=(3, 3, 3),
        except_targets=["cuda"],
    )


@tvm.testing.uses_gpu
def test_conv3d_winograd():
    class WinogradFallback(autotvm.FallbackContext):
        def _query_inside(self, target, workload):
            key = (target, workload)
            if key in self.memory:
                return self.memory[key]
            cfg = autotvm.task.space.FallbackConfigEntity()
            cfg.is_fallback = False
            cfg.cost = 0.1 if "winograd" in workload[0] else 1
            cfg["tile_b"] = autotvm.task.space.SplitEntity([-1, 1, 1, 1])
            cfg["tile_y"] = autotvm.task.space.SplitEntity([-1, 1, 1, 1])
            cfg["tile_x"] = autotvm.task.space.SplitEntity([-1, 1, 1, 1])
            cfg["tile_rc"] = autotvm.task.space.SplitEntity([-1, 1])
            cfg["auto_unroll_max_step"] = autotvm.task.space.OtherOptionEntity(0)
            cfg["unroll_explicit"] = autotvm.task.space.OtherOptionEntity(1)
            self.memory[key] = cfg
            return cfg

    def run_test_conv3d_cuda(
        dtype,
        out_dtype,
        scale,
        dshape,
        kshape,
        padding=(1, 1, 1),
        groups=1,
        dilation=(1, 1, 1),
        prepack=False,
        **attrs,
    ):

        x = relay.var("x", shape=dshape, dtype=dtype)
        w = relay.var("w", shape=kshape, dtype=dtype)
        if prepack:
            tile_size = _infer_tile_size(np.zeros(shape=dshape), np.zeros(shape=kshape))
            w_packed = relay.nn.contrib_conv3d_winograd_weight_transform(w, tile_size)

            y = relay.nn.contrib_conv3d_winograd_without_weight_transform(
                x,
                w_packed,
                tile_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
                channels=kshape[0],
                **attrs,
            )
        else:
            y = relay.nn.conv3d(x, w, padding=padding, dilation=dilation, groups=groups, **attrs)
        func = relay.Function([x, w], y)
        mod = tvm.IRModule()
        mod["main"] = func
        mod = relay.transform.InferType()(mod)

        data = np.random.uniform(-scale, scale, size=dshape).astype(dtype)
        kernel = np.random.uniform(-scale, scale, size=kshape).astype(dtype)
        ref_res = tvm.topi.testing.conv3d_ncdhw_python(
            data.astype(out_dtype), kernel.astype(out_dtype), 1, padding, groups=groups
        )

        with WinogradFallback(), tvm.transform.PassContext(opt_level=3):
            for target, dev in tvm.testing.enabled_targets():
                if target != "cuda":
                    continue
                dev = tvm.device(target, 0)
                params = {"w": tvm.nd.array(kernel)}
                graph, lib, params = relay.build_module.build(mod, target=target, params=params)
                module = tvm.contrib.graph_executor.create(graph, lib, dev)
                module.set_input("x", tvm.nd.array(data))
                module.set_input(**params)
                module.run()
                op_res1 = module.get_output(0)
                tvm.testing.assert_allclose(op_res1.numpy(), ref_res, rtol=1e-3, atol=1e-3)

    # normal winograd: stride 1, padding 1, kernel 3x3x3
    dshape = (1, 32, 16, 16, 16)
    kshape = (64, 32, 3, 3, 3)
    run_test_conv3d_cuda(
        "float32", "float32", 1, dshape, kshape, padding=(1, 1, 1), kernel_size=(3, 3, 3)
    )
    # Without depth transform using 1x3x3 kernel.
    kshape = (64, 32, 1, 3, 3)
    run_test_conv3d_cuda(
        "float32", "float32", 1, dshape, kshape, padding=(0, 1, 1), kernel_size=(1, 3, 3)
    )

    # extended winograd: stride 1, padding N, kernel NxNxN
    dshape = (1, 61, 20, 20, 20)
    kshape = (120, 61, 5, 5, 5)
    run_test_conv3d_cuda(
        "float32",
        "float32",
        1,
        dshape,
        kshape,
        padding=(2, 2, 2),
        channels=120,
        kernel_size=(5, 5, 5),
    )
    # Without depth transform
    kshape = (120, 61, 1, 5, 5)
    run_test_conv3d_cuda(
        "float32",
        "float32",
        1,
        dshape,
        kshape,
        padding=(0, 2, 2),
        channels=120,
        kernel_size=(1, 5, 5),
    )


@tvm.testing.uses_gpu
def test_conv3d_transpose_infer_type():
    # symbolic in batch dimension
    n, c, d, h, w = te.size_var("n"), 10, 224, 224, 224
    x = relay.var("x", relay.ty.TensorType((n, c, d, h, w), "float32"))
    w = relay.var("w")
    y = relay.nn.conv3d_transpose(x, w, kernel_size=(3, 3, 3), padding=(1, 1, 1), channels=2)
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, 2, 224, 224, 224), "float32")

    assert yy.args[1].checked_type == relay.TensorType((10, 2, 3, 3, 3), "float32")

    # infer by shape of w, mixed precision
    n, c, d, h, w = te.size_var("n"), 10, 224, 224, 224
    x = relay.var("x", relay.TensorType((n, c, d, h, w), "int8"))
    w = relay.var("w", relay.TensorType((10, 12, 3, 3, 3), "int8"))
    y = relay.nn.conv3d_transpose(x, w, out_dtype="int32")
    assert 'out_dtype="int32"' in y.astext()
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, 12, 226, 226, 226), "int32")

    # infer shape in case of different dtypes for input and weight.
    n, c, d, h, w = te.size_var("n"), 10, 224, 224, 224
    x = relay.var("x", relay.TensorType((n, c, d, h, w), "uint8"))
    w = relay.var("w", relay.TensorType((10, 12, 3, 3, 3), "int8"))
    y = relay.nn.conv3d_transpose(x, w, out_dtype="int32")
    assert 'out_dtype="int32"' in y.astext()
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, 12, 226, 226, 226), "int32")


@tvm.testing.uses_gpu
def test_conv3d_transpose_ncdhw_run():
    dshape = (1, 3, 24, 24, 24)
    kshape = (3, 4, 2, 2, 2)

    x = relay.var("x", shape=dshape)
    w = relay.var("w")
    y = relay.nn.conv3d_transpose(
        x, w, channels=4, kernel_size=(2, 2, 2), strides=(1, 1, 1), padding=(1, 1, 1)
    )
    func = relay.Function([x, w], y)
    dtype = "float32"

    data = np.random.uniform(size=dshape).astype(dtype)
    kernel = np.random.uniform(size=kshape).astype(dtype)
    ref_res = tvm.topi.testing.conv3d_transpose_ncdhw_python(data, kernel, 1, 1, 0)

    for target, dev in tvm.testing.enabled_targets():
        op_res1 = relay.create_executor("graph", device=dev, target=target).evaluate(func)(
            data, kernel
        )
        tvm.testing.assert_allclose(op_res1.numpy(), ref_res, rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_conv2d_transpose_infer_type():
    # symbolic in batch dimension
    n, c, h, w = te.size_var("n"), 10, 10, 12
    x = relay.var("x", relay.TensorType((n, c, h, w), "float32"))
    w = relay.var("w", relay.IncompleteType())
    y = relay.nn.conv2d_transpose(x, w, kernel_size=(3, 3), padding=(1, 1), channels=15)
    assert "channels=15" in y.astext()
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, 15, 10, 12), "float32")
    assert yy.args[1].checked_type == relay.TensorType((10, 15, 3, 3), "float32")

    # infer by shape of w, mixed precision
    n, h, w, c = te.size_var("n"), 10, 10, 12
    x = relay.var("x", relay.TensorType((n, h, w, c), "float32"))
    w = relay.var("w", relay.TensorType((12, 11, 5, 5), "float32"))
    y = relay.nn.conv2d_transpose(x, w, output_padding=(1, 1), channels=11, data_layout="NHWC")
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, 15, 15, 11), "float32")


@tvm.testing.uses_gpu
def test_conv2d_transpose_nchw_run():
    k_layouts = {"OIHW": (10, 3, 3, 3), "IOHW": (3, 10, 3, 3)}
    output_padding = (1, 1)

    for k_layout, kshape in k_layouts.items():
        dshape = (1, 3, 18, 18)
        x = relay.var("x", shape=dshape)
        w = relay.var("w")
        y = relay.nn.conv2d_transpose(
            x,
            w,
            channels=10,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=(1, 1),
            output_padding=output_padding,
            kernel_layout=k_layout,
            data_layout="NCHW",
        )
        func = relay.Function([x, w], y)
        dtype = "float32"
        data = np.random.uniform(size=dshape).astype(dtype)
        kernel = np.random.uniform(size=kshape).astype(dtype)

        if k_layout != "IOHW":
            # Must be OIHW so switch
            kernel_iohw = np.transpose(kernel, [1, 0, 2, 3])
        else:
            kernel_iohw = kernel

        ref_res = tvm.topi.testing.conv2d_transpose_nchw_python(
            data, kernel_iohw, 2, 1, output_padding
        )

        enabled_targets = tvm.testing.enabled_targets()

        if cudnn.exists() and k_layout == "IOHW":
            enabled_targets.append(("cuda -libs=cudnn", tvm.cuda(0)))

        for target, dev in enabled_targets:
            op_res1 = relay.create_executor("graph", device=dev, target=target).evaluate(func)(
                data, kernel
            )
            tvm.testing.assert_allclose(op_res1.numpy(), ref_res, rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_conv2d_transpose_nhwc_run():
    dshape_nhwc = (1, 18, 18, 3)
    kshape_hwoi = (3, 3, 10, 3)
    x = relay.var("x", shape=dshape_nhwc)
    w = relay.var("w")

    y = relay.nn.conv2d_transpose(
        x,
        w,
        channels=10,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=(1, 1),
        output_padding=(1, 1),
        data_layout="NHWC",
        kernel_layout="HWOI",
    )
    func = relay.Function([x, w], y)
    dtype = "float32"
    data = np.random.uniform(size=dshape_nhwc).astype(dtype)
    kernel = np.random.uniform(size=kshape_hwoi).astype(dtype)

    ref_res = tvm.topi.testing.conv2d_transpose_nhwc_python(
        data, kernel, "HWOI", 2, 1, output_padding=(1, 1)
    )

    for target, dev in tvm.testing.enabled_targets():
        op_res1 = relay.create_executor("graph", device=dev, target=target).evaluate(func)(
            data, kernel
        )
        tvm.testing.assert_allclose(op_res1.numpy(), ref_res, rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_conv2d_transpose_nhwc_cudnn():
    if not cudnn.exists():
        return

    dshape_nhwc = (1, 18, 18, 3)
    kshape_ihwo = (3, 3, 3, 10)
    x = relay.var("x", shape=dshape_nhwc)
    w = relay.var("w", shape=kshape_ihwo)

    y = relay.nn.conv2d_transpose(
        x,
        w,
        channels=10,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=(1, 1),
        output_padding=(1, 1),
        data_layout="NHWC",
        kernel_layout="IHWO",
    )
    func = relay.Function([x, w], y)
    dtype = "float32"
    data = np.random.uniform(size=dshape_nhwc).astype(dtype)
    kernel = np.random.uniform(size=kshape_ihwo).astype(dtype)

    ref_res = tvm.topi.testing.conv2d_transpose_nhwc_python(
        data, np.transpose(kernel, [1, 2, 3, 0]), "HWOI", 2, 1, output_padding=(1, 1)
    )

    target = "cuda -libs=cudnn"
    dev = tvm.cuda(0)

    op_res1 = relay.create_executor("graph", device=dev, target=target).evaluate(func)(data, kernel)
    tvm.testing.assert_allclose(op_res1.numpy(), ref_res, rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_conv1d_transpose_ncw_run():
    dshape = (1, 3, 18)
    kshape = (3, 10, 3)
    oshape = (1, 10, 36)
    x = relay.var("x", shape=dshape)
    w = relay.var("w")
    y = relay.nn.conv1d_transpose(
        x, w, channels=10, kernel_size=(3,), strides=(2,), padding=(1,), output_padding=(1,)
    )
    func = relay.Function([x, w], y)
    dtype = "float32"
    data = np.random.uniform(size=dshape).astype(dtype)
    kernel = np.random.uniform(size=kshape).astype(dtype)
    ref_res = tvm.topi.testing.conv1d_transpose_ncw_python(data, kernel, 2, 1, output_padding=(1,))

    for target, dev in tvm.testing.enabled_targets():
        op_res1 = relay.create_executor("graph", device=dev, target=target).evaluate(func)(
            data, kernel
        )
        tvm.testing.assert_allclose(op_res1.numpy(), ref_res, rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_upsampling_infer_type():
    n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), te.size_var("w")
    scale = tvm.tir.const(2.0, "float64")
    x = relay.var("x", relay.TensorType((n, c, h, w), "float32"))
    y = relay.nn.upsampling(x, scale_h=2, scale_w=2, layout="NCHW", method="bilinear")
    'method="BINLINEAR"' in y.astext()
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType(
        (
            n,
            c,
            tvm.tir.Cast("int32", te.round(h * scale)),
            tvm.tir.Cast("int32", te.round(w * scale)),
        ),
        "float32",
    )
    n, c = te.size_var("n"), te.size_var("c")
    x = relay.var("x", relay.TensorType((n, c, 100, 200), "float32"))
    y = relay.nn.upsampling(x, scale_h=2, scale_w=2, layout="NCHW", method="bilinear")
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, c, 200, 400), "float32")


@tvm.testing.uses_gpu
def test_upsampling3d_infer_type():
    n, c, d, h, w = (
        te.size_var("n"),
        te.size_var("c"),
        te.size_var("d"),
        te.size_var("h"),
        te.size_var("w"),
    )
    scale = tvm.tir.const(2.0, "float64")
    x = relay.var("x", relay.TensorType((n, c, d, h, w), "float32"))
    y = relay.nn.upsampling3d(
        x, scale_d=2, scale_h=2, scale_w=2, layout="NCDHW", method="trilinear"
    )

    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType(
        (
            n,
            c,
            tvm.tir.Cast("int32", te.round(d * scale)),
            tvm.tir.Cast("int32", te.round(h * scale)),
            tvm.tir.Cast("int32", te.round(w * scale)),
        ),
        "float32",
    )
    n, c = te.size_var("n"), te.size_var("c")
    x = relay.var("x", relay.TensorType((n, c, 100, 100, 200), "float32"))
    y = relay.nn.upsampling3d(
        x, scale_d=2, scale_h=2, scale_w=2, layout="NCDHW", method="trilinear"
    )
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, c, 200, 200, 400), "float32")


def _test_global_pool2d(opfunc, reffunc):
    n, c, h, w = te.size_var("n"), te.size_var("c"), 224, 224
    x = relay.var("x", relay.TensorType((n, h, w, c), "float32"))
    y = opfunc(x, layout="NHWC")
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, 1, 1, c), "float32")

    n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), te.size_var("w")
    x = relay.var("x", relay.TensorType((n, c, h, w), "float32"))
    y = opfunc(x)
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, c, 1, 1), "float32")
    # test execution
    dtype = "float32"
    dshape = (1, 1024, 7, 7)
    x = relay.var("x", shape=dshape)
    y = opfunc(x)
    func = relay.Function([x], y)
    data = np.random.uniform(size=dshape).astype(dtype)
    ref_res = reffunc(data, axis=(2, 3), keepdims=True)
    for target, dev in tvm.testing.enabled_targets():
        op_res1 = relay.create_executor("graph", device=dev, target=target).evaluate(func)(data)
        tvm.testing.assert_allclose(op_res1.numpy(), ref_res, rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_pool2d():
    def _test_pool2d(opfunc, pool_type, pool_size=2, strides=2, dilation=1, padding=0):
        n, c, h, w = te.size_var("n"), 10, 224, 224
        x = relay.var("x", relay.TensorType((n, c, h, w), "float32"))
        y = opfunc(x, pool_size=(1, 1))
        assert "pool_size=" in y.astext()
        yy = run_infer_type(y)
        assert yy.checked_type == relay.TensorType((n, 10, 224, 224), "float32")
        # test execution
        dtype = "float32"
        dshape = (1, 3, 28, 28)
        x = relay.var("x", shape=dshape)
        y = opfunc(x, pool_size=pool_size, strides=strides, dilation=dilation, padding=padding)
        func = relay.Function([x], y)
        data = np.random.uniform(size=dshape).astype(dtype)
        ref_res = tvm.topi.testing.poolnd_python(
            data,
            [pool_size, pool_size],
            [strides, strides],
            [dilation, dilation],
            [padding, padding],
            [padding, padding],
            pool_type,
            count_include_pad=False,
            ceil_mode=False,
        )
        for target, dev in tvm.testing.enabled_targets():
            op_res1 = relay.create_executor("graph", device=dev, target=target).evaluate(func)(data)
            tvm.testing.assert_allclose(op_res1.numpy(), ref_res, rtol=1e-5, atol=1e-5)

    def _test_pool2d_int(opfunc, reffunc, dtype):
        n, c, h, w = te.size_var("n"), 10, 224, 224
        x = relay.var("x", relay.TensorType((n, c, h, w), dtype))
        y = opfunc(x, pool_size=(1, 1))
        assert "pool_size=" in y.astext()
        yy = run_infer_type(y)
        assert yy.checked_type == relay.TensorType((n, 10, 224, 224), dtype)
        # test execution
        dtype = "int32"
        dshape = (1, 3, 28, 28)
        for shape_dtype in ["int32", "int64"]:
            x = relay.var("x", shape=[tvm.tir.IntImm(shape_dtype, x) for x in dshape], dtype=dtype)
            y = opfunc(x, pool_size=(2, 2), strides=(2, 2), padding=(0, 0))
            func = relay.Function([x], y)
            data = np.random.randint(low=-128, high=128, size=dshape)
            ref_res = reffunc(data.reshape(1, 3, 14, 2, 14, 2), axis=(3, 5)).astype(dtype)
            for target, dev in tvm.testing.enabled_targets():
                op_res1 = relay.create_executor("graph", device=dev, target=target).evaluate(func)(
                    data
                )
                tvm.testing.assert_allclose(op_res1.numpy(), ref_res, rtol=1e-5, atol=1e-5)

    _test_pool2d(relay.nn.max_pool2d, "max")
    _test_pool2d(relay.nn.max_pool2d, "max", pool_size=2, strides=2, padding=0)
    _test_pool2d(relay.nn.max_pool2d, "max", pool_size=2, strides=2, padding=0, dilation=2)
    _test_pool2d(relay.nn.avg_pool2d, "avg")
    _test_pool2d(relay.nn.avg_pool2d, "avg", pool_size=2, strides=2, padding=0)
    _test_pool2d(relay.nn.avg_pool2d, "avg", pool_size=2, strides=2, padding=0, dilation=2)

    _test_pool2d_int(relay.nn.avg_pool2d, np.mean, "int32")
    _test_pool2d_int(relay.nn.avg_pool2d, np.mean, "uint16")
    _test_global_pool2d(relay.nn.global_max_pool2d, np.max)
    _test_global_pool2d(relay.nn.global_avg_pool2d, np.mean)


def _test_global_pool1d(opfunc, reffunc):
    n, c, w = te.size_var("n"), te.size_var("c"), 224
    x = relay.var("x", relay.TensorType((n, w, c), "float32"))
    y = opfunc(x, layout="NWC")
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, 1, c), "float32")

    n, c, w = te.size_var("n"), te.size_var("c"), te.size_var("w")
    x = relay.var("x", relay.TensorType((n, c, w), "float32"))
    y = opfunc(x)
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, c, 1), "float32")
    # test execution
    dtype = "float32"
    dshape = (1, 1024, 7)
    x = relay.var("x", shape=dshape)
    y = opfunc(x)
    func = relay.Function([x], y)
    data = np.random.uniform(size=dshape).astype(dtype)
    ref_res = reffunc(data, axis=(2,), keepdims=True)
    for target, dev in tvm.testing.enabled_targets():
        op_res1 = relay.create_executor("graph", device=dev, target=target).evaluate(func)(data)
        tvm.testing.assert_allclose(op_res1.numpy(), ref_res, rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_pool1d():
    def _test_pool1d(
        opfunc, pool_type, pool_size=2, strides=2, dilation=1, padding=0, dtype="float32"
    ):
        n, c, w = te.var("n"), 10, 224
        x = relay.var("x", relay.TensorType((n, c, w), "float32"))
        y = opfunc(x, pool_size=(1,))
        assert "pool_size=" in y.astext()
        yy = run_infer_type(y)
        assert yy.checked_type == relay.TensorType((n, 10, 224), "float32")
        # test execution
        dshape = (1, 3, 32)
        for shape_dtype in ["int32", "int64"]:
            x = relay.var("x", shape=[tvm.tir.IntImm(shape_dtype, x) for x in dshape], dtype=dtype)
            pool_type = "max" if "max" in str(opfunc) else "avg"
            y = opfunc(x, pool_size=pool_size, strides=strides, dilation=dilation, padding=padding)
            func = relay.Function([x], y)
            data = np.random.uniform(size=dshape).astype(dtype)
            ref_res = tvm.topi.testing.poolnd_python(
                data,
                [pool_size],
                [strides],
                [dilation],
                [padding],
                [padding],
                pool_type,
                count_include_pad=False,
                ceil_mode=False,
            )
            for target, dev in tvm.testing.enabled_targets():
                op_res1 = relay.create_executor("graph", device=dev, target=target).evaluate(func)(
                    data
                )
                tvm.testing.assert_allclose(op_res1.numpy(), ref_res, rtol=1e-5, atol=1e-5)

    _test_pool1d(relay.nn.max_pool1d, "max")
    _test_pool1d(relay.nn.max_pool1d, "max", dtype="int32")
    _test_pool1d(relay.nn.max_pool1d, "max", pool_size=2, strides=2, padding=0)
    _test_pool1d(relay.nn.max_pool1d, "max", pool_size=2, strides=2, padding=0, dilation=2)
    _test_pool1d(relay.nn.avg_pool1d, "avg")
    _test_pool1d(relay.nn.avg_pool1d, "avg", dtype="int32")
    _test_pool1d(relay.nn.avg_pool1d, "avg", pool_size=2, strides=2, padding=0)
    _test_pool1d(relay.nn.avg_pool1d, "avg", pool_size=2, strides=2, padding=0, dilation=2)
    _test_global_pool1d(relay.nn.global_max_pool1d, np.max)
    _test_global_pool1d(relay.nn.global_avg_pool1d, np.mean)


@tvm.testing.uses_gpu
def test_pool3d():
    def _test_pool3d(
        opfunc,
        pool_type,
        pool_size=2,
        strides=2,
        dilation=1,
        padding=[0, 0, 0, 0, 0, 0],
        dtype="float32",
    ):
        n, c, d, h, w = te.size_var("n"), 10, 5, 224, 224
        x = relay.var("x", relay.TensorType((n, c, d, h, w), "float32"))
        y = opfunc(x, pool_size=(1, 1, 1))
        assert "pool_size=" in y.astext()
        yy = run_infer_type(y)
        assert yy.checked_type == relay.TensorType((n, 10, 5, 224, 224), "float32")
        # test execution
        dtype = "float32"
        dshape = (1, 3, 32, 32, 32)
        for shape_dtype in ["int32", "int64"]:
            x = relay.var("x", shape=[tvm.tir.IntImm(shape_dtype, x) for x in dshape], dtype=dtype)
            pool_type = "max" if "max" in str(opfunc) else "avg"
            y = opfunc(
                x,
                pool_size=pool_size,
                strides=strides,
                padding=padding,
                dilation=dilation,
            )
            func = relay.Function([x], y)
            data = np.random.uniform(size=dshape).astype(dtype)
            ref_res = tvm.topi.testing.poolnd_python(
                data,
                [pool_size, pool_size, pool_size],
                [strides, strides, strides],
                [dilation, dilation, dilation],
                padding[:3],
                padding[3:],
                pool_type,
                count_include_pad=False,
                ceil_mode=False,
            )
            for target, dev in tvm.testing.enabled_targets():
                op_res1 = relay.create_executor("graph", device=dev, target=target).evaluate(func)(
                    data
                )
                tvm.testing.assert_allclose(op_res1.numpy(), ref_res, rtol=1e-5, atol=1e-5)

    _test_pool3d(relay.nn.max_pool3d, "max")
    _test_pool3d(relay.nn.max_pool3d, "max", dtype="int32")
    _test_pool3d(relay.nn.max_pool3d, "max", padding=(2, 0, 0, 2, 0, 0))
    _test_pool3d(relay.nn.max_pool3d, "max", padding=(0, 3, 0, 0, 3, 0))
    _test_pool3d(relay.nn.max_pool3d, "max", padding=(0, 0, 4, 0, 0, 4))
    _test_pool3d(relay.nn.max_pool3d, "max", pool_size=2, strides=2)
    _test_pool3d(relay.nn.max_pool3d, "max", pool_size=2, strides=2, dilation=2)
    _test_pool3d(relay.nn.avg_pool3d, "avg")
    _test_pool3d(relay.nn.avg_pool3d, "avg", dtype="int32")
    _test_pool3d(relay.nn.avg_pool3d, "avg", padding=(2, 0, 0, 2, 0, 0))
    _test_pool3d(relay.nn.avg_pool3d, "avg", padding=(0, 3, 0, 0, 3, 0))
    _test_pool3d(relay.nn.avg_pool3d, "avg", padding=(0, 0, 4, 0, 0, 4))
    _test_pool3d(relay.nn.avg_pool3d, "avg", pool_size=2, strides=2)
    _test_pool3d(relay.nn.avg_pool3d, "avg", pool_size=2, strides=2, dilation=2)


@tvm.testing.uses_gpu
def test_avg_pool2d_no_count_pad():
    kh, kw = (4, 4)
    sh, sw = (2, 2)
    ph, pw = (2, 2)
    n = 1
    (ic, ih, iw) = (3, 28, 28)
    (oc, oh, ow) = (3, 15, 15)
    dshape = (n, ic, ih, iw)
    x = relay.var("x", shape=dshape)
    y = relay.nn.avg_pool2d(
        x, pool_size=(kh, kw), strides=(sw, sw), padding=(ph, pw), count_include_pad=False
    )
    func = relay.Function([x], y)
    dtype = "float32"
    a_np = np.random.uniform(low=0.001, size=(n, ic, ih, iw)).astype(dtype)
    pad_np = np.zeros(shape=(n, ic, ih + 2 * ph, iw + 2 * pw)).astype(dtype)
    no_zero = (range(n), range(ic), (range(ph, ih + ph)), (range(pw, iw + pw)))
    pad_np[np.ix_(*no_zero)] = a_np
    b_np = np.zeros(shape=(n, oc, oh, ow)).astype(dtype)
    for i in range(oh):
        for j in range(ow):
            pad_count = np.sum(
                pad_np[:, :, i * sh : i * sh + kh, j * sw : j * sw + kw] > 0, axis=(2, 3)
            )
            b_np[:, :, i, j] = np.sum(
                pad_np[:, :, i * sh : i * sh + kh, j * sw : j * sw + kw], axis=(2, 3)
            ) / np.maximum(pad_count, 1)
    ref_res = np.maximum(b_np, 0.0)
    data = a_np

    for target, dev in tvm.testing.enabled_targets():
        op_res1 = relay.create_executor("graph", device=dev, target=target).evaluate(func)(data)
        tvm.testing.assert_allclose(op_res1.numpy(), ref_res, rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_flatten_infer_type(executor_kind):
    d1, d2, d3, d4 = te.size_var("d1"), te.size_var("d2"), te.size_var("d3"), te.size_var("d4")
    x = relay.var("x", relay.TensorType((d1, d2, d3, d4), "float32"))
    y = relay.nn.batch_flatten(x)
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((d1, ((d2 * d3) * d4)), "float32")

    x = relay.var("x", relay.TensorType((3, 2, 4, 3), "float32"))
    y = relay.nn.batch_flatten(x)
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((3, 24), "float32")

    x = relay.var("x", relay.TensorType((d1, 2, d3, 3), "float32"))
    y = relay.nn.batch_flatten(x)
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((d1, ((2 * d3) * 3)), "float32")

    shape = (1, 5, 10, 10)
    o_shape = (1, 500)
    dtype = "float32"
    x = relay.var("x", relay.TensorType(shape, dtype))
    z = relay.nn.batch_flatten(x)
    yy = run_infer_type(z)
    assert yy.checked_type == relay.TensorType(o_shape, dtype)
    func = relay.Function([x], z)
    x_data = np.random.uniform(low=-1, high=1, size=shape).astype(dtype)
    ref_res = x_data.flatten().reshape(o_shape)

    for target, dev in tvm.testing.enabled_targets():
        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            x_data
        )
        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)


@tvm.testing.uses_gpu
def test_pad_infer_type():
    # entirely concrete cases
    n, c, h, w = 1, 2, 3, 4
    t = relay.var("t", relay.TensorType((n, c, h, w), "float32"))
    y = relay.nn.pad(t, ((1, 1), (2, 2), (3, 3), (4, 4)))
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((3, 6, 9, 12), "float32")

    n, c, h, w = 4, 6, 3, 5
    t = relay.var("t", relay.TensorType((n, c, h, w), "float32"))
    y = relay.nn.pad(t, ((-1, -1), (2, -2), (0, -3), (4, 4)), pad_mode="reflect")
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((2, 6, 0, 13), "float32")

    # some symbolic values
    n, c, h, w = te.size_var("n"), 2, 3, te.size_var("w")
    t = relay.var("t", relay.TensorType((n, c, h, w), "float32"))
    y = relay.nn.pad(t, ((1, 1), (2, 2), (3, 3), (4, 4)))
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n + 2, 6, 9, w + 8), "float32")

    n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), te.size_var("w")
    t = relay.var("t", relay.TensorType((n, c, h, w), "float32"))
    y = relay.nn.pad(t, ((-1, -1), (-2, -2), (1, -3), (4, 4)))
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n + (-2), c + (-4), h + (-2), w + 8), "float32")

    # dealing with dynamic vals
    n, c, h, w = te.size_var("n"), 2, 3, te.size_var("w")
    t = relay.var("t", relay.TensorType((n, c, h, w), "float32"))
    y = relay.nn.pad(
        t, ((1, 1), (2, 2), (3, 3), (4, 4)), pad_value=relay.var("pad_value", "float32")
    )
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n + 2, 6, 9, w + 8), "float32")


def _get_numpy_pad(dshape, data, pad, pad_value=0):
    mod_pad = []
    for axis, (pad_x, pad_y) in enumerate(pad):
        indices = range(dshape[axis])
        if pad_x < 0:
            indices = indices[abs(pad_x) :]
            pad_x = 0
        if pad_y < 0:
            indices = indices[:pad_y]
            pad_y = 0
        data = np.take(data, indices, axis)
        mod_pad.append((pad_x, pad_y))
    return np.pad(data, tuple(mod_pad), "constant", constant_values=pad_value)


@tvm.testing.uses_gpu
def test_pad_run():
    def _test_run(dtype):
        dshape_list = [(4, 10, 7, 7), (4, 6, 3, 5)]
        pad_list = [((1, 1), (2, 2), (3, 3), (4, 4)), ((-1, -1), (2, -2), (0, -2), (4, 4))]

        for dshape, pad in zip(dshape_list, pad_list):
            x = relay.var("x", shape=dshape)
            y = relay.nn.pad(x, pad)
            func = relay.Function([x], y)
            data = np.random.uniform(size=dshape).astype(dtype)
            ref_res = _get_numpy_pad(dshape, data, pad)
            for target, dev in tvm.testing.enabled_targets():
                op_res1 = relay.create_executor("graph", device=dev, target=target).evaluate(func)(
                    data
                )
                tvm.testing.assert_allclose(op_res1.numpy(), ref_res, rtol=1e-5, atol=1e-5)

    _test_run("float32")
    _test_run("int32")


@tvm.testing.uses_gpu
def test_pad_run_dynamic_pad_value():
    def _test_run(dtype):
        dshape = (4, 6, 3, 5)
        pad = ((-1, -1), (2, -2), (0, -2), (4, 4))

        data = relay.var("data", shape=dshape, dtype=dtype)
        pad_value = relay.var("pad_value", dtype)
        pad_data = relay.nn.pad(data, pad, pad_value=pad_value)
        f = relay.Function([data, pad_value], pad_data)

        data_arr = np.random.uniform(-10, 10, size=dshape).astype(dtype)
        pad_value_arr = 2.0
        ref_res = _get_numpy_pad(dshape, data_arr, pad, pad_value=pad_value_arr)

        for target, dev in tvm.testing.enabled_targets():
            result = relay.create_executor(kind="graph", device=dev, target=target).evaluate(f)(
                data_arr, pad_value_arr
            )
            tvm.testing.assert_allclose(result.numpy(), ref_res, rtol=1e-5, atol=1e-5)

    _test_run("float32")
    _test_run("int32")


@tvm.testing.uses_gpu
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_lrn(executor_kind, dtype):
    n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), te.size_var("w")
    x = relay.var("x", shape=(n, c, h, w), dtype=dtype)
    y = relay.nn.lrn(x, size=10, axis=2, bias=0.5, alpha=0.00001, beta=0.75)
    "alpha=" in y.astext()
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, c, h, w), dtype)

    shape = (1, 5, 10, 10)
    x = relay.var("x", relay.TensorType(shape, dtype))
    size = 5
    axis = 1
    bias = 0.5
    alpha = 0.00001
    beta = 0.75
    z = relay.nn.lrn(x, size=size, axis=axis, bias=bias, alpha=alpha, beta=beta)
    yy = run_infer_type(z)
    assert yy.checked_type == relay.TensorType(shape, dtype)
    func = relay.Function([x], z)
    x_data = np.random.uniform(low=-1, high=1, size=shape).astype(dtype)
    ref_res = tvm.topi.testing.lrn_python(x_data, size, axis, bias, alpha, beta)

    for target, dev in tvm.testing.enabled_targets():
        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            x_data
        )
        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)


@tvm.testing.uses_gpu
def test_l2_normalize(executor_kind):
    n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), te.size_var("w")
    x = relay.var("x", shape=(n, c, h, w))
    y = relay.nn.l2_normalize(x, eps=0.001, axis=[1])
    "axis=" in y.astext()
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, c, h, w))

    shape = (1, 5, 10, 10)
    dtype = "float32"
    x = relay.var("x", relay.TensorType(shape, dtype))
    eps = 0.001
    axis = 1
    z = relay.nn.l2_normalize(x, eps=0.001, axis=[axis])
    yy = run_infer_type(z)
    assert yy.checked_type == relay.TensorType(shape, dtype)
    func = relay.Function([x], z)
    x_data = np.random.uniform(low=-1, high=1, size=shape).astype(dtype)
    ref_res = tvm.topi.testing.l2_normalize_python(x_data, eps, axis)

    for target, dev in tvm.testing.enabled_targets():
        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            x_data
        )
        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)


def batch_flatten(data):
    shape = data.shape
    target_dim = 1
    for i in range(len(shape) - 1):
        target_dim = target_dim * shape[i + 1]
    return np.reshape(data, (shape[0], target_dim))


@tvm.testing.uses_gpu
def test_batch_flatten():
    t1 = relay.TensorType((5, 10, 5))
    x = relay.Var("x", t1)
    func = relay.Function([x], relay.nn.batch_flatten(x))

    data = np.random.rand(5, 10, 5).astype(t1.dtype)
    ref_res = batch_flatten(data)
    for target, dev in tvm.testing.enabled_targets():
        op_res = relay.create_executor("graph", device=dev, target=target).evaluate(func)(data)
        np.testing.assert_allclose(op_res.numpy(), ref_res, rtol=0.01)


def _test_upsampling(layout, method, align_corners=False):
    n, c, h, w = te.size_var("n"), 16, 32, 32
    scale_h = 2.0
    scale_w = 2.0
    dtype = "float32"

    def get_shape():
        if layout == "NCHW":
            return (c, h, w), (c, int(round(h * scale_h)), int(round(w * scale_w)))
        else:
            return (h, w, c), (int(round(h * scale_h)), int(round(w * scale_w)), c)

    ishape, oshape = get_shape()
    x = relay.var("x", relay.TensorType((n,) + ishape, dtype))
    y = relay.nn.upsampling(
        x,
        scale_h=scale_h,
        scale_w=scale_w,
        layout=layout,
        method=method,
        align_corners=align_corners,
    )
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n,) + oshape, dtype)
    dshape = (1,) + ishape
    x = relay.var("x", shape=dshape)
    y = relay.nn.upsampling(
        x,
        scale_h=scale_h,
        scale_w=scale_w,
        layout=layout,
        method=method,
        align_corners=align_corners,
    )
    func = relay.Function([x], y)

    data = np.random.uniform(size=dshape).astype(dtype)
    ref = tvm.topi.testing.resize2d_python(
        data,
        (scale_h, scale_w),
        layout,
        method[2:] if method[0:2] == "bi" else method,
        "align_corners" if align_corners else "asymmetric",
    )
    for target, dev in tvm.testing.enabled_targets():
        out = relay.create_executor("graph", device=dev, target=target).evaluate(func)(data)
        tvm.testing.assert_allclose(out.numpy(), ref, rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_upsampling():
    _test_upsampling("NCHW", "nearest_neighbor")
    _test_upsampling("NCHW", "bilinear", True)
    _test_upsampling("NHWC", "nearest_neighbor")
    _test_upsampling("NHWC", "bilinear", True)


def _test_upsampling3d(layout, method, coordinate_transformation_mode="half_pixel"):
    n, c, d, h, w = te.size_var("n"), 8, 16, 16, 16
    scale_d = 2.0
    scale_h = 2.0
    scale_w = 2.0
    dtype = "float32"

    def get_shape():
        if layout == "NCDHW":
            return (c, d, h, w), (
                c,
                int(round(d * scale_d)),
                int(round(h * scale_h)),
                int(round(w * scale_w)),
            )
        else:
            return (d, h, w, c), (
                int(round(d * scale_d)),
                int(round(h * scale_h)),
                int(round(w * scale_w)),
                c,
            )

    ishape, oshape = get_shape()
    x = relay.var("x", relay.TensorType((n,) + ishape, dtype))
    y = relay.nn.upsampling3d(
        x,
        scale_d=scale_d,
        scale_h=scale_h,
        scale_w=scale_w,
        layout=layout,
        method=method,
        coordinate_transformation_mode=coordinate_transformation_mode,
    )

    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n,) + oshape, dtype)
    dshape = (1,) + ishape
    x = relay.var("x", shape=dshape)
    y = relay.nn.upsampling3d(
        x,
        scale_d=scale_d,
        scale_h=scale_h,
        scale_w=scale_w,
        layout=layout,
        method=method,
        coordinate_transformation_mode=coordinate_transformation_mode,
    )
    func = relay.Function([x], y)

    data = np.random.uniform(size=dshape).astype(dtype)
    ref = tvm.topi.testing.resize3d_python(
        data,
        (scale_d, scale_h, scale_w),
        layout,
        method[3:] if method[0:3] == "tri" else method,
        coordinate_transformation_mode,
    )
    for target, dev in tvm.testing.enabled_targets():
        out = relay.create_executor("graph", device=dev, target=target).evaluate(func)(data)
        tvm.testing.assert_allclose(out.numpy(), ref, rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_upsampling3d():
    _test_upsampling3d("NCDHW", "nearest_neighbor", "asymmetric")
    _test_upsampling3d("NCDHW", "trilinear", "align_corners")
    _test_upsampling3d("NDHWC", "nearest_neighbor", "asymmetric")
    _test_upsampling3d("NDHWC", "trilinear", "align_corners")


@tvm.testing.requires_x86
@pytest.mark.skipif(tvm.target.codegen.llvm_version_major() < 8, reason="Requires LLVM 8")
class TestConv2DInt8Intrinsics:
    supported_targets = [
        "llvm -mcpu=nehalem",
        "llvm -mcpu=core-avx2",
        "llvm -mcpu=skylake-avx512",
        "llvm -mcpu=cascadelake",
    ]

    unsupported_targets = [
        "llvm -mcpu=x86-64",
    ]

    data_layout, kernel_layout = tvm.testing.parameters(
        ("NCHW", "OIHW"),
        # TODO(@anijain2305, @icemelon9): disable conv2d_int8 for NHWC data layout.
        #   Re-enable this after adding conv2d_NCHWc_int8 support for NHWC.
        # ("NHWC", "HWIO"),
    )

    input_channels, output_channels = tvm.testing.parameters(
        # Sweep the input channels to check int8 robustness
        # Input channels should be a multiple of 4 internally.
        (1, 16),
        (4, 16),
        (6, 16),
        # Sweep the output channels to check int8 robustness
        # Output channels should be a multiple of 16 internally.
        (8, 4),
        (8, 16),
        (8, 20),
        # Check that both non-divisible oc and ic work
        (17, 29),
    )

    @tvm.testing.fixture
    def fast_int8_intrinsic(self, target):
        if "nehalem" in target or "core-avx2" in target or "skylake-avx512" in target:
            return "pmaddubs"
        elif "cascadelake" in target:
            return "vpdpbusd"
        else:
            assert False, "Target should be Skylake or Cascadelake"

    @tvm.testing.fixture
    def assembly(
        self,
        target,
        dtypes,
        input_channels,
        output_channels,
        data_layout,
        kernel_layout,
    ):
        input_dtype, weight_dtype, output_dtype = dtypes

        image_size = (64, 64)
        kernel_size = (3, 3)
        batch_size = 1

        h, w = image_size

        if data_layout == "NCHW":
            data_shape = (batch_size, input_channels, *image_size)
        elif data_layout == "NHWC":
            data_shape = (batch_size, *image_size, input_channels)
        else:
            raise ValueError(f"Unsupported data layout: {data_layout}")
        x = relay.var("x", relay.TensorType(data_shape, input_dtype))

        if kernel_layout == "OIHW":
            kernel_shape = (output_channels, input_channels, *kernel_size)
        elif kernel_layout == "HWIO":
            kernel_shape = (*kernel_size, input_channels, output_channels)
        else:
            raise ValueError("Not supported")
        weight = relay.var("weight", relay.TensorType(kernel_shape, weight_dtype))

        y = relay.nn.conv2d(
            x,
            weight,
            kernel_size=kernel_size,
            channels=output_channels,
            padding=(0, 0, 0, 1),
            dilation=(1, 1),
            data_layout=data_layout,
            kernel_layout=kernel_layout,
            out_dtype=output_dtype,
        )

        func = relay.Function([x, weight], y)

        wdata = np.random.rand(*kernel_shape) * 10
        parameters = {"weight": tvm.nd.array(wdata.astype(weight_dtype))}

        with tvm.transform.PassContext(opt_level=3):
            graph, lib, params = relay.build(func, target, params=parameters)

        return lib.get_source("asm")

    # Ensure that code uses the fast int8 instructions when available.
    @tvm.testing.parametrize_targets(*supported_targets)
    @pytest.mark.parametrize(
        "dtypes",
        [
            # compile conv2d for x86 (skylake, cascadelake) and test
            # assembly contains *pmadd* instructions
            ("uint8", "int8", "int32"),
            # Check that int8 x int8 goes through legalization so that
            # fast instructions can be picked up.
            ("int8", "int8", "int32"),
        ],
    )
    def test_uses_intrinsic(
        self,
        fast_int8_intrinsic,
        assembly,
    ):
        assert fast_int8_intrinsic in assembly

    # For datatypes that don't have HW support, ensure that code is
    # generated without the fast int8 intrinsic.
    @tvm.testing.parametrize_targets(*supported_targets)
    @pytest.mark.parametrize("dtypes", [("uint8", "uint8", "int32")])
    def test_no_intrinsic(
        self,
        fast_int8_intrinsic,
        assembly,
    ):
        assert fast_int8_intrinsic not in assembly

    # Check that a vectorized instruction is generated for older Intel
    # generations, because we default to NCHWc layout.
    @tvm.testing.parametrize_targets(*unsupported_targets)
    @pytest.mark.parametrize("dtypes", [("uint8", "int8", "int32")])
    def test_uses_vectorized_instruction(self, assembly):
        assert "pmulhw" in assembly and "paddd" in assembly


@tvm.testing.uses_gpu
def test_depthwise_conv2d_int8():
    input_dtype = "uint8"
    weight_dtype = "int8"
    output_dtype = "int32"

    data_shape = (1, 64, 56, 56)
    x = relay.var("x", relay.TensorType(data_shape, input_dtype))

    kernel_shape = (64, 1, 3, 3)
    weight = relay.var("weight", relay.TensorType(kernel_shape, weight_dtype))

    y = relay.nn.conv2d(
        x,
        weight,
        kernel_size=(3, 3),
        groups=64,
        padding=(1, 1),
        dilation=(1, 1),
        out_dtype=output_dtype,
    )
    func = relay.Function([x, weight], y)
    wdata = np.random.rand(*kernel_shape) * 10
    parameters = {"weight": tvm.nd.array(wdata.astype(weight_dtype))}

    targets = ["llvm -mcpu=skylake-avx512", "llvm -mcpu=cascadelake"]
    llvm_version = tvm.target.codegen.llvm_version_major()
    for target in targets:
        if llvm_version >= 8:
            with tvm.transform.PassContext(opt_level=3):
                graph, lib, params = relay.build(func, target, params=parameters)


@tvm.testing.uses_gpu
def test_bitserial_conv2d_infer_type():
    # Basic shape test with ambiguous batch.
    n, c, h, w = te.size_var("n"), 32, 224, 224
    x = relay.var("x", relay.ty.TensorType((n, c, h, w), "int16"))
    w = relay.var("w", relay.ty.TensorType((32, 32, 3, 3), "int16"))
    y = relay.nn.bitserial_conv2d(x, w, kernel_size=(3, 3), padding=(0, 0), channels=32)
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, 32, 222, 222), "int16")


@tvm.testing.uses_gpu
def test_bitpack_infer_type():
    # Test axis packing shape inference.
    o, i, h, w = 32, 32, 128, 128
    x = relay.var("x", relay.ty.TensorType((o, i, h, w), "int16"))
    y = relay.nn.bitpack(x, bit_axis=4, pack_axis=1, pack_type="uint16", bits=1)
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((32, 2, 128, 128, 1), "uint16")


# TODO(@jwfromm): Need to add bitserial_conv2d & bitpack run test cases


@tvm.testing.uses_gpu
def test_correlation():
    def _test_correlation(
        data_shape,
        kernel_size,
        max_displacement,
        stride1,
        stride2,
        padding,
        is_multiply,
        dtype="float32",
    ):
        data1 = relay.var("data1", relay.ty.TensorType(data_shape, dtype))
        data2 = relay.var("data2", relay.ty.TensorType(data_shape, dtype))
        y = relay.nn.correlation(
            data1,
            data2,
            kernel_size,
            max_displacement,
            stride1,
            stride2,
            padding,
            is_multiply,
            "NCHW",
        )
        yy = run_infer_type(y)
        padded_height = data_shape[2] + 2 * padding
        padded_width = data_shape[3] + 2 * padding
        border_size = (kernel_size - 1) // 2 + max_displacement
        displacement_radius = max_displacement // stride2
        out_channel = ((2 * displacement_radius) + 1) ** 2
        out_height = (padded_height - 2 * border_size + stride1 - 1) // stride1
        out_width = (padded_width - 2 * border_size + stride1 - 1) // stride1
        assert yy.checked_type == relay.TensorType(
            (data_shape[0], out_channel, out_height, out_width), dtype
        )
        func = relay.Function([data1, data2], y)
        data1_np = np.random.uniform(size=data_shape).astype(dtype)
        data2_np = np.random.uniform(size=data_shape).astype(dtype)
        ref_res = tvm.topi.testing.correlation_nchw_python(
            data1_np,
            data2_np,
            kernel_size,
            max_displacement,
            stride1,
            stride2,
            padding,
            is_multiply,
        )

        for target, dev in tvm.testing.enabled_targets():
            op_res1 = relay.create_executor("graph", device=dev, target=target).evaluate(func)(
                data1_np, data2_np
            )
            tvm.testing.assert_allclose(op_res1.numpy(), ref_res, rtol=1e-5, atol=1e-5)

    _test_correlation(
        (1, 3, 10, 10),
        kernel_size=1,
        max_displacement=4,
        stride1=1,
        stride2=1,
        padding=4,
        is_multiply=True,
    )
    _test_correlation(
        (1, 3, 10, 10),
        kernel_size=1,
        max_displacement=5,
        stride1=1,
        stride2=1,
        padding=5,
        is_multiply=True,
    )
    _test_correlation(
        (5, 1, 4, 4),
        kernel_size=3,
        max_displacement=1,
        stride1=2,
        stride2=1,
        padding=2,
        is_multiply=True,
    )
    _test_correlation(
        (5, 1, 6, 4),
        kernel_size=3,
        max_displacement=1,
        stride1=2,
        stride2=2,
        padding=2,
        is_multiply=False,
    )
    _test_correlation(
        (5, 1, 11, 11),
        kernel_size=5,
        max_displacement=1,
        stride1=1,
        stride2=1,
        padding=2,
        is_multiply=False,
    )


@pytest.mark.skip("Requires GFX10 AMDGPU")
def test_conv2d_rocm_sdot4():
    d_shape = (1, 64, 56, 56)
    w_shape = (64, 64, 3, 3)
    padding = (1, 1)
    strides = (1, 1)
    data_dtype = "int8"
    weight_dtype = "int8"
    out_dtype = "int32"

    data = relay.var("data", shape=d_shape, dtype=data_dtype)
    weight = relay.var("weight", shape=w_shape, dtype=weight_dtype)
    out_channel = w_shape[0]
    conv2d = relay.nn.conv2d(
        data=data,
        weight=weight,
        kernel_size=w_shape[2:],
        channels=out_channel,
        padding=padding,
        strides=strides,
        out_dtype=out_dtype,
    )

    mod = tvm.IRModule.from_expr(conv2d)

    data_np = np.random.uniform(1, 10, d_shape).astype("int8")
    weight_np = np.random.uniform(1, 10, size=w_shape).astype("int8")

    target = "rocm -mattr=+dotprod"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params={"weight": weight_np})

    asm = lib.lib.imported_modules[0].get_source("asm")
    assert "v_dot4_i32_i8" in asm

    dev = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    runtime.set_input("data", data_np)
    runtime.run()

    out = runtime.get_output(0).numpy()

    ref = tvm.topi.testing.conv2d_nchw_python(
        data_np.astype("int32"), weight_np.astype("int32"), strides, padding
    )

    np.testing.assert_equal(out, ref)


def np_float2tvm_bf16(arr):
    """Convert a numpy array of float to a TVM array
    of bf16"""
    orig = arr.view("<u4")
    bias = np.bitwise_and(np.right_shift(orig, 16), 1) + 0x7FFF
    nparr = np.right_shift(orig + bias, 16).astype("uint16")
    return tvm.nd.empty(nparr.shape, "bfloat16").copyfrom(nparr)


def np_bf162np_float(arr):
    """Convert a numpy array of bf16 (uint16) to a numpy array
    of float"""
    u32 = np.left_shift(arr.astype("uint32"), 16)
    return u32.view("<f4")


@tvm.testing.requires_x86
def test_conv2d_nchw_dnnl():
    if not tvm.get_global_func("tvm.contrib.dnnl.conv2d", allow_missing=True):
        print(
            "skip because extern dnnl function is not available, \
                built with dnnl=ON"
        )
        return
    d_shape = (1, 64, 56, 56)
    w_shape = (64, 64, 3, 3)
    padding = (1, 1)
    strides = (1, 1)

    def get_subgraph(dtype):
        data = relay.var("data", shape=d_shape, dtype=dtype)
        weight = relay.var("weight", shape=w_shape, dtype=dtype)
        out_channel = w_shape[0]
        conv2d = relay.nn.conv2d(
            data=data,
            weight=weight,
            kernel_size=w_shape[2:],
            channels=out_channel,
            padding=padding,
            strides=strides,
            out_dtype=dtype,
        )
        return conv2d

    for t in ["float32", "bfloat16"]:
        mod = tvm.IRModule.from_expr(get_subgraph(t))

        data_np = np.random.uniform(1, 10, d_shape).astype("float32")
        weight_np = np.random.uniform(1, 10, size=w_shape).astype("float32")
        ref = tvm.topi.testing.conv2d_nchw_python(data_np, weight_np, strides, padding)

        if t == "bfloat16":
            data_np = np_float2tvm_bf16(data_np)
            weight_np = np_float2tvm_bf16(weight_np)

        target = "llvm -mcpu=skylake-avx512 -libs=dnnl"
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params={"weight": weight_np})

        dev = tvm.device(target, 0)
        runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

        runtime.set_input("data", data_np)
        runtime.run()

        out = runtime.get_output(0).numpy()

        if t == "bfloat16":
            out = np_bf162np_float(out)
            np.testing.assert_allclose(out, ref, rtol=1e-2)
        else:
            np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


@tvm.testing.requires_x86
def test_conv2d_nhwc_dnnl():
    if not tvm.get_global_func("tvm.contrib.dnnl.conv2d", allow_missing=True):
        print(
            "skip because extern dnnl function is not available, \
                built with dnnl=ON"
        )
        return
    d_shape = (1, 56, 56, 64)
    w_shape = (3, 3, 64, 64)
    padding = (1, 1)
    strides = (1, 1)

    def get_subgraph(dtype):
        data = relay.var("data", shape=d_shape, dtype=dtype)
        weight = relay.var("weight", shape=w_shape, dtype=dtype)
        out_channel = w_shape[3]
        conv2d = relay.nn.conv2d(
            data=data,
            weight=weight,
            kernel_size=w_shape[:2],
            channels=out_channel,
            padding=padding,
            strides=strides,
            out_dtype=dtype,
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        return conv2d

    for t in ["float32", "bfloat16"]:
        mod = tvm.IRModule.from_expr(get_subgraph(t))

        data_np = np.random.uniform(1, 10, d_shape).astype("float32")
        weight_np = np.random.uniform(1, 10, size=w_shape).astype("float32")
        ref = tvm.topi.testing.conv2d_nhwc_python(data_np, weight_np, strides, padding)

        if t == "bfloat16":
            data_np = np_float2tvm_bf16(data_np)
            weight_np = np_float2tvm_bf16(weight_np)

        target = "llvm -mcpu=skylake-avx512 -libs=dnnl"
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params={"weight": weight_np})

        dev = tvm.device(target, 0)
        runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

        runtime.set_input("data", data_np)
        runtime.run()

        out = runtime.get_output(0).numpy()

        if t == "bfloat16":
            out = np_bf162np_float(out)
            np.testing.assert_allclose(out, ref, rtol=1e-2)
        else:
            np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def _test_conv2d_int8_alter_dtype(data_dtype, target, dot_product_instr):
    def get_conv2d_nchw(
        d_shape,
        w_shape,
        data_dtype,
    ):
        out_dtype = "int32"
        strides = (1, 1)
        padding = (1, 1)
        data = relay.var("data", shape=d_shape, dtype=data_dtype)
        weight = relay.var("weight", shape=w_shape, dtype="int8")
        out_channel = w_shape[0]
        return relay.nn.conv2d(
            data=data,
            weight=weight,
            kernel_size=w_shape[2:],
            channels=out_channel,
            padding=padding,
            strides=strides,
            out_dtype=out_dtype,
        )

    I, O, H, W = 64, 64, 56, 56
    kH = kW = 3

    data_shape = (1, I, H, W)
    weight_shape = (O, I, kH, kW)
    bias_shape = (1, weight_shape[0], 1, 1)

    bias = relay.var("bias", shape=bias_shape, dtype="int32")
    bias_np = np.random.randint(low=-127, high=128, size=bias_shape).astype("int32")
    weight_np = np.random.uniform(-128, 127, size=weight_shape).astype("int8")

    conv2d = get_conv2d_nchw(data_shape, weight_shape, data_dtype)
    bias_add = relay.add(conv2d, bias)
    mod = tvm.IRModule.from_expr(bias_add)

    if data_dtype == "uint8":
        data_np = np.random.uniform(0, 255, size=data_shape).astype("uint8")
    else:
        data_np = np.random.uniform(-128, 127, size=data_shape).astype("int8")

    params = {"weight": weight_np, "bias": bias_np}

    ref = (
        relay.create_executor("graph", mod=mod, device=tvm.cpu(0), target="llvm")
        .evaluate()(*[data_np, weight_np, bias_np])
        .numpy()
    )

    dev = tvm.cpu(0)

    with tvm.transform.PassContext(
        opt_level=3,
    ):
        lib = relay.build(mod, target=target, params=params)

    assert dot_product_instr in lib.lib.get_source("asm")

    rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    rt_mod.set_input("data", data_np)

    rt_mod.run()

    out = rt_mod.get_output(0).numpy()

    np.testing.assert_equal(out, ref)


@tvm.testing.requires_arm_dot
def test_conv2d_int8_alter_dtype_arm():
    _test_conv2d_int8_alter_dtype(
        "uint8", "llvm -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+dotprod", "sdot"
    )


@tvm.testing.requires_cascadelake
def test_conv2d_int8_alter_dtype_vnni():
    _test_conv2d_int8_alter_dtype("int8", "llvm -mcpu=cascadelake", "vpdpbusd")


if __name__ == "__main__":
    tvm.testing.main()
