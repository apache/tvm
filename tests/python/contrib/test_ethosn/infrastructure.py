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

"""Ethos-N test functions"""

from __future__ import absolute_import, print_function
import tvm
from tvm import relay
from tvm.contrib import util, graph_runtime, download
from hashlib import md5
from itertools import zip_longest, combinations
import numpy as np
from PIL import Image
import os

from . import _infrastructure
from tvm.relay.op.contrib import get_pattern_table


def get_real_image(im_height, im_width):
    repo_base = "https://github.com/dmlc/web-data/raw/master/tensorflow/models/InceptionV1/"
    img_name = "elephant-299.jpg"
    image_url = os.path.join(repo_base, img_name)
    img_path = download.download_testdata(image_url, img_name, module="data")
    image = Image.open(img_path).resize((im_height, im_width))
    x = np.array(image).astype("uint8")
    data = np.reshape(x, (1, im_height, im_width, 3))
    return data


def assert_lib_hash(lib, golden):
    temp = util.tempdir()
    path = temp.relpath("lib.cmm")
    lib.imported_modules[1].save(path)
    lib_hash = md5(open(path, "rb").read()).hexdigest()
    assert lib_hash == golden, "Expected hash: {} Got hash: {}".format(golden, lib_hash)


def make_module(func, params):
    func = relay.Function(relay.analysis.free_vars(func), func)
    if params:
        relay.build_module.bind_params_by_name(func, params)
    return tvm.IRModule.from_expr(func)


def make_ethosn_composite(ethosn_expr, name):
    vars = relay.analysis.free_vars(ethosn_expr)
    func = relay.Function([relay.Var("a")], ethosn_expr)
    func = func.with_attr("Composite", name)
    call = relay.Call(func, vars)
    return call


def make_ethosn_partition(ethosn_expr):
    # Create an Ethos-N global function
    mod = tvm.IRModule({})
    vars = relay.analysis.free_vars(ethosn_expr)
    func = relay.Function(vars, ethosn_expr)
    func = func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Inline", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Compiler", "ethos-n")
    func = func.with_attr("global_symbol", "ethos-n_0")
    g1 = relay.GlobalVar("ethos-n_0")
    mod[g1] = func

    # These are the vars to call the Ethos-N partition with
    more_vars = relay.analysis.free_vars(ethosn_expr)
    # Call the Ethos-N partition in main
    call_fn1 = g1(*more_vars)
    mod["main"] = relay.Function(more_vars, call_fn1)
    return mod


def get_host_op_count(mod):
    class Counter(tvm.relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.count = 0

        def visit_call(self, call):
            if isinstance(call.op, tvm.ir.Op):
                self.count += 1
            super().visit_call(call)

    c = Counter()
    c.visit(mod["main"])
    return c.count


def build(mod, params, npu=True, expected_host_ops=0, npu_partitions=1):
    relay.backend.compile_engine.get().clear()
    with tvm.transform.PassContext(
        opt_level=3, config={"relay.ext.ethos-n.options": {"variant": 0}}
    ):
        with tvm.target.Target("llvm"):
            if npu:
                f = relay.build_module.bind_params_by_name(mod["main"], params)
                mod = tvm.IRModule()
                mod["main"] = f
                pattern = get_pattern_table("ethos-n")
                mod = relay.transform.MergeComposite(pattern)(mod)
                mod = relay.transform.AnnotateTarget("ethos-n")(mod)
                mod = relay.transform.MergeCompilerRegions()(mod)
                mod = relay.transform.PartitionGraph()(mod)
                host_op_count = get_host_op_count(mod)
                assert (
                    host_op_count == expected_host_ops
                ), "Got {} host operators, expected {}".format(host_op_count, expected_host_ops)
                partition_count = 0
                for global_var in mod.get_global_vars():
                    if "ethos-n" in global_var.name_hint:
                        partition_count += 1

                assert (
                    npu_partitions == partition_count
                ), "Got {} ethos-n partitions, expected {}".format(partition_count, npu_partitions)

            return relay.build(mod, params=params)


def run(lib, inputs, outputs, npu=True):
    # Export and load lib to confirm this works
    lib_name = "mod.so"
    temp = util.tempdir()
    lib_path = temp.relpath(lib_name)
    lib.export_library(lib_path)
    lib = tvm.runtime.load_module(lib_path)
    module = graph_runtime.GraphModule(lib["default"](tvm.cpu()))
    module.set_input(**inputs)
    module.run()
    out = [module.get_output(i) for i in range(outputs)]
    if not npu:
        inference_result(0, out)
    return out


def build_and_run(
    mod, inputs, outputs, params, ctx=tvm.cpu(), npu=True, expected_host_ops=0, npu_partitions=1
):
    lib = build(mod, params, npu, expected_host_ops, npu_partitions)
    return run(lib, inputs, outputs, npu)


def verify(answers, atol, rtol=1e-07, verify_saturation=True):
    """Compare the array of answers. Each entry is a list of outputs"""
    if len(answers) < 2:
        print("No results to compare: expected at least two, found ", len(answers))
    for answer in zip_longest(*answers):
        for outs in combinations(answer, 2):
            if verify_saturation:
                assert (
                    np.count_nonzero(outs[0].asnumpy() == 255) < 0.25 * outs[0].asnumpy().size
                ), "Output is saturated: {}".format(outs[0])
                assert (
                    np.count_nonzero(outs[0].asnumpy() == 0) < 0.25 * outs[0].asnumpy().size
                ), "Output is saturated: {}".format(outs[0])
            tvm.testing.assert_allclose(outs[0].asnumpy(), outs[1].asnumpy(), rtol=rtol, atol=atol)


def inference_result(checksum, outputs):
    """Set the expected results of an Ethos inference, if the testing
    infrastructure is available. This assumes that the entire graph
    was offloaded to the neural processor."""
    if tvm.get_global_func("relay.ethos-n.test.infra.inference_result", True):
        return _infrastructure.inference_result(checksum, *outputs)
    return False


def test_error(mod, params, err_msg):
    caught = None
    with tvm.transform.PassContext(opt_level=3):
        with tvm.target.Target("llvm"):
            try:
                relay.build(mod, params)
            except tvm.error.TVMError as e:
                caught = e.args[0]
            finally:
                relay.backend.compile_engine.get().clear()

    assert caught is not None
    assert err_msg in caught, caught


def get_conv2d(var, shape):
    """Standard convolution to test activation functions"""

    weight_shape = (1, 1, shape[3], 1)
    w = tvm.nd.array(np.ones(weight_shape, "uint8"))
    weights = relay.const(w, "uint8")
    conv = relay.qnn.op.conv2d(
        var,
        weights,
        input_zero_point=relay.const(0, "int32"),
        kernel_zero_point=relay.const(0, "int32"),
        input_scale=relay.const(1.0, "float32"),
        kernel_scale=relay.const(1.0, "float32"),
        kernel_size=(1, 1),
        channels=1,
        data_layout="NHWC",
        kernel_layout="HWIO",
    )
    b = tvm.nd.array(np.zeros((shape[0],), "int32"))
    biasc = relay.const(b, "int32")
    bias = relay.nn.bias_add(conv, biasc, axis=0)
    req = relay.qnn.op.requantize(
        bias,
        relay.const(1.0, "float32"),  # input zero scale
        relay.const(0, "int32"),  # input zero point
        relay.const(1.1, "float32"),  # output zero scale
        relay.const(0, "int32"),  # output zero point
        out_dtype="uint8",
    )
    params = {"w": w, "b": b}
    return req, params


def get_conv2d_qnn_params(input_zp, input_sc, kernel_zp, kernel_sc, kernel_h, kernel_w, channels):
    input_max = input_sc * (255 - input_zp)
    input_min = -input_sc * input_zp
    kernel_max = kernel_sc * (255 - kernel_zp)
    kernel_min = -kernel_sc * kernel_zp
    output_limits = [
        kernel_max * kernel_h * kernel_w * channels * input_max,
        kernel_min * kernel_h * kernel_w * channels * input_max,
        kernel_min * kernel_h * kernel_w * channels * input_min,
        kernel_max * kernel_h * kernel_w * channels * input_min,
    ]
    output_max = max(output_limits)
    output_min = min(output_limits)
    output_sc = (output_max - output_min) / 255
    output_zp = -int(output_min / output_sc)
    return output_zp, output_sc
