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

"""Arm(R) Ethos(TM)-N test functions"""

from __future__ import absolute_import, print_function
from hashlib import md5
from itertools import zip_longest, combinations
import os
from typing import Tuple
import math

import numpy as np
from PIL import Image

import tvm
from tvm import relay
from tvm.contrib import utils, graph_executor, download
from tvm.relay.op.contrib import partition_for_ethosn
from tvm.driver.tvmc.target import parse_target

from . import _infrastructure


def get_real_image(im_height, im_width):
    repo_base = "https://github.com/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/"
    img_name = "elephant-299.jpg"
    image_url = os.path.join(repo_base, img_name)
    img_path = download.download_testdata(image_url, img_name, module="data")
    image = Image.open(img_path).resize((im_height, im_width))
    x = np.array(image).astype("uint8")
    data = np.reshape(x, (1, im_height, im_width, 3))
    return data


def assert_lib_hash(lib, golden):
    """Check that the Ethos-N runtime modules in a library hash to the same values
    as given by the golden hash(es).

    If there's only one Ethos-N module, the golden hash may be provided as a str.
    If there's multiple, a set of golden hashes should be provided to correspond
    with each Ethos-N module that is expected.

    This function is used to ensure that no change is made which alters the output
    of a compilation. If such a change is made deliberately (eg. to fix a bug) then
    the golden hash should be updated after verifying on hardware that the behaviour
    is still correct.

    This method is used because of the lack of hardware availability in upstream CI.
    """
    # Convert str hash into a set of hashes
    if isinstance(golden, str):
        golden = {golden}

    temp = utils.tempdir()
    path = temp.relpath("lib.cmm")
    hash_set = set()
    for mod in lib.imported_modules:
        if mod.type_key == "ethos-n":
            mod.save(path)
            with open(path, "rb") as compiled_model:
                lib_hash = md5(compiled_model.read()).hexdigest()
            hash_set.add(lib_hash)

    assert hash_set == golden, "Expected hash: {} Got hash: {}".format(golden, hash_set)


def make_module(func, params):
    func = relay.Function(relay.analysis.free_vars(func), func)
    if params:
        relay.build_module.bind_params_by_name(func, params)
    mod = tvm.IRModule.from_expr(func)
    return relay.transform.InferType()(mod)


def make_ethosn_composite(ethosn_expr, name):
    variables = relay.analysis.free_vars(ethosn_expr)
    inner_vars = [relay.Var(v.name_hint, v.type_annotation) for v in variables]
    func = relay.Function(inner_vars, ethosn_expr)
    func = func.with_attr("Composite", name)
    call = relay.Call(func, variables)
    return call


def make_ethosn_partition(ethosn_expr):
    """Make an Ethos(TM)-N partition."""

    # Create an Ethos-N global function
    mod = tvm.IRModule({})
    variables = relay.analysis.free_vars(ethosn_expr)
    # NB: it is illegal to reuse variables inside and outside a scope in Relay
    # if you want to duplicate types and names you must re-allocate them.
    fresh_vars = [relay.Var(v.name_hint, v.type_annotation) for v in variables]
    binds = {}
    for var, fresh_var in zip(variables, fresh_vars):
        binds[var] = fresh_var
    ethosn_expr_fresh = relay.bind(ethosn_expr, binds)
    func = relay.Function(fresh_vars, ethosn_expr_fresh)
    func = func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Inline", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Compiler", "ethos-n")
    func = func.with_attr("global_symbol", "ethos-n_0")
    global_var = relay.GlobalVar("ethos-n_0")
    mod[global_var] = func
    mod = relay.transform.InferType()(mod)

    # These are the vars to call the Ethos-N partition with
    more_vars = relay.analysis.free_vars(ethosn_expr)
    # Call the Ethos-N partition in main
    call_fn1 = global_var(*more_vars)
    mod["main"] = relay.Function(more_vars, call_fn1)
    return relay.transform.InferType()(mod)


def get_host_op_count(mod):
    """Return the number of host operators."""

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


def build(
    mod, params, npu=True, expected_host_ops=0, npu_partitions=1, additional_config_args=None
):
    """Build a network with or without Ethos-N offloading.

    Parameters
    ----------
    mod : IRModule
        The Relay module to build.
    params : dict of str to NDArray
        The weights to build with.
    npu : bool, optional
        Whether to build with Ethos-N offloading.
    expected_host_ops : int, optional
        The number of ops expected to remain on the host.
    npu_partitions : int, optional
        The number of Ethos-N partitions expected.
    additional_config_args : dict, optional
        Additional compiler config options for the NPU.
    """
    relay.backend.te_compiler.get().clear()
    if not additional_config_args:
        additional_config_args = {}
    npu_config = {**get_ethosn_device_options(), **additional_config_args}
    print(npu_config)
    with tvm.transform.PassContext(opt_level=3, config={"relay.ext.ethos-n.options": npu_config}):
        with tvm.target.Target("llvm"):
            if npu:
                mod = partition_for_ethosn(mod, params)
                host_op_count = get_host_op_count(mod)
                assert (
                    host_op_count == expected_host_ops
                ), "Got {} host operators, expected {}".format(host_op_count, expected_host_ops)

                attrs = [
                    mod[var.name_hint].attrs
                    for var in mod.get_global_vars()
                    if mod[var.name_hint].attrs
                ]
                partition_count = sum(
                    [
                        key == "Compiler" and value == "ethos-n"
                        for attr in attrs
                        for key, value in attr.items()
                    ]
                )
                assert (
                    npu_partitions == partition_count
                ), "Got {} ethos-n partitions, expected {}".format(partition_count, npu_partitions)

            return relay.build(mod, params=params)


def run(lib, inputs, outputs, npu=True):
    """Run a module with specified inputs.

    Parameters
    ----------
    lib : runtime.Module
        The runtime module.
    inputs : dict of str to NDArray
        The input dictionary.
    outputs : int
        The expected number of outputs.
    npu : bool
        Whether or not any part of the lib is offloaded to Ethos-N.
        If it's false (i.e. it's all running on the CPU), we set
        the mocked result equal to the output so that a subsequent
        mocked run on the NPU returns the same value.

    Returns
    -------
    out : list of NDArray
        The results.

    """
    # Export and load lib to confirm this works
    lib_name = "mod.so"
    temp = utils.tempdir()
    lib_path = temp.relpath(lib_name)
    lib.export_library(lib_path)
    lib = tvm.runtime.load_module(lib_path)
    module = graph_executor.GraphModule(lib["default"](tvm.cpu()))
    module.set_input(**inputs)
    module.run()
    out = [module.get_output(i) for i in range(outputs)]
    if not npu:
        inference_result(out)
    return out


def build_and_run(
    mod,
    inputs,
    outputs,
    params,
    npu=True,
    expected_host_ops=0,
    npu_partitions=1,
    additional_config_args=None,
):
    """
    Convenient wrapper for building and running a module on the NPU.
    """
    lib = build(mod, params, npu, expected_host_ops, npu_partitions, additional_config_args)
    return run(lib, inputs, outputs, npu)


def verify(answers, dtype, atol, rtol=1e-07, verify_saturation=True):
    """Compare the array of answers. Each entry is a list of outputs"""
    if len(answers) < 2:
        print("No results to compare: expected at least two, found ", len(answers))
    for answer in zip_longest(*answers):
        for outs in combinations(answer, 2):
            if verify_saturation:
                assert (
                    np.count_nonzero(outs[0].numpy() == np.iinfo(dtype).max)
                    < 0.25 * outs[0].numpy().size
                ), "Output is saturated: {}".format(outs[0])
                assert (
                    np.count_nonzero(outs[0].numpy() == np.iinfo(dtype).min)
                    < 0.25 * outs[0].numpy().size
                ), "Output is saturated: {}".format(outs[0])
            tvm.testing.assert_allclose(outs[0].numpy(), outs[1].numpy(), rtol=rtol, atol=atol)


def inference_result(outputs):
    """Set the expected results of an Ethos inference, if the testing
    infrastructure is available. This assumes that the entire graph
    was offloaded to the neural processor."""
    if tvm.get_global_func("relay.ethos-n.test.infra.inference_result", True):
        return _infrastructure.inference_result(*outputs)
    return False


def test_error(mod, params, err_msg):
    """Test an operator error message."""

    caught = None
    with tvm.transform.PassContext(
        opt_level=3, config={"relay.ext.ethos-n.options": get_ethosn_device_options()}
    ):
        with tvm.target.Target("llvm"):
            try:
                mod = relay.transform.InferType()(mod)
                relay.build(mod, params=params)
            except tvm.error.TVMError as error:
                caught = error.args[0]
            finally:
                relay.backend.te_compiler.get().clear()

    assert caught is not None
    assert err_msg in caught, caught


def get_conv2d(var, shape, dtype):
    """Standard convolution to test activation functions"""

    weight_shape = (1, 1, shape[3], 1)
    weights_array = tvm.nd.array(np.ones(weight_shape, dtype))
    weights = relay.const(weights_array, dtype)
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
        out_dtype=dtype,
    )
    params = {"w": weights_array, "b": b}
    return req, params


def get_conv2d_qnn_params(
    dtype, input_zp, input_sc, kernel_zp, kernel_sc, kernel_h, kernel_w, channels
):
    """Return Conv2D QNN params."""

    kernel_sc = (
        kernel_sc.numpy() if isinstance(kernel_sc, tvm.runtime.ndarray.NDArray) else [kernel_sc]
    )
    dtype_min = np.iinfo(dtype).min
    dtype_max = np.iinfo(dtype).max

    input_max = input_sc * (dtype_max - input_zp)
    input_min = input_sc * (dtype_min - input_zp)

    kernel_max = max(kernel_sc) * (dtype_max - kernel_zp)
    kernel_min = min(kernel_sc) * (dtype_min - kernel_zp)

    output_limits = [
        kernel_max * kernel_h * kernel_w * channels * input_max,
        kernel_min * kernel_h * kernel_w * channels * input_max,
        kernel_min * kernel_h * kernel_w * channels * input_min,
        kernel_max * kernel_h * kernel_w * channels * input_min,
    ]
    output_max = max(output_limits)
    output_min = min(output_limits)

    output_sc = (output_max - output_min) / (dtype_max - dtype_min)
    output_zp = int(dtype_min - (output_min / output_sc))
    return output_zp, output_sc


def get_same_padding(
    data: Tuple[int, int],
    kernel: Tuple[int, int],
    dilation: Tuple[int, int],
    stride: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """
    Get the padding values required for 'SAME' padding.

    Parameters
    ----------
    data : Tuple[int, int]
        The height and width of the data respectively.
    kernel : Tuple[int, int]
        The height and width of the kernel respectively.
    dilation : Tuple[int, int]
        The dilation of the kernel.
    stride : Tuple[int, int]
        The stride of the kernel.

    Returns
    -------
    Tuple[int, int, int, int]
        The padding values for top, left, bottom and right respectively.
    """
    dilated_kernel_h = dilation[0] * (kernel[0] - 1) + 1
    dilated_kernel_w = dilation[1] * (kernel[1] - 1) + 1
    out = int(math.ceil(float(data[0]) / float(stride[0])))
    pad = max(0, (out - 1) * stride[0] + dilated_kernel_h - data[0])
    pad_top = pad // 2
    pad_bottom = pad - pad_top

    out = int(math.ceil(float(data[1]) / float(stride[1])))
    pad = max(0, (out - 1) * stride[1] + dilated_kernel_w - data[1])
    pad_left = pad // 2
    pad_right = pad - pad_left
    return (pad_top, pad_left, pad_bottom, pad_right)


def get_ethosn_device_options():
    """Determine the NPU configuration used for testing."""
    default_target_string = "ethos-n -variant=n78 -tops=1 -ple_ratio=2"
    target_string = os.getenv("ETHOSN_TEST_TARGET_CONFIG", default_target_string)
    target = parse_target(target_string)
    return target[0]["opts"]
