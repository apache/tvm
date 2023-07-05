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

"""CMSIS-NN integration tests: fuse_pads pass"""
import numpy as np
import pytest
import tvm
from tvm.testing.aot import get_dtype_range
from tvm import relay
from .utils import CheckForPadsWithinCompositeFunc

tvm._ffi._init_api("relay.ext.cmsisnn.transform", __name__)


def set_external_func_attr(func, compiler, ext_symbol):
    func = func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Compiler", compiler)
    func = func.with_attr("global_symbol", ext_symbol)
    return func


def set_composite_func_attr(func, name):
    func = func.with_attr("Composite", name)
    return func


@pytest.mark.parametrize(
    "ifm_shape, pad_width, conv2d_padding, ofm_shape",
    [
        [(1, 25, 25, 12), ((0, 0), (0, 2), (1, 2), (0, 0)), (1, 1, 1, 1), (1, 26, 28, 2)],
        [(1, 64, 100, 4), ((0, 0), (1, 3), (1, 1), (0, 0)), (0, 0, 0, 0), (1, 64, 100, 2)],
        [(1, 55, 55, 3), ((0, 0), (2, 1), (3, 5), (0, 0)), (0, 0, 1, 1), (1, 57, 59, 2)],
    ],
)
def test_invalid_padding_for_fusion(ifm_shape, pad_width, conv2d_padding, ofm_shape):
    """Negative tests for pads preceding Conv2D that cannot be fused."""
    dtype = "int8"
    kernel_size = (3, 3)
    ofm_channels = 2
    local_input = relay.var("local_input", shape=ifm_shape, dtype=dtype)
    pad = relay.nn.pad(
        local_input,
        pad_width=pad_width,  # ((), (top, bottom), (left, right), ())
        pad_value=10,
        pad_mode="constant",
    )
    rng = np.random.default_rng(12321)
    in_min, in_max = get_dtype_range(dtype)
    local_weight = tvm.nd.array(
        rng.integers(
            in_min,
            high=in_max,
            size=(ofm_channels, kernel_size[0], kernel_size[1], ifm_shape[3]),
            dtype=dtype,
        )
    )
    local_weight = relay.const(local_weight, dtype)
    conv2d = relay.qnn.op.conv2d(
        pad,
        local_weight,
        relay.const(1, "int32"),
        relay.const(1, "int32"),
        relay.const(1, "float32"),
        relay.const(1, "float32"),
        data_layout="NHWC",
        kernel_layout="OHWI",
        channels=ofm_channels,
        kernel_size=(3, 3),
        padding=conv2d_padding,
        out_dtype="int32",
    )
    requantize = relay.qnn.op.requantize(
        conv2d,
        relay.const(1, "float32"),
        relay.const(1, "int32"),
        relay.const(1, "float32"),
        relay.const(1, "int32"),
        axis=0,
        out_dtype=dtype,
    )
    local_func = relay.Function(relay.analysis.free_vars(requantize), requantize)
    local_func = set_composite_func_attr(local_func, "cmsis-nn.qnn_conv2d")

    mod = tvm.IRModule()
    ext_input = relay.var("ext_input", shape=ifm_shape, dtype=dtype)
    call_local_func = relay.Call(local_func, [ext_input])
    extern_func = relay.Function(relay.analysis.free_vars(call_local_func), call_local_func)
    extern_var = relay.GlobalVar("external_function")
    extern_func = set_external_func_attr(extern_func, "cmsis-nn", extern_var.name_hint)
    mod[extern_var] = extern_func

    main_input = relay.var("main_input", shape=ifm_shape, dtype=dtype)
    call_extern_func = relay.Call(extern_var, [main_input])
    main_func = relay.Function([main_input], call_extern_func, relay.TensorType(ofm_shape, dtype))
    main_var = relay.GlobalVar("main")
    mod[main_var] = main_func

    mod = relay.transform.InferType()(mod)

    error_regex = r"Difference on each side of a dimension should be either 0 or 1"

    with pytest.raises(tvm.TVMError, match=error_regex):
        mod = CMSISNNFusePads()(mod)


@pytest.mark.parametrize(
    "ifm_shape, pad_width, conv2d_padding, ofm_shape",
    [
        [(1, 25, 25, 12), ((0, 0), (0, 1), (1, 2), (0, 0)), (1, 1, 1, 1), (1, 26, 28, 2)],
        [(1, 64, 100, 4), ((0, 0), (1, 1), (1, 1), (0, 0)), (0, 0, 0, 0), (1, 64, 100, 2)],
        [(1, 55, 55, 3), ((0, 0), (2, 1), (3, 2), (0, 0)), (0, 0, 1, 1), (1, 57, 59, 2)],
    ],
)
def test_pad_conv2d_fusion_noncmsisnn_target(ifm_shape, pad_width, conv2d_padding, ofm_shape):
    """Tests the pads and conv2d fusion for non-cmsisnn targets.
    It is expected that pad will not be fused with Conv2D in this case.
    """
    dtype = "int8"
    kernel_size = (3, 3)
    ofm_channels = 2
    local_input = relay.var("local_input", shape=ifm_shape, dtype=dtype)
    pad = relay.nn.pad(
        local_input,
        pad_width=pad_width,  # ((), (top, bottom), (left, right), ())
        pad_value=10,
        pad_mode="constant",
    )
    rng = np.random.default_rng(12321)
    in_min, in_max = get_dtype_range(dtype)
    local_weight = tvm.nd.array(
        rng.integers(
            in_min,
            high=in_max,
            size=(ofm_channels, kernel_size[0], kernel_size[1], ifm_shape[3]),
            dtype=dtype,
        )
    )
    local_weight = relay.const(local_weight, dtype)
    conv2d = relay.qnn.op.conv2d(
        pad,
        local_weight,
        relay.const(1, "int32"),
        relay.const(1, "int32"),
        relay.const(1, "float32"),
        relay.const(1, "float32"),
        data_layout="NHWC",
        kernel_layout="OHWI",
        channels=ofm_channels,
        kernel_size=(3, 3),
        padding=conv2d_padding,
        out_dtype="int32",
    )
    requantize = relay.qnn.op.requantize(
        conv2d,
        relay.const(1, "float32"),
        relay.const(1, "int32"),
        relay.const(1, "float32"),
        relay.const(1, "int32"),
        axis=0,
        out_dtype=dtype,
    )
    local_func = relay.Function(relay.analysis.free_vars(requantize), requantize)
    local_func = set_composite_func_attr(local_func, "noncmsis-nn.qnn_conv2d")

    mod = tvm.IRModule()
    ext_input = relay.var("ext_input", shape=ifm_shape, dtype=dtype)
    call_local_func = relay.Call(local_func, [ext_input])
    extern_func = relay.Function(relay.analysis.free_vars(call_local_func), call_local_func)
    extern_var = relay.GlobalVar("external_function")
    extern_func = set_external_func_attr(extern_func, "noncmsis-nn", extern_var.name_hint)
    mod[extern_var] = extern_func

    main_input = relay.var("main_input", shape=ifm_shape, dtype=dtype)
    call_extern_func = relay.Call(extern_var, [main_input])
    main_func = relay.Function([main_input], call_extern_func, relay.TensorType(ofm_shape, dtype))
    main_var = relay.GlobalVar("main")
    mod[main_var] = main_func

    mod = relay.transform.InferType()(mod)

    mod = CMSISNNFusePads()(mod)
    pad_verifier = CheckForPadsWithinCompositeFunc()
    pad_verifier.visit_function(mod[extern_var])
    pad_verifier.assert_pads_within_func()


@pytest.mark.parametrize(
    "ifm_shape, pad_width, conv2d_padding, ofm_shape",
    [
        [(1, 25, 25, 12), ((0, 0), (0, 1), (1, 2), (0, 0)), (1, 1, 1, 1), (1, 26, 28, 2)],
        [(1, 64, 100, 4), ((0, 0), (1, 1), (1, 1), (0, 0)), (0, 0, 0, 0), (1, 64, 100, 2)],
        [(1, 55, 55, 3), ((0, 0), (2, 1), (3, 2), (0, 0)), (0, 0, 1, 1), (1, 57, 59, 2)],
    ],
)
def test_pad_conv2d_fusion(ifm_shape, pad_width, conv2d_padding, ofm_shape):
    """Tests the pads and conv2d fusion."""
    dtype = "int8"
    kernel_size = (3, 3)
    ofm_channels = 2
    local_input = relay.var("local_input", shape=ifm_shape, dtype=dtype)
    pad = relay.nn.pad(
        local_input,
        pad_width=pad_width,  # ((), (top, bottom), (left, right), ())
        pad_value=10,
        pad_mode="constant",
    )
    rng = np.random.default_rng(12321)
    kmin, kmax = get_dtype_range(dtype)
    local_weight = tvm.nd.array(
        rng.integers(
            kmin,
            high=kmax,
            size=(ofm_channels, kernel_size[0], kernel_size[1], ifm_shape[3]),
            dtype=dtype,
        )
    )
    local_weight = relay.const(local_weight, dtype)
    conv2d = relay.qnn.op.conv2d(
        pad,
        local_weight,
        relay.const(1, "int32"),
        relay.const(1, "int32"),
        relay.const(1, "float32"),
        relay.const(1, "float32"),
        data_layout="NHWC",
        kernel_layout="OHWI",
        channels=ofm_channels,
        kernel_size=(3, 3),
        padding=conv2d_padding,
        out_dtype="int32",
    )
    requantize = relay.qnn.op.requantize(
        conv2d,
        relay.const(1, "float32"),
        relay.const(1, "int32"),
        relay.const(1, "float32"),
        relay.const(1, "int32"),
        axis=0,
        out_dtype=dtype,
    )
    local_func = relay.Function(relay.analysis.free_vars(requantize), requantize)
    local_func = set_composite_func_attr(local_func, "cmsis-nn.qnn_conv2d")

    mod = tvm.IRModule()
    ext_input = relay.var("ext_input", shape=ifm_shape, dtype=dtype)
    call_local_func = relay.Call(local_func, [ext_input])
    extern_func = relay.Function(relay.analysis.free_vars(call_local_func), call_local_func)
    extern_var = relay.GlobalVar("external_function")
    extern_func = set_external_func_attr(extern_func, "cmsis-nn", extern_var.name_hint)
    mod[extern_var] = extern_func

    main_input = relay.var("main_input", shape=ifm_shape, dtype=dtype)
    call_extern_func = relay.Call(extern_var, [main_input])
    main_func = relay.Function([main_input], call_extern_func, relay.TensorType(ofm_shape, dtype))
    main_var = relay.GlobalVar("main")
    mod[main_var] = main_func

    mod = relay.transform.InferType()(mod)

    mod = CMSISNNFusePads()(mod)
    pad_verifier = CheckForPadsWithinCompositeFunc()
    pad_verifier.visit_function(mod[extern_var])
    pad_verifier.assert_no_pads_within_func()


def test_without_preceding_pad():
    """Tests the pass FusePads when padding is not present before qnn.conv2d."""
    dtype = "int8"
    ifm_shape = (1, 56, 56, 64)
    ofm_shape = (1, 56, 56, 64)
    local_input = relay.var("local_input", shape=ifm_shape, dtype=dtype)
    rng = np.random.default_rng(12321)
    kmin, kmax = get_dtype_range(dtype)
    local_weight = tvm.nd.array(
        rng.integers(
            kmin,
            high=kmax,
            size=(64, 3, 3, 64),
            dtype=dtype,
        )
    )
    local_weight = relay.const(local_weight, dtype)
    conv2d = relay.qnn.op.conv2d(
        local_input,
        local_weight,
        relay.const(1, "int32"),
        relay.const(1, "int32"),
        relay.const(1, "float32"),
        relay.const(1, "float32"),
        data_layout="NHWC",
        kernel_layout="OHWI",
        channels=64,
        kernel_size=(3, 3),
        padding=(1, 1, 1, 1),
        out_dtype="int32",
    )
    requantize = relay.qnn.op.requantize(
        conv2d,
        relay.const(1, "float32"),
        relay.const(1, "int32"),
        relay.const(1, "float32"),
        relay.const(1, "int32"),
        axis=0,
        out_dtype=dtype,
    )
    relu = relay.nn.relu(requantize)
    local_func = relay.Function(relay.analysis.free_vars(relu), relu)
    local_func = set_composite_func_attr(local_func, "cmsis-nn.qnn_conv2d")

    mod = tvm.IRModule()
    ext_input = relay.var("ext_input", shape=ifm_shape, dtype=dtype)
    call_local_func = relay.Call(local_func, [ext_input])
    extern_func = relay.Function(relay.analysis.free_vars(call_local_func), call_local_func)
    extern_var = relay.GlobalVar("external_function")
    extern_func = set_external_func_attr(extern_func, "cmsis-nn", extern_var.name_hint)
    mod[extern_var] = extern_func

    main_input = relay.var("main_input", shape=ifm_shape, dtype=dtype)
    call_extern_func = relay.Call(extern_var, [main_input])
    main_func = relay.Function(relay.analysis.free_vars(call_extern_func), call_extern_func)
    main_func = relay.Function([main_input], call_extern_func, relay.TensorType(ofm_shape, dtype))
    main_var = relay.GlobalVar("main")
    mod[main_var] = main_func

    mod = relay.transform.InferType()(mod)

    mod = CMSISNNFusePads()(mod)
    pad_verifier = CheckForPadsWithinCompositeFunc()
    pad_verifier.visit_function(mod[extern_var])
    pad_verifier.assert_no_pads_within_func()
