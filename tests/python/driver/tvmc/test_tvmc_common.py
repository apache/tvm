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
import argparse
import os
from os import path

import pytest

import tvm
from tvm import relay
from tvm.driver import tvmc

from tvm.driver.tvmc.common import TVMCException


def test_compile_tflite_module_nhwc_to_nchw(tflite_mobilenet_v1_1_quant):
    # some CI environments wont offer TFLite, so skip in case it is not present
    pytest.importorskip("tflite")

    before, _ = tvmc.frontends.load_model(tflite_mobilenet_v1_1_quant)

    expected_layout = "NCHW"
    after = tvmc.common.convert_graph_layout(before, expected_layout)

    layout_transform_calls = []

    def _is_layout_transform(node):
        if isinstance(node, tvm.relay.expr.Call):
            layout_transform_calls.append(
                node.op.name == "layout_transform"
                and node.attrs.src_layout == "NHWC"
                and node.attrs.dst_layout == "NCHW"
            )

    tvm.relay.analysis.post_order_visit(after["main"], _is_layout_transform)

    assert any(layout_transform_calls), "Expected 'layout_transform NHWC->NCHW' not found"


def test_compile_onnx_module_nchw_to_nhwc(onnx_resnet50):
    # some CI environments wont offer ONNX, so skip in case it is not present
    pytest.importorskip("onnx")

    before, _ = tvmc.frontends.load_model(onnx_resnet50)

    expected_layout = "NHWC"
    after = tvmc.common.convert_graph_layout(before, expected_layout)

    layout_transform_calls = []

    def _is_layout_transform(node):
        if isinstance(node, tvm.relay.expr.Call):
            layout_transform_calls.append(
                node.op.name == "layout_transform"
                and node.attrs.src_layout == "NCHW"
                and node.attrs.dst_layout == "NHWC"
            )

    tvm.relay.analysis.post_order_visit(after["main"], _is_layout_transform)

    assert any(layout_transform_calls), "Expected 'layout_transform NCWH->NHWC' not found"


def test_compile_tflite_module__same_layout__nhwc_to_nhwc(tflite_mobilenet_v1_1_quant):
    # some CI environments wont offer TFLite, so skip in case it is not present
    pytest.importorskip("tflite")

    before, _ = tvmc.frontends.load_model(tflite_mobilenet_v1_1_quant)

    expected_layout = "NHWC"
    after = tvmc.common.convert_graph_layout(before, expected_layout)

    layout_transform_calls = []

    def _is_layout_transform(node):
        if isinstance(node, tvm.relay.expr.Call):
            layout_transform_calls.append(
                node.op.name == "layout_transform"
                and node.attrs.src_layout == "NHWC"
                and node.attrs.dst_layout == "NHWC"
            )

    tvm.relay.analysis.post_order_visit(after["main"], _is_layout_transform)

    assert not any(layout_transform_calls), "Unexpected 'layout_transform' call"


def test_compile_onnx_module__same_layout__nchw_to_nchw(onnx_resnet50):
    # some CI environments wont offer ONNX, so skip in case it is not present
    pytest.importorskip("onnx")

    before, _ = tvmc.frontends.load_model(onnx_resnet50)

    expected_layout = "NCHW"
    after = tvmc.common.convert_graph_layout(before, expected_layout)

    layout_transform_calls = []

    def _is_layout_transform(node):
        if isinstance(node, tvm.relay.expr.Call):
            layout_transform_calls.append(
                node.op.name == "layout_transform"
                and node.attrs.src_layout == "NCHW"
                and node.attrs.dst_layout == "NCHW"
            )

    tvm.relay.analysis.post_order_visit(after["main"], _is_layout_transform)

    assert not any(layout_transform_calls), "Unexpected 'layout_transform' call"


def test_tracker_host_port_from_cli__hostname_port():
    input_str = "1.2.3.4:9090"
    expected_host = "1.2.3.4"
    expected_port = 9090

    actual_host, actual_port = tvmc.common.tracker_host_port_from_cli(input_str)

    assert expected_host == actual_host
    assert expected_port == actual_port


def test_tracker_host_port_from_cli__hostname_port__empty():
    input_str = ""

    actual_host, actual_port = tvmc.common.tracker_host_port_from_cli(input_str)

    assert actual_host is None
    assert actual_port is None


def test_tracker_host_port_from_cli__only_hostname__default_port_is_9090():
    input_str = "1.2.3.4"
    expected_host = "1.2.3.4"
    expected_port = 9090

    actual_host, actual_port = tvmc.common.tracker_host_port_from_cli(input_str)

    assert expected_host == actual_host
    assert expected_port == actual_port


def test_shape_parser():
    # Check that a valid input is parsed correctly
    shape_string = "input:[10,10,10]"
    shape_dict = tvmc.common.parse_shape_string(shape_string)
    assert shape_dict == {"input": [10, 10, 10]}
    # Check that multiple valid input shapes are parse correctly
    shape_string = "input:[10,10,10] input2:[20,20,20,20]"
    shape_dict = tvmc.common.parse_shape_string(shape_string)
    assert shape_dict == {"input": [10, 10, 10], "input2": [20, 20, 20, 20]}
    # Check that alternate syntax parses correctly
    shape_string = "input: [10, 10, 10] input2: [20, 20, 20, 20]"
    shape_dict = tvmc.common.parse_shape_string(shape_string)
    assert shape_dict == {"input": [10, 10, 10], "input2": [20, 20, 20, 20]}
    shape_string = "input:[10,10,10],input2:[20,20,20,20]"
    shape_dict = tvmc.common.parse_shape_string(shape_string)
    assert shape_dict == {"input": [10, 10, 10], "input2": [20, 20, 20, 20]}
    # Check that negative dimensions parse to Any correctly.
    shape_string = "input:[-1,3,224,224]"
    shape_dict = tvmc.common.parse_shape_string(shape_string)
    # Convert to strings to allow comparison with Any.
    assert str(shape_dict) == "{'input': [?, 3, 224, 224]}"

    # Check that invalid pattern raises expected error.
    shape_string = "input:[a,10]"
    with pytest.raises(argparse.ArgumentTypeError):
        tvmc.common.parse_shape_string(shape_string)
    # Check that input with invalid separators raises error.
    shape_string = "input:5,10 input2:10,10"
    with pytest.raises(argparse.ArgumentTypeError):
        tvmc.common.parse_shape_string(shape_string)


def test_target_from_cli__error_duplicate():
    with pytest.raises(TVMCException):
        _ = tvmc.common.target_from_cli("llvm, llvm")


def test_target_from_cli__error_target_not_found():
    with pytest.raises(TVMCException):
        _ = tvmc.common.target_from_cli("invalidtarget")


def test_target_from_cli__error_no_tvm_target():
    with pytest.raises(TVMCException):
        _ = tvmc.common.target_from_cli("ethos-n77")


def test_tokenize_target_with_opts():
    tokens = tvmc.common.tokenize_target("foo -opt1=value1 --flag, bar -opt2=value2")
    expected_tokens = ["foo", "-opt1=value1", "--flag", ",", "bar", "-opt2=value2"]

    assert len(tokens) == len(expected_tokens)
    assert tokens == expected_tokens


def test_tokenize_target_with_plus_sign():
    tokens = tvmc.common.tokenize_target("foo -opt1=+value1 --flag, bar -opt2=test,+v")
    expected_tokens = ["foo", "-opt1=+value1", "--flag", ",", "bar", "-opt2=test,+v"]

    assert len(tokens) == len(expected_tokens)
    assert tokens == expected_tokens


def test_tokenize_target_with_commas():
    tokens = tvmc.common.tokenize_target("foo -opt1=v,a,l,u,e,1 --flag")
    expected_tokens = ["foo", "-opt1=v,a,l,u,e,1", "--flag"]

    assert len(tokens) == len(expected_tokens)
    assert tokens == expected_tokens


def test_tokenize_target_with_commas_and_single_quotes():
    tokens = tvmc.common.tokenize_target("foo -opt1='v, a, l, u, e', bar")
    expected_tokens = ["foo", "-opt1='v, a, l, u, e'", ",", "bar"]

    assert len(tokens) == len(expected_tokens)
    assert tokens == expected_tokens


def test_tokenize_target_with_commas_and_double_quotes():
    tokens = tvmc.common.tokenize_target('foo -opt1="v, a, l, u, e", bar')
    expected_tokens = ["foo", '-opt1="v, a, l, u, e"', ",", "bar"]

    assert len(tokens) == len(expected_tokens)
    assert tokens == expected_tokens


def test_tokenize_target_with_dashes():
    tokens = tvmc.common.tokenize_target("foo-bar1 -opt-1=t-e-s-t, baz")
    expected_tokens = ["foo-bar1", "-opt-1=t-e-s-t", ",", "baz"]

    assert len(tokens) == len(expected_tokens)
    assert tokens == expected_tokens


def test_parse_single_target_with_opts():
    targets = tvmc.common.parse_target("llvm -device=arm_cpu --system-lib")

    assert len(targets) == 1
    assert "device" in targets[0]["opts"]
    assert "system-lib" in targets[0]["opts"]


def test_parse_multiple_target():
    targets = tvmc.common.parse_target("compute-library, llvm -device=arm_cpu --system-lib")

    assert len(targets) == 2
    assert "compute-library" == targets[0]["name"]
    assert "llvm" == targets[1]["name"]


def test_parse_multiple_target_with_opts():
    targets = tvmc.common.parse_target("ethos-n77 -myopt=value, llvm -device=arm_cpu --system-lib")

    assert len(targets) == 2
    assert "ethos-n77" == targets[0]["name"]
    assert "myopt" in targets[0]["opts"]
    assert "value" == targets[0]["opts"]["myopt"]
    assert "llvm" == targets[1]["name"]


def test_parse_quotes_and_separators_on_options():
    targets_no_quote = tvmc.common.parse_target("foo -option1=+v1.0x,+value,+bar")
    targets_single_quote = tvmc.common.parse_target("foo -option1='+v1.0x,+value'")
    targets_double_quote = tvmc.common.parse_target('foo -option1="+v1.0x,+value"')

    assert len(targets_no_quote) == 1
    assert "+v1.0x,+value,+bar" == targets_no_quote[0]["opts"]["option1"]

    assert len(targets_single_quote) == 1
    assert "+v1.0x,+value" == targets_single_quote[0]["opts"]["option1"]

    assert len(targets_double_quote) == 1
    assert "+v1.0x,+value" == targets_double_quote[0]["opts"]["option1"]
