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
from tvm.driver import tvmc


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
