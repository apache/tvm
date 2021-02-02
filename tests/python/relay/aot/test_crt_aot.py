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

import tflite
import os
import io
import struct
import numpy as np
import pathlib
import shutil
import subprocess
import tempfile
import tarfile


import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.op.contrib import get_pattern_table
from tvm.contrib import utils
from tvm.relay.backend import compile_engine
from tvm.contrib import utils
from tvm.contrib import graph_runtime
from tvm.micro import export_model_library_format
from tvm.relay import testing

from infra import *


def test_conv_with_params():
    RELAY_MODEL = """
#[version = "0.0.5"]
def @main(%data : Tensor[(1, 3, 64, 64), uint8], %weight : Tensor[(8, 3, 5, 5), int8]) {
    %1 = nn.conv2d(
         %data,
         %weight,
         padding=[2, 2],
         channels=8,
         kernel_size=[5, 5],
         data_layout="NCHW",
         kernel_layout="OIHW",
         out_dtype="int32");
  %1
}
"""
    mod = tvm.parser.fromtext(RELAY_MODEL)
    main_func = mod["main"]
    shape_dict = {p.name_hint: p.checked_type.concrete_shape for p in main_func.params}
    type_dict = {p.name_hint: p.checked_type.dtype for p in main_func.params}

    weight_data = np.ones(shape_dict["weight"]).astype(type_dict["weight"])
    input_data = np.ones(shape_dict["data"]).astype(type_dict["data"])

    params = {"weight": weight_data}
    inputs = {"data": input_data}
    output_list = generate_ref_data(mod, inputs, params)

    input_list = [input_data]
    verify_source(mod, input_list, output_list, params)


def test_add_with_params():
    x = relay.var("x", shape=(1, 10))
    y = relay.var("y", shape=(1, 10))
    z = relay.add(x, y)
    func = relay.Function([x, y], z)

    x_in = np.ones((1, 10)).astype("float32")
    y_in = np.random.uniform(size=(1, 10)).astype("float32")

    params = {"x": x_in}
    inputs = {"y": y_in}
    output_list = generate_ref_data(func, inputs, params)

    input_list = [y_in]
    verify_source(func, input_list, output_list, params)


def test_conv2d():
    """Test a subgraph with a single conv2d operator."""

    def conv2d_direct():
        dtype = "float32"
        ishape = (1, 32, 14, 14)
        w1shape = (32, 32, 3, 3)

        data0 = relay.var("data", shape=ishape, dtype=dtype)
        weight0 = relay.var("weight", shape=w1shape, dtype=dtype)
        out = relay.nn.conv2d(data0, weight0, kernel_size=(3, 3), padding=(1, 1))
        main_f = relay.Function([data0, weight0], out)
        mod = tvm.IRModule()
        mod["main"] = main_f
        mod = transform.InferType()(mod)

        i_data = np.random.uniform(0, 1, ishape).astype(dtype)
        w1_data = np.random.uniform(0, 1, w1shape).astype(dtype)

        return mod, {"data": i_data, "weight": w1_data}, (1, 32, 14, 14)

    def group_conv2d():
        dtype = "float32"
        ishape = (1, 32, 14, 14)
        w2shape = (32, 1, 3, 3)

        data0 = relay.var("data", shape=(ishape), dtype=dtype)
        weight0 = relay.var("weight", shape=(w2shape), dtype=dtype)
        out = relay.nn.conv2d(data0, weight0, kernel_size=(3, 3), padding=(1, 1), groups=32)
        main_f = relay.Function([data0, weight0], out)
        mod = tvm.IRModule()
        mod["main"] = main_f
        mod = transform.InferType()(mod)

        i_data = np.random.uniform(0, 1, ishape).astype(dtype)
        w_data = np.random.uniform(0, 1, w2shape).astype(dtype)

        return mod, {"data": i_data, "weight": w_data}, (1, 32, 14, 14)

    for mod, inputs, out_shape in [conv2d_direct(), group_conv2d()]:
        output_list = generate_ref_data(mod, inputs)
        input_list = [inputs["data"], inputs["weight"]]
        verify_source(mod, input_list, output_list)


def test_concatenate():
    dtype = "float32"
    x = relay.var("x", shape=(10, 5), dtype=dtype)
    y = relay.var("y", shape=(10, 5), dtype=dtype)
    t = relay.var("z", shape=(), dtype=dtype)
    z = relay.concatenate((x, y), axis=1)
    z = relay.add(z, t)
    # Check result.
    func = relay.Function([x, y, t], z)
    x_data = np.random.rand(10, 5).astype(dtype)
    y_data = np.random.rand(10, 5).astype(dtype)
    t_data = np.random.uniform(size=()).astype(dtype)
    inputs = {"x": x_data, "y": y_data, "z": t_data}

    output_list = generate_ref_data(func, inputs)
    input_list = [inputs["x"], inputs["y"], inputs["z"]]
    verify_source(func, input_list, output_list)


def test_nested_tuples():
    x = relay.var("x", shape=(10,))
    x1 = x + relay.const(1.0)
    x2 = x1 + relay.const(1.0)
    x3 = x2 + relay.const(1.0)
    x4 = x3 + relay.const(1.0)
    out = relay.Tuple([x1, relay.Tuple([relay.Tuple([x2, x3]), x4])])
    func = relay.Function([x], out)

    x_data = np.random.uniform(size=(10,)).astype(np.float32)
    inputs = {"x": x_data}
    output_list = generate_ref_data(func, inputs)
    input_list = [x_data]
    verify_source(func, input_list, output_list)


def test_tuple_getitem():
    func = relay.Function([], relay.TupleGetItem(relay.Tuple([relay.const(1), relay.const(2)]), 0))
    output_list = generate_ref_data(func, {})
    input_list = []
    verify_source(func, input_list, output_list)


def test_id():
    x = relay.var("x", "float32")
    ident = relay.Function([x], x)
    one = np.array(1.0, "float32")
    inputs = {"x": one}
    output_list = generate_ref_data(ident, inputs)
    input_list = [one]
    verify_source(ident, input_list, output_list)


def test_add_const():
    two = relay.add(relay.const(1), relay.const(1))
    func = relay.Function([], two)
    output_list = generate_ref_data(func, {})
    input_list = []
    verify_source(func, input_list, output_list)


def test_mul_param():
    x = relay.var("x", shape=(10, 10))
    y = relay.var("y", shape=(1, 10))
    func = relay.Function([x, y], relay.multiply(x, y))
    x_data = np.random.rand(10, 10).astype("float32")
    y_data = np.random.rand(1, 10).astype("float32")
    inputs = {"x": x_data, "y": y_data}
    output_list = generate_ref_data(func, inputs)
    input_list = [inputs["x"], inputs["y"]]
    verify_source(func, input_list, output_list)


def test_subtract():
    i = relay.var("i", shape=[], dtype="int32")
    sub = relay.subtract(i, relay.const(1, dtype="int32"))
    func = relay.Function([i], sub, ret_type=relay.TensorType([], "int32"))
    i_data = np.array(1, dtype="int32")
    inputs = {"i": i_data}
    output_list = generate_ref_data(func, inputs)
    input_list = [inputs["i"]]
    verify_source(func, input_list, output_list)


def test_tuple_output():
    x = relay.var("x", shape=(6, 9))
    y = relay.split(x, 3).astuple()
    a = relay.TupleGetItem(y, 0)
    b = relay.TupleGetItem(y, 1)
    c = relay.TupleGetItem(y, 2)
    out = relay.Tuple([a, b])
    func = relay.Function([x], out)
    x_data = np.random.rand(6, 9).astype("float32")
    inputs = {"x": x_data}
    output_list = generate_ref_data(func, inputs)
    input_list = [inputs["x"]]
    verify_source(func, input_list, output_list)


def test_mobilenet():
    mod, params = testing.mobilenet.get_workload(batch_size=1)
    data_shape = [int(x) for x in mod["main"].checked_type.arg_types[0].shape]
    data = np.random.uniform(size=data_shape).astype("float32")
    inputs = {"data": data}
    output_list = generate_ref_data(mod, inputs, params)
    input_list = [inputs["data"]]
    verify_source(mod, input_list, output_list, params)


if __name__ == "__main__":
    test_tuple_output()
    test_mobilenet()
    test_subtract()
    test_mul_param()
    test_id()
    test_add_const()
    test_tuple_getitem()
    test_nested_tuples()
    test_concatenate()
    test_conv_with_params()
    test_add_with_params()
    test_conv2d()
