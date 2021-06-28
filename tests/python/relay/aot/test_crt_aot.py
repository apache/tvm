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

import os
import io
import struct
import numpy as np
import pathlib
import shutil
import subprocess
import tempfile
import tarfile
import pytest

import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.op.contrib import get_pattern_table
from tvm.contrib import utils
from tvm.relay.backend import compile_engine
from tvm.contrib import utils
from tvm.contrib import graph_executor
from tvm.micro import export_model_library_format
from tvm.relay import testing
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.contrib import utils
from tvm.relay.expr_functor import ExprMutator

from aot_test_utils import *


@pytest.mark.parametrize("use_calculated_workspaces", [True, False])
@pytest.mark.parametrize("target_options", ["--unpacked-api=0", "--unpacked-api=1"])
def test_conv_with_params(use_calculated_workspaces, target_options):
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
    compile_and_run(mod, input_list, output_list, target_options, use_calculated_workspaces, params)


@pytest.mark.parametrize("use_calculated_workspaces", [True, False])
@pytest.mark.parametrize("target_options", ["--unpacked-api=0", "--unpacked-api=1"])
def test_add_with_params(use_calculated_workspaces, target_options):
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
    compile_and_run(
        func, input_list, output_list, target_options, use_calculated_workspaces, params
    )


@pytest.mark.parametrize("use_calculated_workspaces", [True, False])
@pytest.mark.parametrize("target_options", ["--unpacked-api=0", "--unpacked-api=1"])
def test_conv2d(use_calculated_workspaces, target_options):
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
        compile_and_run(mod, input_list, output_list, target_options, use_calculated_workspaces)


@pytest.mark.parametrize("use_calculated_workspaces", [True, False])
@pytest.mark.parametrize("target_options", ["--unpacked-api=0", "--unpacked-api=1"])
def test_concatenate(use_calculated_workspaces, target_options):
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
    compile_and_run(func, input_list, output_list, target_options, use_calculated_workspaces)


@pytest.mark.parametrize("use_calculated_workspaces", [True, False])
@pytest.mark.parametrize("target_options", ["--unpacked-api=0", "--unpacked-api=1"])
def test_nested_tuples(use_calculated_workspaces, target_options):
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
    compile_and_run(func, input_list, output_list, target_options, use_calculated_workspaces)


@pytest.mark.parametrize("use_calculated_workspaces", [True, False])
@pytest.mark.parametrize("target_options", ["--unpacked-api=0", "--unpacked-api=1"])
def test_tuple_getitem(use_calculated_workspaces, target_options):
    func = relay.Function([], relay.TupleGetItem(relay.Tuple([relay.const(1), relay.const(2)]), 0))
    output_list = generate_ref_data(func, {})
    input_list = []
    compile_and_run(func, input_list, output_list, target_options, use_calculated_workspaces)


@pytest.mark.parametrize("use_calculated_workspaces", [True, False])
@pytest.mark.parametrize("target_options", ["--unpacked-api=0", "--unpacked-api=1"])
def test_id(use_calculated_workspaces, target_options):
    x = relay.var("x", "float32")
    ident = relay.Function([x], x)
    one = np.array(1.0, "float32")
    inputs = {"x": one}
    output_list = generate_ref_data(ident, inputs)
    input_list = [one]
    compile_and_run(ident, input_list, output_list, target_options, use_calculated_workspaces)


@pytest.mark.parametrize("use_calculated_workspaces", [True, False])
@pytest.mark.parametrize("target_options", ["--unpacked-api=0", "--unpacked-api=1"])
def test_add_const(use_calculated_workspaces, target_options):
    two = relay.add(relay.const(1), relay.const(1))
    func = relay.Function([], two)
    output_list = generate_ref_data(func, {})
    input_list = []
    compile_and_run(func, input_list, output_list, target_options, use_calculated_workspaces)


@pytest.mark.parametrize("use_calculated_workspaces", [True, False])
@pytest.mark.parametrize("target_options", ["--unpacked-api=0", "--unpacked-api=1"])
def test_mul_param(use_calculated_workspaces, target_options):
    x = relay.var("x", shape=(10, 10))
    y = relay.var("y", shape=(1, 10))
    func = relay.Function([x, y], relay.multiply(x, y))
    x_data = np.random.rand(10, 10).astype("float32")
    y_data = np.random.rand(1, 10).astype("float32")
    inputs = {"x": x_data, "y": y_data}
    output_list = generate_ref_data(func, inputs)
    input_list = [inputs["x"], inputs["y"]]
    compile_and_run(func, input_list, output_list, target_options, use_calculated_workspaces)


@pytest.mark.parametrize("use_calculated_workspaces", [True, False])
@pytest.mark.parametrize("target_options", ["--unpacked-api=0", "--unpacked-api=1"])
def test_subtract(use_calculated_workspaces, target_options):
    i = relay.var("i", shape=[], dtype="int32")
    sub = relay.subtract(i, relay.const(1, dtype="int32"))
    func = relay.Function([i], sub, ret_type=relay.TensorType([], "int32"))
    i_data = np.array(1, dtype="int32")
    inputs = {"i": i_data}
    output_list = generate_ref_data(func, inputs)
    input_list = [inputs["i"]]
    compile_and_run(func, input_list, output_list, target_options, use_calculated_workspaces)


@pytest.mark.parametrize("use_calculated_workspaces", [True, False])
@pytest.mark.parametrize("target_options", ["--unpacked-api=0", "--unpacked-api=1"])
def test_tuple_output(use_calculated_workspaces, target_options):
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
    compile_and_run(func, input_list, output_list, target_options, use_calculated_workspaces)


@pytest.mark.parametrize(
    "use_calculated_workspaces_and_alignment", [(True, 1), (True, 16), (False, 1)]
)
@pytest.mark.parametrize("target_options", ["--unpacked-api"])
def test_mobilenet(use_calculated_workspaces_and_alignment, target_options):
    use_calculated_workspaces = use_calculated_workspaces_and_alignment[0]
    workspace_byte_alignment = use_calculated_workspaces_and_alignment[1]

    mod, params = testing.mobilenet.get_workload(batch_size=1)
    data_shape = [int(x) for x in mod["main"].checked_type.arg_types[0].shape]
    data = np.random.uniform(size=data_shape).astype("float32")
    inputs = {"data": data}
    output_list = generate_ref_data(mod, inputs, params)
    input_list = [inputs["data"]]
    compile_and_run(
        mod,
        input_list,
        output_list,
        target_options,
        use_calculated_workspaces,
        params,
        workspace_byte_alignment,
    )


class CcompilerAnnotator(ExprMutator):
    """
    This is used to create external functions for ccompiler.
    A simple annotator that creates the following program:
           |
      -- begin --
           |
          add
           |
        subtract
           |
        multiply
           |
       -- end --
           |
    """

    def __init__(self):
        super(CcompilerAnnotator, self).__init__()
        self.in_compiler = 0

    def visit_call(self, call):
        if call.op.name == "add":  # Annotate begin at args
            if self.in_compiler == 1:
                lhs = compiler_begin(super().visit(call.args[0]), "ccompiler")
                rhs = compiler_begin(super().visit(call.args[1]), "ccompiler")
                op = relay.add(lhs, rhs)
                self.in_compiler = 2
                return op
        elif call.op.name == "subtract":
            if self.in_compiler == 1:
                lhs = super().visit(call.args[0])
                rhs = super().visit(call.args[1])
                if isinstance(lhs, relay.expr.Var):
                    lhs = compiler_begin(lhs, "ccompiler")
                if isinstance(rhs, relay.expr.Var):
                    rhs = compiler_begin(rhs, "ccompiler")
                return relay.subtract(lhs, rhs)
        elif call.op.name == "multiply":  # Annotate end at output
            self.in_compiler = 1
            lhs = super().visit(call.args[0])
            rhs = super().visit(call.args[1])
            if isinstance(lhs, relay.expr.Var):
                lhs = compiler_begin(lhs, "ccompiler")
            if isinstance(rhs, relay.expr.Var):
                rhs = compiler_begin(rhs, "ccompiler")
            op = relay.multiply(lhs, rhs)
            if self.in_compiler == 2:
                op = compiler_end(op, "ccompiler")
            self.in_compiler = 0
            return op
        return super().visit_call(call)


@pytest.mark.parametrize("use_calculated_workspaces", [True, False])
@pytest.mark.parametrize("target_options", [""])
def test_byoc_microtvm(use_calculated_workspaces, target_options):
    """This is a simple test case to check BYOC capabilities of AOT"""
    x = relay.var("x", shape=(10, 10))
    w0 = relay.var("w0", shape=(10, 10))
    w1 = relay.var("w1", shape=(10, 10))
    w2 = relay.var("w2", shape=(10, 10))
    w3 = relay.var("w3", shape=(10, 10))
    w4 = relay.var("w4", shape=(10, 10))
    w5 = relay.var("w5", shape=(10, 10))
    w6 = relay.var("w6", shape=(10, 10))
    w7 = relay.var("w7", shape=(10, 10))

    # C compiler
    z0 = relay.add(x, w0)
    p0 = relay.subtract(z0, w1)
    q0 = relay.multiply(p0, w2)

    z1 = relay.add(x, w3)
    p1 = relay.subtract(z1, w4)
    q1 = relay.multiply(p1, w5)

    # Other parts on TVM
    z2 = relay.add(x, w6)
    q2 = relay.subtract(z2, w7)

    r = relay.concatenate((q0, q1, q2), axis=0)
    f = relay.Function([x, w0, w1, w2, w3, w4, w5, w6, w7], r)
    mod = tvm.IRModule()
    ann = CcompilerAnnotator()
    mod["main"] = ann.visit(f)

    mod = tvm.relay.transform.PartitionGraph("mod_name")(mod)
    mod = tvm.relay.transform.InferType()(mod)

    x_data = np.random.rand(10, 10).astype("float32")
    w_data = []
    for _ in range(8):
        w_data.append(np.random.rand(10, 10).astype("float32"))

    map_inputs = {"w{}".format(i): w_data[i] for i in range(8)}
    map_inputs["x"] = x_data
    output_list = generate_ref_data(mod, map_inputs)
    input_list = [map_inputs["x"]]
    input_list.extend([map_inputs["w{}".format(i)] for i in range(8)])
    compile_and_run(
        mod, input_list, output_list, target_options, use_calculated_workspaces, mod_name="my_mod"
    )


@pytest.mark.parametrize("target_options", ["--unpacked-api=0", "--unpacked-api=1"])
def test_add_name_mangling_with_params(target_options):
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
    compile_and_run(
        func,
        input_list,
        output_list,
        target_options,
        use_calculated_workspaces=False,
        params=params,
        mod_name="my_mod",
    )


@pytest.mark.parametrize("target_options", ["--unpacked-api=0", "--unpacked-api=1"])
def test_multiple_models(target_options):
    # Identity model without params
    x = relay.var("x", "float32")
    mod1 = relay.Function([x], x)
    one = np.array(1.0, "float32")
    inputs1 = {"x": one}
    output_list1 = generate_ref_data(mod1, inputs1)
    input_list1 = [one]
    params1 = None

    # Convolution model
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
    mod2 = tvm.parser.fromtext(RELAY_MODEL)
    main_func = mod2["main"]
    shape_dict = {p.name_hint: p.checked_type.concrete_shape for p in main_func.params}
    type_dict = {p.name_hint: p.checked_type.dtype for p in main_func.params}

    weight_data = np.ones(shape_dict["weight"]).astype(type_dict["weight"])
    input_data = np.ones(shape_dict["data"]).astype(type_dict["data"])

    params2 = {"weight": weight_data}
    inputs2 = {"data": input_data}
    output_list2 = generate_ref_data(mod2, inputs2, params2)
    input_list2 = [input_data]

    input_list_map = {"mod1": input_list1, "mod2": input_list2}
    output_list_map = {"mod1": output_list1, "mod2": output_list2}
    mod_map = {"mod1": mod1, "mod2": mod2}
    param_map = {"mod1": params1, "mod2": params2}

    compile_and_run_multiple_models(
        mod_map, input_list_map, output_list_map, target_options, param_map
    )


if __name__ == "__main__":
    pytest.main([__file__])
