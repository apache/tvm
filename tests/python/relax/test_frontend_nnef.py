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

from functools import reduce
from math import prod

import numpy as np

import _nnef
import nnef
import pytest

import tvm
import tvm.testing
from tvm import relax
import tvm.relax.frontend.nnef

from tvm.script import relax as R
from tvm.script import tir as T
from tvm.relax.testing import nn
import tvm.topi as topi

# test parameters
dims = [2, 3, 4]
dims_conv = [1, 2, 3]


def gen_case_graph_unary(operation, dim, dt="scalar"):
    return """
        version 1.0;

        graph G( input ) -> ( output )
        {{
            input = external<{dtype}>(shape = {shape});
            output = {op}(input);
        }}
        """.format(
        op=operation, shape=[4, 16] + [32] * (dim - 2), dtype=dt
    )


def gen_case_graph_unary_attrs(operation, dim, dt="scalar", attrs=""):
    return """
        version 1.0;

        graph G( input ) -> ( output )
        {{
            input = external<{dtype}>(shape = {shape});
            output = {op}(input{attrs});
        }}
        """.format(
        op=operation, shape=[4, 16] + [32] * (dim - 2), attrs=attrs, dtype=dt
    )


def gen_case_graph_binary(operation, dim, dt="scalar", broadcast=False):
    return """
        version 1.0;

        graph G( input1, input2 ) -> ( output )
        {{
            input1 = external<{dtype}>(shape = {shape1});
            input2 = external<{dtype}>(shape = {shape2});
            output = {op}(input1, input2);
        }}
        """.format(
        op=operation,
        shape1=[4, 16] + [32] * (dim - 2),
        shape2=[4, 16] + [32] * (dim - 2) if not broadcast else [4, 16] + [1] * (dim - 2),
        dtype=dt,
    )


def gen_case_graph_unary_vars2(
    operation, dim, attrs, dt="scalar", shape1=None, shape2=None, shape3=None
):
    return """
        version 1.0;

        graph G( input1 ) -> ( output )
        {{
            input1 = external<{dtype}>(shape = {shape1});
            input2 = variable<{dtype}>(shape = {shape2}, label="kernel");
            input3 = variable<{dtype}>(shape = {shape3}, label="bias");
            output = {op}(input1, input2, input3{attrs});
        }}
        """.format(
        op=operation,
        shape1=[4, 8] + [32] * dim if not shape1 else shape1,
        shape2=[16, 8] + [3] * dim if not shape2 else shape2,
        shape3=[1, 16] if not shape3 else shape3,
        attrs=attrs,
        dtype=dt,
    )


def fill_variables(graph):
    # some operations (conv, box, pools etc.) have variable tensors as parameters, and they need to be filled
    for operation in graph.operations:
        if operation.name == "variable":
            tensor_name = operation.outputs["output"]

            shape = operation.attribs["shape"]

            # they are only scalar in these cases, so float32 is fine for default value
            assert (
                operation.dtype == "scalar"
            ), f"variable of type {operation.dtype} is not supported, please update fill_variables"

            data = np.ones(shape).astype("float32")

            tensor = graph.tensors[tensor_name]
            graph.tensors[tensor_name] = _nnef.Tensor(
                tensor.name, tensor.dtype, shape, data, tensor.quantization
            )


def get_unary_mod(method, dim, dt="float32"):
    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp = nn.Placeholder([4, 16] + [32] * (dim - 2), name="inp", dtype=dt)
            out = method(inp)
            out = bb.emit_output(out)
        bb.emit_func_output(out, [inp])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})

    return expected


def get_binary_mod(method, dim, dt="float32", broadcast=False, dt2=None):
    if not dt2:
        dt2 = dt
    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16] + [32] * (dim - 2), name="inp1", dtype=dt)
            inp2 = nn.Placeholder(
                [4, 16] + [32] * (dim - 2) if not broadcast else [4, 16] + [1] * (dim - 2),
                name="inp2",
                dtype=dt2,
            )
            out = method(inp1, inp2)
            out = bb.emit_output(out)
        bb.emit_func_output(out, [inp1, inp2])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 2})

    return expected


def get_tertiary_mod(method, dim, dt="float32", dt2=None, shape1=None, shape2=None, shape3=None):
    if not dt2:
        dt2 = dt
    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder(
                [4, 8] + [32] * dim if not shape1 else shape1, name="inp1", dtype=dt
            )
            inp2 = nn.Placeholder(
                [16, 8] + [3] * dim if not shape2 else shape2, name="inp2", dtype=dt2
            )
            inp3 = nn.Placeholder([1, 16] if not shape3 else shape3, name="inp3", dtype=dt2)
            out = method(inp1, inp2, inp3)
            out = bb.emit_output(out)
        bb.emit_func_output(out, [inp1, inp2, inp3])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})

    return expected


def verify_model_struct(graph_str, binding, expected):
    graph = nnef.parse_string(graph_str)
    fill_variables(graph)
    mod = relax.frontend.nnef.from_nnef(graph)

    binding = {k: tvm.nd.array(v) for k, v in binding.items()}
    expected = relax.transform.BindParams("main", binding)(expected)

    tvm.ir.assert_structural_equal(mod, expected)


@pytest.mark.parametrize("dim", dims)
def test_copy(dim):
    def method(inp):
        bb = relax.BlockBuilder.current()
        return bb.emit_te(topi.identity, inp)

    expected = get_unary_mod(method, dim)
    graph = gen_case_graph_unary("copy", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_neg(dim):
    expected = get_unary_mod(R.negative, dim)
    graph = gen_case_graph_unary("neg", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_rcp(dim):
    expected = get_unary_mod(lambda x: R.divide(R.const(1, "float32"), x), dim)
    graph = gen_case_graph_unary("rcp", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_exp(dim):
    expected = get_unary_mod(R.exp, dim)
    graph = gen_case_graph_unary("exp", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_log(dim):
    expected = get_unary_mod(R.log, dim)
    graph = gen_case_graph_unary("log", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_sin(dim):
    expected = get_unary_mod(R.sin, dim)
    graph = gen_case_graph_unary("sin", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_cos(dim):
    expected = get_unary_mod(R.cos, dim)
    graph = gen_case_graph_unary("cos", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_tan(dim):
    expected = get_unary_mod(R.tan, dim)
    graph = gen_case_graph_unary("tan", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_sinh(dim):
    expected = get_unary_mod(R.sinh, dim)
    graph = gen_case_graph_unary("sinh", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_cosh(dim):
    expected = get_unary_mod(R.cosh, dim)
    graph = gen_case_graph_unary("cosh", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_tanh(dim):
    expected = get_unary_mod(R.tanh, dim)
    graph = gen_case_graph_unary("tanh", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_asin(dim):
    expected = get_unary_mod(R.asin, dim)
    graph = gen_case_graph_unary("asin", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_acos(dim):
    expected = get_unary_mod(R.acos, dim)
    graph = gen_case_graph_unary("acos", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_atan(dim):
    expected = get_unary_mod(R.atan, dim)
    graph = gen_case_graph_unary("atan", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_asinh(dim):
    expected = get_unary_mod(R.asinh, dim)
    graph = gen_case_graph_unary("asinh", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_acosh(dim):
    expected = get_unary_mod(R.acosh, dim)
    graph = gen_case_graph_unary("acosh", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_atanh(dim):
    expected = get_unary_mod(R.atanh, dim)
    graph = gen_case_graph_unary("atanh", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_abs(dim):
    expected = get_unary_mod(R.abs, dim)
    graph = gen_case_graph_unary("abs", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_sign(dim):
    expected = get_unary_mod(R.sign, dim)
    graph = gen_case_graph_unary("sign", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_not(dim):
    expected = get_unary_mod(R.logical_not, dim, dt="bool")
    graph = gen_case_graph_unary("not", dim, dt="logical")
    verify_model_struct(graph, binding={}, expected=expected)


@pytest.mark.parametrize("dim", dims)
def test_floor(dim):
    expected = get_unary_mod(R.floor, dim)
    graph = gen_case_graph_unary("floor", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_ceil(dim):
    expected = get_unary_mod(R.ceil, dim)
    graph = gen_case_graph_unary("ceil", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_round(dim):
    expected = get_unary_mod(R.round, dim)
    graph = gen_case_graph_unary("round", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_add(dim):
    expected = get_binary_mod(R.add, dim)
    graph = gen_case_graph_binary("add", dim)
    verify_model_struct(graph, {}, expected)

    expected = get_binary_mod(R.add, dim, broadcast=True)
    graph = gen_case_graph_binary("add", dim, broadcast=True)
    verify_model_struct(graph, {}, expected)

    expected = get_unary_mod(lambda x: R.add(x, R.const(0.5, "float32")), dim)
    graph = gen_case_graph_unary_attrs("add", dim, attrs=", 0.5")
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_sub(dim):
    expected = get_binary_mod(R.subtract, dim)
    graph = gen_case_graph_binary("sub", dim)
    verify_model_struct(graph, {}, expected)

    expected = get_binary_mod(R.subtract, dim, broadcast=True)
    graph = gen_case_graph_binary("sub", dim, broadcast=True)
    verify_model_struct(graph, {}, expected)

    expected = get_unary_mod(lambda x: R.subtract(x, R.const(0.5, "float32")), dim)
    graph = gen_case_graph_unary_attrs("sub", dim, attrs=", 0.5")
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_mul(dim):
    expected = get_binary_mod(R.multiply, dim)
    graph = gen_case_graph_binary("mul", dim)
    verify_model_struct(graph, {}, expected)

    expected = get_binary_mod(R.multiply, dim, broadcast=True)
    graph = gen_case_graph_binary("mul", dim, broadcast=True)
    verify_model_struct(graph, {}, expected)

    expected = get_unary_mod(lambda x: R.multiply(x, R.const(0.5, "float32")), dim)
    graph = gen_case_graph_unary_attrs("mul", dim, attrs=", 0.5")
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_div(dim):
    expected = get_binary_mod(R.divide, dim)
    graph = gen_case_graph_binary("div", dim)
    verify_model_struct(graph, {}, expected)

    expected = get_binary_mod(R.divide, dim, broadcast=True)
    graph = gen_case_graph_binary("div", dim, broadcast=True)
    verify_model_struct(graph, {}, expected)

    expected = get_unary_mod(lambda x: R.divide(x, R.const(0.5, "float32")), dim)
    graph = gen_case_graph_unary_attrs("div", dim, attrs=", 0.5")
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_pow(dim):
    expected = get_binary_mod(R.power, dim)
    graph = gen_case_graph_binary("pow", dim)
    verify_model_struct(graph, {}, expected)

    expected = get_binary_mod(R.power, dim, broadcast=True)
    graph = gen_case_graph_binary("pow", dim, broadcast=True)
    verify_model_struct(graph, {}, expected)

    expected = get_unary_mod(lambda x: R.power(x, R.const(0.5, "float32")), dim)
    graph = gen_case_graph_unary_attrs("pow", dim, attrs=", 0.5")
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_lt(dim):
    expected = get_binary_mod(R.less, dim)
    graph = gen_case_graph_binary("lt", dim)
    verify_model_struct(graph, {}, expected)

    expected = get_binary_mod(R.less, dim, broadcast=True)
    graph = gen_case_graph_binary("lt", dim, broadcast=True)
    verify_model_struct(graph, {}, expected)

    expected = get_unary_mod(lambda x: R.less(x, R.const(0.5, "float32")), dim)
    graph = gen_case_graph_unary_attrs("lt", dim, attrs=", 0.5")
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_gt(dim):
    expected = get_binary_mod(R.greater, dim)
    graph = gen_case_graph_binary("gt", dim)
    verify_model_struct(graph, {}, expected)

    expected = get_binary_mod(R.greater, dim, broadcast=True)
    graph = gen_case_graph_binary("gt", dim, broadcast=True)
    verify_model_struct(graph, {}, expected)

    expected = get_unary_mod(lambda x: R.greater(x, R.const(0.5, "float32")), dim)
    graph = gen_case_graph_unary_attrs("gt", dim, attrs=", 0.5")
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_le(dim):
    expected = get_binary_mod(R.less_equal, dim)
    graph = gen_case_graph_binary("le", dim)
    verify_model_struct(graph, {}, expected)

    expected = get_binary_mod(R.less_equal, dim, broadcast=True)
    graph = gen_case_graph_binary("le", dim, broadcast=True)
    verify_model_struct(graph, {}, expected)

    expected = get_unary_mod(lambda x: R.less_equal(x, R.const(0.5, "float32")), dim)
    graph = gen_case_graph_unary_attrs("le", dim, attrs=", 0.5")
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_ge(dim):
    expected = get_binary_mod(R.greater_equal, dim)
    graph = gen_case_graph_binary("ge", dim)
    verify_model_struct(graph, {}, expected)

    expected = get_binary_mod(R.greater_equal, dim, broadcast=True)
    graph = gen_case_graph_binary("ge", dim, broadcast=True)
    verify_model_struct(graph, {}, expected)

    expected = get_unary_mod(lambda x: R.greater_equal(x, R.const(0.5, "float32")), dim)
    graph = gen_case_graph_unary_attrs("ge", dim, attrs=", 0.5")
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_eq(dim):
    expected = get_binary_mod(R.equal, dim)
    graph = gen_case_graph_binary("eq", dim)
    verify_model_struct(graph, {}, expected)

    expected = get_binary_mod(R.equal, dim, broadcast=True)
    graph = gen_case_graph_binary("eq", dim, broadcast=True)
    verify_model_struct(graph, {}, expected)

    expected = get_unary_mod(lambda x: R.equal(x, R.const(0.5, "float32")), dim)
    graph = gen_case_graph_unary_attrs("eq", dim, attrs=", 0.5")
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_ne(dim):
    expected = get_binary_mod(R.not_equal, dim)
    graph = gen_case_graph_binary("ne", dim)
    verify_model_struct(graph, {}, expected)

    expected = get_binary_mod(R.not_equal, dim, broadcast=True)
    graph = gen_case_graph_binary("ne", dim, broadcast=True)
    verify_model_struct(graph, {}, expected)

    expected = get_unary_mod(lambda x: R.not_equal(x, R.const(0.5, "float32")), dim)
    graph = gen_case_graph_unary_attrs("ne", dim, attrs=", 0.5")
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_and(dim):
    expected = get_binary_mod(R.logical_and, dim, dt="bool")
    graph = gen_case_graph_binary("and", dim, dt="logical")
    verify_model_struct(graph, {}, expected)

    expected = get_binary_mod(R.logical_and, dim, broadcast=True, dt="bool")
    graph = gen_case_graph_binary("and", dim, broadcast=True, dt="logical")
    verify_model_struct(graph, {}, expected)

    expected = get_unary_mod(lambda x: R.logical_and(x, R.const(False, "bool")), dim, dt="bool")
    graph = gen_case_graph_unary_attrs("and", dim, attrs=", false", dt="logical")
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_or(dim):
    expected = get_binary_mod(R.logical_or, dim, dt="bool")
    graph = gen_case_graph_binary("or", dim, dt="logical")
    verify_model_struct(graph, {}, expected)

    expected = get_binary_mod(R.logical_or, dim, broadcast=True, dt="bool")
    graph = gen_case_graph_binary("or", dim, broadcast=True, dt="logical")
    verify_model_struct(graph, {}, expected)

    expected = get_unary_mod(lambda x: R.logical_or(x, R.const(False, "bool")), dim, dt="bool")
    graph = gen_case_graph_unary_attrs("or", dim, attrs=", false", dt="logical")
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_select(dim):
    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            cond = nn.Placeholder([4, 16] + [32] * (dim - 2), name="cond", dtype="bool")
            in1 = nn.Placeholder([4, 16] + [32] * (dim - 2), name="in1")
            in2 = nn.Placeholder([4, 16] + [32] * (dim - 2), name="in2")
            out = R.where(cond, in1, in2)
            out = bb.emit_output(out)
        bb.emit_func_output(out, [cond, in1, in2])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 3})
    graph = """
        version 1.0;

        graph G( cond, input1, input2 ) -> ( output )
        {{
            cond = external<logical>(shape = {shape});
            input1 = external<scalar>(shape = {shape});
            input2 = external<scalar>(shape = {shape});
            output = select(cond, input1, input2);
        }}
    """.format(
        shape=[4, 16] + [32] * (dim - 2)
    )
    verify_model_struct(graph, binding={}, expected=expected)

    for cond in [True, False]:
        expected = get_binary_mod(lambda a, b: R.where(R.const(cond), a, b), dim)
        graph = """
            version 1.0;

            graph G( input1, input2 ) -> ( output )
            {{
                input1 = external<scalar>(shape = {shape});
                input2 = external<scalar>(shape = {shape});
                output = select({cond}, input1, input2);
            }}
        """.format(
            shape=[4, 16] + [32] * (dim - 2), cond=str.lower(str(cond))
        )

        verify_model_struct(graph, binding={}, expected=expected)

    expected = get_binary_mod(
        lambda x, a: R.where(x, a, R.const(0.0)), dim, dt="bool", dt2="float32"
    )
    graph = """
        version 1.0;

        graph G( cond, input1 ) -> ( output )
        {{
            cond = external<logical>(shape = {shape});
            input1 = external<scalar>(shape = {shape});
            output = select(cond, input1, {inp});
        }}
    """.format(
        shape=[4, 16] + [32] * (dim - 2), inp="0.0"
    )
    verify_model_struct(graph, binding={}, expected=expected)

    expected = get_binary_mod(
        lambda x, b: R.where(x, R.const(0.0), b), dim, dt="bool", dt2="float32"
    )
    graph = """
            version 1.0;

            graph G( cond, input2 ) -> ( output )
            {{
                cond = external<logical>(shape = {shape});
                input2 = external<scalar>(shape = {shape});
                output = select(cond, {inp}, input2);
            }}
        """.format(
        shape=[4, 16] + [32] * (dim - 2), inp="0.0"
    )
    verify_model_struct(graph, binding={}, expected=expected)


@pytest.mark.parametrize("dim", dims)
def test_sqr(dim):
    expected = get_unary_mod(lambda x: R.power(x, R.const(2, "float32")), dim)
    graph = gen_case_graph_unary("sqr", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_sqrt(dim):
    expected = get_unary_mod(R.sqrt, dim)
    graph = gen_case_graph_unary("sqrt", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_rsqr(dim):
    expected = get_unary_mod(lambda x: R.power(x, R.const(-2, "float32")), dim)
    graph = gen_case_graph_unary("rsqr", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_rsqrt(dim):
    expected = get_unary_mod(R.rsqrt, dim)
    graph = gen_case_graph_unary("rsqrt", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_log2(dim):
    def method(inp):
        bb = relax.BlockBuilder.current()
        return bb.emit_te(topi.log2, inp)

    expected = get_unary_mod(method, dim)
    graph = gen_case_graph_unary("log2", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_min(dim):
    expected = get_binary_mod(R.minimum, dim)
    graph = gen_case_graph_binary("min", dim)
    verify_model_struct(graph, {}, expected)

    expected = get_binary_mod(R.minimum, dim, broadcast=True)
    graph = gen_case_graph_binary("min", dim, broadcast=True)
    verify_model_struct(graph, {}, expected)

    expected = get_unary_mod(lambda x: R.minimum(x, R.const(0.5, "float32")), dim)
    graph = gen_case_graph_unary_attrs("min", dim, attrs=", 0.5")
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_max(dim):
    expected = get_binary_mod(R.maximum, dim)
    graph = gen_case_graph_binary("max", dim)
    verify_model_struct(graph, {}, expected)

    expected = get_binary_mod(R.maximum, dim, broadcast=True)
    graph = gen_case_graph_binary("max", dim, broadcast=True)
    verify_model_struct(graph, {}, expected)

    expected = get_unary_mod(lambda x: R.maximum(x, R.const(0.5, "float32")), dim)
    graph = gen_case_graph_unary_attrs("max", dim, attrs=", 0.5")
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_clamp(dim):
    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16] + [32] * (dim - 2), name="inp1")
            inp2 = nn.Placeholder([4, 16] + [32] * (dim - 2), name="inp2")
            inp3 = nn.Placeholder([4, 16] + [32] * (dim - 2), name="inp3")
            lv = R.minimum(inp1, inp3)
            out = R.maximum(lv, inp2)
            out = bb.emit_output(out)
        bb.emit_func_output(out, [inp1, inp2, inp3])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 3})

    graph = """
        version 1.0;

        graph G( input1, input2, input3 ) -> ( output )
        {{
            input1 = external<scalar>(shape = {shape});
            input2 = external<scalar>(shape = {shape});
            input3 = external<scalar>(shape = {shape});
            output = clamp(input1, input2, input3);
        }}
    """.format(
        shape=[4, 16] + [32] * (dim - 2)
    )

    verify_model_struct(graph, {}, expected)

    def method(in1):
        return R.clip(in1, R.prim_value(T.float32(0.25)), R.prim_value(T.float32(0.75)))

    expected = get_unary_mod(method, dim)

    graph = gen_case_graph_unary_attrs("clamp", dim, attrs=", 0.25, 0.75")
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims_conv)
def test_conv(dim):
    if dim == 1:
        op = R.nn.conv1d
    elif dim == 2:
        op = R.nn.conv2d
    elif dim == 3:
        op = R.nn.conv3d

    kernel_size = [16, 8] + [3] * dim
    bias_size = [1, 16]

    # padded
    def method(inp, w, b):
        lv = op(inp, w, padding=[1, 1] * dim)
        lv1 = R.reshape(b, bias_size + [1] * dim)
        return R.add(lv, lv1)

    expected = get_tertiary_mod(method, dim)
    binding = {
        "inp2": np.ones(kernel_size, dtype="float32"),
        "inp3": np.ones(bias_size, dtype="float32"),
    }
    graph = gen_case_graph_unary_vars2("conv", dim, attrs=", padding = " + str([(1, 1)] * dim))
    verify_model_struct(graph, binding, expected)

    # strides
    def method(inp, w, b):
        lv = op(inp, w, padding=[0] * dim + [1] * dim, strides=[2] * dim)
        lv1 = R.reshape(b, bias_size + [1] * dim)
        return R.add(lv, lv1)

    expected = get_tertiary_mod(method, dim)
    binding = {
        "inp2": np.ones(kernel_size, dtype="float32"),
        "inp3": np.ones(bias_size, dtype="float32"),
    }
    graph = gen_case_graph_unary_vars2("conv", dim, attrs=", stride = " + str([2] * dim))
    verify_model_struct(graph, binding, expected)

    # groups 0
    g_kernel_size = [16, 1] + [3] * dim

    def method(inp, w, b):
        lv = op(inp, w, groups=16, padding=[1, 1] * dim)
        lv1 = R.reshape(b, bias_size + [1] * dim)
        return R.add(lv, lv1)

    expected = get_tertiary_mod(method, dim, shape1=[4, 16] + [32] * dim, shape2=g_kernel_size)
    binding = {
        "inp2": np.ones(g_kernel_size, dtype="float32"),
        "inp3": np.ones(bias_size, dtype="float32"),
    }
    graph = gen_case_graph_unary_vars2(
        "conv", dim, attrs=", groups = 0", shape1=[4, 16] + [32] * dim, shape2=g_kernel_size
    )
    verify_model_struct(graph, binding, expected)

    # nobias
    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 8] + [32] * dim, name="inp1")
            inp2 = nn.Placeholder(kernel_size, name="inp2")
            out = op(inp1, inp2, padding=[1, 1] * dim)
            out = bb.emit_output(out)
        bb.emit_func_output(out, [inp1, inp2])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    binding = {
        "inp2": np.ones(kernel_size, dtype="float32"),
    }
    graph = """
        version 1.0;

        graph G( input1 ) -> ( output )
        {{
            input1 = external<scalar>(shape = {shape1});
            weight = variable<scalar>(shape = {shape2}, label="weight");
            output = conv(input1, weight, 0.0);
        }}
    """.format(
        shape1=[4, 8] + [32] * dim, shape2=kernel_size
    )
    verify_model_struct(graph, binding, expected)


@pytest.mark.parametrize("dim", dims_conv)
def test_deconv(dim):
    if dim == 1:
        op = R.nn.conv1d_transpose
    elif dim == 2:
        op = R.nn.conv2d_transpose
    elif dim == 3:
        pytest.skip("3 dimensional conv_transpose is not supported in Relax")

    kernel_size = [8, 16] + [3] * dim
    bias_size = [1, 16]

    # padded
    def method(inp, w, b):
        lv = op(inp, w, padding=[1, 1] * dim)
        lv1 = R.reshape(b, bias_size + [1] * dim)
        return R.add(lv, lv1)

    expected = get_tertiary_mod(method, dim, shape2=kernel_size)
    binding = {
        "inp2": np.ones(kernel_size, dtype="float32"),
        "inp3": np.ones(bias_size, dtype="float32"),
    }
    graph = gen_case_graph_unary_vars2(
        "deconv", dim, attrs=", padding = " + str([(1, 1)] * dim), shape2=kernel_size
    )
    verify_model_struct(graph, binding, expected)

    # strides
    def method(inp, w, b):
        lv = op(inp, w, padding=[0] * dim + [1] * dim, strides=[2] * dim)
        lv1 = R.reshape(b, bias_size + [1] * dim)
        return R.add(lv, lv1)

    expected = get_tertiary_mod(method, dim, shape2=kernel_size)
    binding = {
        "inp2": np.ones(kernel_size, dtype="float32"),
        "inp3": np.ones(bias_size, dtype="float32"),
    }
    graph = gen_case_graph_unary_vars2(
        "deconv", dim, attrs=", stride = " + str([2] * dim), shape2=kernel_size
    )
    verify_model_struct(graph, binding, expected)

    # groups 0
    g_kernel_size = [16, 1] + [3] * dim

    def method(inp, w, b):
        lv = op(inp, w, groups=16, padding=[1, 1] * dim)
        lv1 = R.reshape(b, bias_size + [1] * dim)
        return R.add(lv, lv1)

    expected = get_tertiary_mod(method, dim, shape1=[4, 16] + [32] * dim, shape2=g_kernel_size)
    binding = {
        "inp2": np.ones(g_kernel_size, dtype="float32"),
        "inp3": np.ones(bias_size, dtype="float32"),
    }
    graph = gen_case_graph_unary_vars2(
        "deconv", dim, attrs=", groups = 0", shape1=[4, 16] + [32] * dim, shape2=g_kernel_size
    )
    verify_model_struct(graph, binding, expected)

    # nobias
    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 8] + [32] * dim, name="inp1")
            inp2 = nn.Placeholder(kernel_size, name="inp2")
            out = op(inp1, inp2, padding=[1, 1] * dim)
            out = bb.emit_output(out)
        bb.emit_func_output(out, [inp1, inp2])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    binding = {
        "inp2": np.ones(kernel_size, dtype="float32"),
    }
    graph = """
        version 1.0;

        graph G( input1 ) -> ( output )
        {{
            input1 = external<scalar>(shape = {shape1});
            weight = variable<scalar>(shape = {shape2}, label="weight");
            output = deconv(input1, weight, 0.0);
        }}
    """.format(
        shape1=[4, 8] + [32] * dim,
        shape2=kernel_size,
    )
    verify_model_struct(graph, binding, expected)


@pytest.mark.parametrize("dim", dims_conv)
def test_box(dim):
    if dim == 1:
        op = R.nn.conv1d
    elif dim == 2:
        op = R.nn.conv2d
    elif dim == 3:
        op = R.nn.conv3d

    kernel_size = [16, 1] + [3] * dim

    def method(inp):
        lv = R.ones(R.shape(kernel_size), dtype="float32")
        return op(inp, lv, strides=[2] * dim, groups=16, padding=[0] * dim + [1] * dim)

    expected = get_unary_mod(method, dim + 2)
    attrs = f", size = {[1, 1] + [3] * dim}, stride = {[1, 1] + [2] * dim}, border = 'constant'"
    graph = gen_case_graph_unary_attrs("box", dim + 2, attrs=attrs)
    verify_model_struct(graph, {}, expected)

    def method(inp):
        lv = R.ones(R.shape(kernel_size), dtype="float32")
        return op(inp, lv, strides=[1] * dim, groups=16, padding=[1] * dim + [1] * dim)

    expected = get_unary_mod(method, dim + 2)
    attrs = f", size = {[1, 1] + [3] * dim}, stride = {[1, 1] + [1] * dim}, border = 'constant'"
    graph = gen_case_graph_unary_attrs("box", dim + 2, attrs=attrs)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims_conv)
def test_debox(dim):
    if dim == 1:
        op = R.nn.conv1d_transpose
    elif dim == 2:
        op = R.nn.conv2d_transpose
    elif dim == 3:
        pytest.skip("3 dimensional conv_transpose is not supported in Relax")

    kernel_size = [16, 1] + [3] * dim

    def method(inp):
        lv = R.ones(R.shape(kernel_size), dtype="float32")
        return op(inp, lv, strides=[2] * dim, padding=[0] * dim + [1] * dim, groups=16)

    expected = get_unary_mod(method, dim + 2)
    attrs = f", size = {[1, 1] + [3] * dim}, stride = {[1, 1] + [2] * dim}, border = 'constant'"
    graph = gen_case_graph_unary_attrs("debox", dim + 2, attrs=attrs)
    verify_model_struct(graph, {}, expected)

    def method(inp):
        lv = R.ones(R.shape(kernel_size), dtype="float32")
        return op(inp, lv, strides=[1] * dim, padding=[1] * dim + [1] * dim, groups=16)

    expected = get_unary_mod(method, dim + 2)
    attrs = f", size = {[1, 1] + [3] * dim}, stride = {[1, 1] + [1] * dim}, border = 'constant'"
    graph = gen_case_graph_unary_attrs("debox", dim + 2, attrs=attrs)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims_conv)
def test_nearest_downsample(dim):
    if dim == 1:
        op = R.nn.conv1d
    elif dim == 2:
        op = R.nn.conv2d
    elif dim == 3:
        op = R.nn.conv3d

    def method(inp):
        lv = R.ones(R.shape([16, 1] + [1] * dim), dtype="float32")
        return op(inp, lv, strides=[2] * dim, padding=[0] * dim + [0] * dim, groups=16)

    expected = get_unary_mod(method, dim + 2)
    graph = gen_case_graph_unary_attrs(
        "nearest_downsample", dim + 2, attrs=f", factor = {[2] * dim}"
    )
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims_conv)
def test_area_downsample(dim):
    if dim == 1:
        op = R.nn.conv1d
    elif dim == 2:
        op = R.nn.conv2d
    elif dim == 3:
        op = R.nn.conv3d

    def method(inp):
        lv = R.full(
            R.shape([16, 1] + [2] * dim),
            R.const(1 / prod([2] * dim), dtype="float32"),
            dtype="float32",
        )
        return op(inp, lv, strides=[2] * dim, padding=[0] * dim + [0] * dim, groups=16)

    expected = get_unary_mod(method, dim + 2)
    graph = gen_case_graph_unary_attrs("area_downsample", dim + 2, attrs=f", factor = {[2] * dim}")
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims_conv)
def test_nearest_upsample(dim):
    if dim == 1:
        op = topi.image.resize1d
    elif dim == 2:
        op = topi.image.resize2d
    elif dim == 3:
        op = topi.image.resize3d

    def method(inp):
        bb = relax.BlockBuilder.current()
        return bb.emit_te(
            op, inp, [0, 0] * dim, [64] * dim, method="nearest_neighbor", rounding_method="round"
        )

    expected = get_unary_mod(method, dim + 2)
    graph = gen_case_graph_unary_attrs("nearest_upsample", dim + 2, attrs=f", factor = {[2] * dim}")
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims_conv)
def test_multilinear_upsample(dim):
    if dim == 1:
        rs = topi.image.resize1d
        cv = R.nn.conv1d_transpose
    elif dim == 2:
        rs = topi.image.resize2d
        cv = R.nn.conv2d_transpose
    elif dim == 3:
        pytest.skip("3 dimensional conv_transpose is not supported in Relax")

    def _upsample_weights_1d(fact, symm):
        if symm:
            _weights = [1 - (i + 0.5) / fact for i in range(fact)]
            _weights = list(reversed(_weights)) + _weights
        else:
            _weights = [1 - abs(i) / float(fact) for i in range(-fact + 1, fact)]
        return np.array(_weights, dtype="float32")

    def _upsample_weights_nd(fact, symm):
        _weights = [_upsample_weights_1d(f, symm) for f in fact]
        return reduce(np.multiply, np.ix_(*_weights))

    # symmetric constant
    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16] + [32] * dim, name="inp1")
            inp2 = nn.Placeholder([1, 1] + [4] * dim, name="inp2")
            w = R.tile(inp2, [16, 1] + [1] * dim)
            out = cv(inp1, w, strides=[2] * dim, padding=[1, 1] * dim, groups=16)
            out = bb.emit_output(out)
        bb.emit_func_output(out, [inp1, inp2])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    weights = _upsample_weights_nd([2] * dim, True)
    weights = np.reshape(weights, newshape=(1, 1) + weights.shape)
    binding = {"inp2": weights}
    graph = gen_case_graph_unary_attrs(
        "multilinear_upsample",
        dim + 2,
        attrs=f", factor = {[2] * dim}, method = 'symmetric', border = 'constant'",
    )
    verify_model_struct(graph, binding, expected)

    # symmetric replicate
    def method(inp):
        bb = relax.BlockBuilder.current()
        return bb.emit_te(
            rs,
            inp,
            [0, 0] * dim,
            [64] * dim,
            method="linear",
            coordinate_transformation_mode="half_pixel",
        )

    expected = get_unary_mod(method, dim + 2)
    graph = gen_case_graph_unary_attrs(
        "multilinear_upsample",
        dim + 2,
        attrs=f", factor = {[2] * dim}, method = 'symmetric', border = 'replicate'",
    )
    verify_model_struct(graph, {}, expected)

    # aligned constant
    def method(inp):
        bb = relax.BlockBuilder.current()
        return bb.emit_te(
            rs,
            inp,
            [0, 0] * dim,
            [64] * dim,
            method="linear",
            coordinate_transformation_mode="align_corners",
        )

    expected = get_unary_mod(method, dim + 2)
    graph = gen_case_graph_unary_attrs(
        "multilinear_upsample",
        dim + 2,
        attrs=f", factor = {[2] * dim}, method = 'aligned', border = 'constant'",
    )
    verify_model_struct(graph, {}, expected)

    # aligned replicate
    def method(inp):
        bb = relax.BlockBuilder.current()
        return bb.emit_te(
            rs,
            inp,
            [0, 0] * dim,
            [64] * dim,
            method="linear",
            coordinate_transformation_mode="align_corners",
        )

    expected = get_unary_mod(method, dim + 2)
    graph = gen_case_graph_unary_attrs(
        "multilinear_upsample",
        dim + 2,
        attrs=f", factor = {[2] * dim}, method = 'aligned', border = 'replicate'",
    )
    verify_model_struct(graph, {}, expected)

    # asymmetric constant
    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16] + [32] * dim, name="inp1")
            inp2 = nn.Placeholder([1, 1] + [3] * dim, name="inp2")
            w = R.tile(inp2, [16, 1] + [1] * dim)
            out = cv(inp1, w, strides=[2] * dim, padding=[1] * dim + [0] * dim, groups=16)
            out = bb.emit_output(out)
        bb.emit_func_output(out, [inp1, inp2])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    weights = _upsample_weights_nd([2] * dim, False)
    weights = np.reshape(weights, newshape=(1, 1) + weights.shape)
    binding = {"inp2": weights}
    graph = gen_case_graph_unary_attrs(
        "multilinear_upsample",
        dim + 2,
        attrs=f", factor = {[2] * dim}, method = 'asymmetric', border = 'constant'",
    )

    verify_model_struct(graph, binding, expected)

    # asymmetric replicate
    # Skip because Replicate - Edge mode padding is currently not supported in Relax


@pytest.mark.parametrize("dim", dims)
def test_sum_reduce(dim):
    expected = get_unary_mod(lambda x: R.sum(x, axis=[1], keepdims=True), dim)
    graph = gen_case_graph_unary_attrs("sum_reduce", dim, attrs=", axes = [1]")
    verify_model_struct(graph, {}, expected)

    expected = get_unary_mod(lambda x: R.sum(x, axis=list(range(2, dim)), keepdims=True), dim)
    graph = gen_case_graph_unary_attrs("sum_reduce", dim, attrs=f", axes = {list(range(2, dim))}")
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_max_reduce(dim):
    expected = get_unary_mod(lambda x: R.max(x, axis=[1], keepdims=True), dim)
    graph = gen_case_graph_unary_attrs("max_reduce", dim, attrs=", axes = [1]")
    verify_model_struct(graph, {}, expected)

    expected = get_unary_mod(lambda x: R.max(x, axis=list(range(2, dim)), keepdims=True), dim)
    graph = gen_case_graph_unary_attrs("max_reduce", dim, attrs=f", axes = {list(range(2, dim))}")
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_min_reduce(dim):
    expected = get_unary_mod(lambda x: R.min(x, axis=[1], keepdims=True), dim)
    graph = gen_case_graph_unary_attrs("min_reduce", dim, attrs=", axes = [1]")
    verify_model_struct(graph, {}, expected)

    expected = get_unary_mod(lambda x: R.min(x, axis=list(range(2, dim)), keepdims=True), dim)
    graph = gen_case_graph_unary_attrs("min_reduce", dim, attrs=f", axes = {list(range(2, dim))}")
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_argmax_reduce(dim):
    def method(in1):
        bb = relax.BlockBuilder.current()
        return bb.emit_te(topi.argmax, in1, axis=[1], keepdims=True)

    expected = get_unary_mod(method, dim)
    graph = gen_case_graph_unary_attrs("argmax_reduce", dim, attrs=", axes = [1]")
    verify_model_struct(graph, {}, expected)

    def method(in1):
        bb = relax.BlockBuilder.current()
        return bb.emit_te(topi.argmax, in1, axis=list(range(2, dim)), keepdims=True)

    expected = get_unary_mod(method, dim)
    graph = gen_case_graph_unary_attrs(
        "argmax_reduce", dim, attrs=f", axes = {list(range(2, dim))}"
    )
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_argmin_reduce(dim):
    def method(in1):
        bb = relax.BlockBuilder.current()
        return bb.emit_te(topi.argmin, in1, axis=[1], keepdims=True)

    expected = get_unary_mod(method, dim)
    graph = gen_case_graph_unary_attrs("argmin_reduce", dim, attrs=", axes = [1]")
    verify_model_struct(graph, {}, expected)

    def method(in1):
        bb = relax.BlockBuilder.current()
        return bb.emit_te(topi.argmin, in1, axis=list(range(2, dim)), keepdims=True)

    expected = get_unary_mod(method, dim)
    graph = gen_case_graph_unary_attrs(
        "argmin_reduce", dim, attrs=f", axes = {list(range(2, dim))}"
    )
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_all_reduce(dim):
    def method(in1):
        bb = relax.BlockBuilder.current()
        return bb.emit_te(topi.all, in1, axis=[1], keepdims=True)

    expected = get_unary_mod(method, dim, dt="bool")
    graph = gen_case_graph_unary_attrs("all_reduce", dim, attrs=", axes = [1]", dt="logical")
    verify_model_struct(graph, {}, expected)

    def method(in1):
        bb = relax.BlockBuilder.current()
        return bb.emit_te(topi.all, in1, axis=list(range(2, dim)), keepdims=True)

    expected = get_unary_mod(method, dim, dt="bool")
    graph = gen_case_graph_unary_attrs(
        "all_reduce", dim, attrs=f", axes = {list(range(2, dim))}", dt="logical"
    )
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_any_reduce(dim):
    def method(in1):
        bb = relax.BlockBuilder.current()
        return bb.emit_te(topi.any, in1, axis=[1], keepdims=True)

    expected = get_unary_mod(method, dim, dt="bool")
    graph = gen_case_graph_unary_attrs("any_reduce", dim, attrs=", axes = [1]", dt="logical")
    verify_model_struct(graph, {}, expected)

    def method(in1):
        bb = relax.BlockBuilder.current()
        return bb.emit_te(topi.any, in1, axis=list(range(2, dim)), keepdims=True)

    expected = get_unary_mod(method, dim, dt="bool")
    graph = gen_case_graph_unary_attrs(
        "any_reduce", dim, attrs=f", axes = {list(range(2, dim))}", dt="logical"
    )
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_mean_reduce(dim):
    expected = get_unary_mod(lambda x: R.mean(x, axis=[1], keepdims=True), dim)
    graph = gen_case_graph_unary_attrs("mean_reduce", dim, attrs=", axes = [1]")
    verify_model_struct(graph, {}, expected)

    expected = get_unary_mod(lambda x: R.mean(x, axis=list(range(2, dim)), keepdims=True), dim)
    graph = gen_case_graph_unary_attrs("mean_reduce", dim, attrs=f", axes = {list(range(2, dim))}")
    verify_model_struct(graph, {}, expected)


def test_reshape():
    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([2, 3, 3, 3, 2], name="inp1")
            out = R.reshape(inp1, R.shape([2, 3, 9, 2]))
            out = bb.emit_output(out)
        bb.emit_func_output(out, [inp1])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    graph = """
        version 1.0;

        graph G( input ) -> ( output )
        {
            input = external(shape = [2,3,3,3,2]);
            output = reshape(input, shape = [0,-1], axis_start = 1, axis_count = 3);
        }
        """
    verify_model_struct(graph, {}, expected)

    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16, 1, 1], name="inp1")
            out = R.reshape(inp1, R.shape([4, 16]))
            out = bb.emit_output(out)
        bb.emit_func_output(out, [inp1])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    graph = """
        version 1.0;

        graph G( input ) -> ( output )
        {
            input = external(shape = [4,16,1,1]);
            output = reshape(input, shape = [4,16]);
        }
        """
    verify_model_struct(graph, {}, expected)

    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16, 32, 32], name="inp1")
            out = R.reshape(inp1, R.shape([4, 16384]))
            out = bb.emit_output(out)
        bb.emit_func_output(out, [inp1])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    graph = """
        version 1.0;

        graph G( input ) -> ( output )
        {
            input = external(shape = [4,16,32,32]);
            output = reshape(input, shape = [4,16384]);
        }
        """
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_squeeze(dim):
    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder(
                [
                    4,
                    16,
                ]
                + [1] * (dim - 2),
                name="inp1",
            )
            out = R.squeeze(inp1, axis=list(range(2, dim)))
            out = bb.emit_output(out)
        bb.emit_func_output(out, [inp1])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})

    graph = """
        version 1.0;

        graph G( input ) -> ( output )
        {{
            input = external(shape = {dim});
            output = squeeze(input, axes = {axes});
        }}
        """.format(
        dim=[4, 16] + [1] * (dim - 2), axes=list(range(2, dim))
    )

    verify_model_struct(graph, {}, expected)


def test_unsqueeze():
    def method(in1):
        lv = R.expand_dims(in1, axis=[2])
        return R.expand_dims(lv, axis=[3])

    expected = get_unary_mod(method, 2)
    graph = gen_case_graph_unary_attrs("unsqueeze", 2, attrs=", axes = [2, 3]")
    verify_model_struct(graph, {}, expected)


def test_transpose():
    expected = get_unary_mod(lambda x: R.permute_dims(x, axes=[0, 3, 1, 2]), 4)
    graph = gen_case_graph_unary_attrs("transpose", 4, attrs=", axes = [0, 3, 1, 2]")
    verify_model_struct(graph, {}, expected)


def test_split():
    # channel
    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16, 32, 32], name="inp1")
            lv = R.split(inp1, indices_or_sections=[8], axis=1)
            o1, o2 = lv[0], lv[1]
            out = R.tuple(o1, o2)
            out = bb.emit_output(out)
        bb.emit_func_output(out, [inp1])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    graph = """
        version 1.0;

        graph G( input ) -> ( output1, output2 )
        {
            input = external(shape = [4,16,32,32]);
            [output1, output2] = split(input, axis = 1, ratios = [1,1]);
        }
        """
    verify_model_struct(graph, {}, expected)

    # unbalanced
    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 32, 3], name="inp1")
            lv = R.split(inp1, indices_or_sections=[12, 16], axis=1)
            o1, o2, o3 = lv[0], lv[1], lv[2]
            out = R.tuple(o1, o2, o3)
            out = bb.emit_output(out)
        bb.emit_func_output(out, [inp1])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    graph = """
        version 1.0;

        graph G( input ) -> ( output1, output2, output3 )
        {
            input = external(shape = [4,32,3]);
            [output1, output2, output3] = split(input, axis = 1, ratios = [3,1,4]);
        }
        """
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_concat(dim):
    expected = get_binary_mod(lambda x, y: R.concat((x, y), axis=1), dim)
    graph = """
        version 1.0;

        graph G( input1, input2 ) -> ( output )
        {{
            input1 = external(shape = {shape});
            input2 = external(shape = {shape});
            output = concat([input1, input2], axis = 1);
        }}
        """.format(
        shape=[4, 16] + [32] * (dim - 2)
    )
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_stack(dim):
    def method(in1, in2):
        lv = R.expand_dims(in1, axis=[1])
        lv1 = R.expand_dims(in2, axis=[1])
        return R.concat((lv, lv1), axis=1)

    expected = get_binary_mod(method, dim)
    graph = """
        version 1.0;

        graph G( input1, input2 ) -> ( output )
        {{
            input1 = external(shape = {shape});
            input2 = external(shape = {shape});
            output = stack([input1, input2], axis = 1);
        }}
        """.format(
        shape=[4, 16] + [32] * (dim - 2)
    )
    verify_model_struct(graph, {}, expected)


def test_unstack():
    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 3, 16], name="inp1")
            lv = R.split(inp1, indices_or_sections=[1, 2], axis=1)
            lv1, lv2, lv3 = lv[0], lv[1], lv[2]
            lv1 = R.squeeze(lv1, axis=1)
            lv2 = R.squeeze(lv2, axis=1)
            lv3 = R.squeeze(lv3, axis=1)
            out = R.tuple(lv1, lv2, lv3)
            out = bb.emit_output(out)
        bb.emit_func_output(out, [inp1])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    graph = """
        version 1.0;

        graph G( input ) -> ( output1, output2, output3 )
        {
            input = external(shape = [4,3,16]);
            [output1, output2, output3] = unstack(input, axis = 1);
        }
        """
    verify_model_struct(graph, {}, expected)


def test_slice():
    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16, 32, 32], name="inp1")
            out = R.strided_slice(
                inp1,
                (R.prim_value(2), R.prim_value(3)),
                (R.prim_value(1), R.prim_value(2)),
                (R.prim_value(-1), R.prim_value(-2)),
                (R.prim_value(1), R.prim_value(1)),
            )
            out = bb.emit_output(out)
        bb.emit_func_output(out, [inp1])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})

    graph = gen_case_graph_unary_attrs(
        "slice", 4, attrs=", axes = [2,3], begin = [1,2], end = [-1,-2]"
    )
    verify_model_struct(graph, {}, expected)

    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16, 32, 32], name="inp1")
            out = R.strided_slice(
                inp1,
                (R.prim_value(1), R.prim_value(2), R.prim_value(3)),
                (R.prim_value(5), R.prim_value(16), R.prim_value(2)),
                (R.prim_value(1), R.prim_value(4), R.prim_value(-1)),
                (R.prim_value(-1), R.prim_value(-1), R.prim_value(1)),
            )
            out = bb.emit_output(out)
        bb.emit_func_output(out, [inp1])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})

    graph = gen_case_graph_unary_attrs(
        "slice", 4, attrs=", axes = [1,2,3], begin = [5,16,2], end = [1,4,-1], stride = [-1,-1,1]"
    )
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims_conv)
def test_pad(dim):
    for before, after in [([0], [1]), ([1], [0]), ([1], [1])]:
        # reflect
        def method(in1):
            bb = relax.BlockBuilder.current()
            return bb.emit_te(
                topi.nn.mirror_pad, in1, [0, 0] + before * dim, [0, 0] + after * dim, "REFLECT"
            )

        expected = get_unary_mod(method, dim + 2)
        graph = gen_case_graph_unary_attrs(
            "pad",
            dim + 2,
            attrs=", padding = {pad}, border = 'reflect'".format(
                pad=list(zip([0, 0] + before * dim, [0, 0] + after * dim))
            ),
        )
        verify_model_struct(graph, {}, expected)

        # constant
        def method(in1):
            return R.nn.pad(
                in1,
                pad_value=R.const(0, "float32"),
                pad_width=[0, 0, 0, 0]
                + [item for p in zip(before * dim, after * dim) for item in p],
            )

        expected = get_unary_mod(method, dim + 2)
        graph = gen_case_graph_unary_attrs(
            "pad",
            dim + 2,
            attrs=", padding = {pad}, border = 'constant'".format(
                pad=list(zip([0, 0] + before * dim, [0, 0] + after * dim))
            ),
        )
        verify_model_struct(graph, {}, expected)

        # Replicate - Edge mode is currently not supported in TVM relax


@pytest.mark.parametrize("dim", dims)
def test_tile(dim):
    # spatial
    expected = get_unary_mod(lambda x: R.tile(x, [1, 1] + [3] * (dim - 2)), dim)
    graph = gen_case_graph_unary_attrs("tile", dim, attrs=f", repeats = {[1, 1] + [3] * (dim - 2)}")
    verify_model_struct(graph, {}, expected)

    # channel
    expected = get_unary_mod(lambda x: R.tile(x, [1, 16] + [1] * (dim - 2)), dim)
    graph = gen_case_graph_unary_attrs(
        "tile", dim, attrs=f", repeats = {[1, 16] + [1] * (dim - 2)}"
    )
    verify_model_struct(graph, {}, expected)

    # batch
    expected = get_unary_mod(lambda x: R.tile(x, [16, 1] + [1] * (dim - 2)), dim)
    graph = gen_case_graph_unary_attrs(
        "tile", dim, attrs=f", repeats = {[16, 1] + [1] * (dim - 2)}"
    )
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", [2, 4])
def test_matmul(dim):
    for trans_a, trans_b in [(False, False), (True, False), (False, True), (True, True)]:
        if dim == 2:
            a_shape = [4, 16]
            b_shape = [16, 4]
        else:
            a_shape = [4, 16] + [32] * (dim - 2)
            b_shape = [4, 16] + [32] * (dim - 2)

        axes = list(range(dim - 2))
        axes.append(dim - 1)
        axes.append(dim - 2)

        if trans_a:
            a_shape = [a_shape[i] for i in axes]
        if trans_b:
            b_shape = [b_shape[i] for i in axes]

        bb = relax.BlockBuilder()
        with bb.function("main"):
            with bb.dataflow():
                inp1 = gin1 = nn.Placeholder(a_shape, name="inp1")
                inp2 = gin2 = nn.Placeholder(b_shape, name="inp2")
                if trans_a:
                    inp1 = R.permute_dims(inp1, axes=axes)
                if trans_b:
                    inp2 = R.permute_dims(inp2, axes=axes)

                out = R.matmul(inp1, inp2)
                out = bb.emit_output(out)
            bb.emit_func_output(out, [gin1, gin2])
        expected = bb.get()
        expected["main"] = expected["main"].with_attrs({"num_input": 2})

        graph = """
            version 1.0;

            graph G( input1, input2 ) -> ( output )
            {{
                input1 = external(shape = {shape1});
                input2 = external(shape = {shape2});
                output = matmul(input1, input2, transposeA={t_a}, transposeB={t_b});
            }}
            """.format(
            shape1=a_shape, shape2=b_shape, t_a=str(trans_a).lower(), t_b=str(trans_b).lower()
        )

        verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_sigmoid(dim):
    expected = get_unary_mod(R.sigmoid, dim)
    graph = gen_case_graph_unary("sigmoid", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_relu(dim):
    expected = get_unary_mod(R.nn.relu, dim)
    graph = gen_case_graph_unary("relu", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_prelu(dim):
    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([16, 16] + [32] * (dim - 2), name="inp1")
            inp2 = nn.Placeholder([16], name="inp2")
            lv = R.less(inp1, R.const(0, "float32"))
            lv1 = R.expand_dims(inp2, axis=[0] + list(range(2, dim)))
            lv2 = R.multiply(lv1, inp1)
            out = R.where(lv, lv2, inp1)
            out = bb.emit_output(out)
        bb.emit_func_output(out, [inp1, inp2])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 2})
    graph = """
        version 1.0;

        graph G( input1, input2 ) -> ( output )
        {{
            input1 = external(shape = {dim});
            input2 = external(shape = [16]);
            output = prelu(input1, input2);
        }}
        """.format(
        dim=[16, 16] + [32] * (dim - 2)
    )
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_leaky_relu(dim):
    alpha = 0.5
    expected = get_unary_mod(lambda x: R.nn.leakyrelu(x, alpha), dim)
    graph = gen_case_graph_unary_attrs("leaky_relu", dim, attrs=f", alpha = {alpha}")
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_elu(dim):
    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16] + [32] * (dim - 2), name="inp1")
            lv = R.exp(inp1)
            lv1 = R.less(inp1, R.const(0, "float32"))
            lv2 = R.subtract(lv, R.const(1, "float32"))
            lv2 = bb.normalize(lv2)
            lv3 = R.multiply(R.const(1, "float32"), lv2)
            out = R.where(lv1, lv3, inp1)
            out = bb.emit_output(out)
        bb.emit_func_output(out, [inp1])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    graph = gen_case_graph_unary("elu", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_selu(dim):
    _lambda = 1.050
    _alpha = 1.673

    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16] + [32] * (dim - 2), name="inp1")
            lv = R.exp(inp1)
            lv1 = R.less(inp1, R.const(0, "float32"))
            lv2 = R.subtract(lv, R.const(1, "float32"))
            lv2 = bb.normalize(lv2)
            lv3 = R.multiply(R.const(_alpha, "float32"), lv2)
            lv4 = R.where(lv1, lv3, inp1)
            lv4 = bb.normalize(lv4)
            gv = R.multiply(R.const(_lambda, "float32"), lv4)
            out = bb.emit_output(gv)
        bb.emit_func_output(out, [inp1])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    graph = gen_case_graph_unary_attrs("selu", dim, attrs=f", alpha = {_alpha}, lambda = {_lambda}")
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_gelu(dim):
    expected = get_unary_mod(R.nn.gelu, dim)
    graph = gen_case_graph_unary("gelu", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_silu(dim):
    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16] + [32] * (dim - 2), name="inp1")
            lv = R.sigmoid(inp1)
            gv = R.multiply(inp1, lv)
            out = bb.emit_output(gv)
        bb.emit_func_output(out, [inp1])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    graph = gen_case_graph_unary("silu", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_softmax(dim):
    expected = get_unary_mod(lambda x: R.nn.softmax(x, axis=1), dim)
    graph = gen_case_graph_unary("softmax", dim)
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_softplus(dim):
    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16] + [32] * (dim - 2), name="inp1")
            lv = R.exp(inp1)
            lv1 = R.add(lv, R.const(1, "float32"))
            gv = R.log(lv1)
            out = bb.emit_output(gv)
        bb.emit_func_output(out, [inp1])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    graph = gen_case_graph_unary("softplus", dim)
    verify_model_struct(graph, {}, expected)


def test_linear():
    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 8], name="inp1")
            weights = nn.Placeholder([16, 8], name="weights")
            bias = nn.Placeholder([1, 16], name="bias")
            lv = R.permute_dims(weights, axes=[1, 0])
            lv1 = R.matmul(inp1, lv)
            lv2 = R.reshape(bias, R.shape([1, 16]))
            gv = R.add(lv1, lv2)
            out = bb.emit_output(gv)
        bb.emit_func_output(out, [inp1, weights, bias])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    binding = {
        "weights": np.ones([16, 8], dtype="float32"),
        "bias": np.ones([1, 16], dtype="float32"),
    }
    graph = gen_case_graph_unary_vars2("linear", 0, "")
    verify_model_struct(graph, binding, expected)

    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16], name="inp1")
            weights = nn.Placeholder([32, 16], name="weights")
            lv = R.permute_dims(weights, axes=[1, 0])
            gv = R.matmul(inp1, lv)
            out = bb.emit_output(gv)
        bb.emit_func_output(out, [inp1, weights])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 2})
    graph = """
        version 1.0;

        graph G( input1, weights ) -> ( output )
        {
            input1 = external(shape = [4,16]);
            weights = external(shape = [32,16]);
            output = linear(input1, weights);
        }
        """
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims_conv)
def test_separable_conv(dim):
    if dim == 1:
        op = R.nn.conv1d
    elif dim == 2:
        op = R.nn.conv2d
    else:
        op = R.nn.conv3d

    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 8] + [32] * dim, name="inp1")
            plane_filter = nn.Placeholder([8, 1] + [3] * dim, name="plane_filter")
            point_filter = nn.Placeholder([16, 8] + [1] * dim, name="point_filter")
            bias = nn.Placeholder([1, 16], name="bias")
            lv = op(inp1, plane_filter, padding=[1] * dim, groups=8)
            lv1 = op(lv, point_filter, groups=1)
            lv2 = R.reshape(bias, R.shape([1, 16] + [1] * dim))
            gv = R.add(lv1, lv2)
            out = bb.emit_output(gv)
        bb.emit_func_output(out, [inp1, plane_filter, point_filter, bias])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    binding = {
        "plane_filter": np.ones([8, 1] + [3] * dim, dtype="float32"),
        "point_filter": np.ones([16, 8] + [1] * dim, dtype="float32"),
        "bias": np.ones([1, 16], dtype="float32"),
    }
    graph = f"""
        version 1.0;

        graph G( input ) -> ( output )
        {{
            input = external(shape = {[4, 8] + [32] * dim});
            plane_filter = variable(shape = {[8, 1] + [3] * dim}, label = 'plane_filter');
            point_filter = variable(shape = {[16, 8] + [1] * dim}, label = 'point_filter');
            bias = variable(shape = [1,16], label = 'bias');
            output = separable_conv(input, plane_filter, point_filter, bias);
        }}
        """
    verify_model_struct(graph, binding, expected)

    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 8] + [32] * dim, name="inp1")
            plane_filter = nn.Placeholder([8, 1] + [3] * dim, name="plane_filter")
            point_filter = nn.Placeholder([16, 8] + [1] * dim, name="point_filter")
            lv = op(inp1, plane_filter, padding=[1] * dim, groups=8)
            gv = op(lv, point_filter, groups=1)
            out = bb.emit_output(gv)
        bb.emit_func_output(out, [inp1, plane_filter, point_filter])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    binding = {
        "plane_filter": np.ones([8, 1] + [3] * dim, dtype="float32"),
        "point_filter": np.ones([16, 8] + [1] * dim, dtype="float32"),
    }
    graph = gen_case_graph_unary_vars2(
        "separable_conv", dim, "", shape2=[8, 1] + [3] * dim, shape3=[16, 8] + [1] * dim
    )
    verify_model_struct(graph, binding, expected)


@pytest.mark.parametrize("dim", dims_conv)
def test_separable_deconv(dim):
    if dim == 1:
        op = R.nn.conv1d_transpose
    elif dim == 2:
        op = R.nn.conv2d_transpose
    else:
        pytest.skip("conv3d_transpose is not supported in TVM relax")

    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16] + [32] * dim, name="inp1")
            plane_filter = nn.Placeholder([8, 1] + [3] * dim, name="plane_filter")
            point_filter = nn.Placeholder([16, 8] + [1] * dim, name="point_filter")
            bias = nn.Placeholder([1, 8], name="bias")
            lv = op(inp1, point_filter, groups=1)
            lv1 = op(lv, plane_filter, padding=[1] * dim, groups=8)
            lv2 = R.reshape(bias, R.shape([1, 8] + [1] * dim))
            gv = R.add(lv1, lv2)
            out = bb.emit_output(gv)
        bb.emit_func_output(out, [inp1, plane_filter, point_filter, bias])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    binding = {
        "plane_filter": np.ones([8, 1] + [3] * dim, dtype="float32"),
        "point_filter": np.ones([16, 8] + [1] * dim, dtype="float32"),
        "bias": np.ones([1, 8], dtype="float32"),
    }
    graph = f"""
        version 1.0;

        graph G( input ) -> ( output )
        {{
            input = external(shape = {[4, 16] + [32] * dim});
            plane_filter = variable(shape = {[8, 1] + [3] * dim}, label = 'plane_filter');
            point_filter = variable(shape = {[16, 8] + [1] * dim}, label = 'point_filter');
            bias = variable(shape = [1,8], label = 'bias');
            output = separable_deconv(input, plane_filter, point_filter, bias);
        }}
        """
    verify_model_struct(graph, binding, expected)

    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16] + [32] * dim, name="inp1")
            plane_filter = nn.Placeholder([8, 1] + [3] * dim, name="plane_filter")
            point_filter = nn.Placeholder([16, 8] + [1] * dim, name="point_filter")
            lv = op(inp1, point_filter, groups=1)
            gv = op(lv, plane_filter, padding=[1] * dim, groups=8)
            out = bb.emit_output(gv)
        bb.emit_func_output(out, [inp1, plane_filter, point_filter])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    binding = {
        "plane_filter": np.ones([8, 1] + [3] * dim, dtype="float32"),
        "point_filter": np.ones([16, 8] + [1] * dim, dtype="float32"),
    }
    graph = gen_case_graph_unary_vars2(
        "separable_deconv",
        dim,
        "",
        shape1=[4, 16] + [32] * dim,
        shape2=[8, 1] + [3] * dim,
        shape3=[16, 8] + [1] * dim,
    )
    verify_model_struct(graph, binding, expected)


@pytest.mark.parametrize("dim", dims_conv)
def test_max_pool(dim):
    if dim == 1:
        op = R.nn.max_pool1d
    elif dim == 2:
        op = R.nn.max_pool2d
    else:
        op = R.nn.max_pool3d

    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16] + [32] * dim, name="inp1")
            lv = op(inp1, pool_size=[3] * dim, strides=[2] * dim, padding=[1, 1] * dim)
            out = bb.emit_output(lv)
        bb.emit_func_output(out, [inp1])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    graph = gen_case_graph_unary_attrs(
        "max_pool",
        dim + 2,
        attrs=f", size = {[1, 1] + [3] * dim},"
        f" stride = {[1, 1] + [2] * dim},"
        f" padding = {([(0, 0)] * 2) + [(1, 1)] * dim},"
        f" border = 'ignore'",
    )
    verify_model_struct(graph, {}, expected)

    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16] + [32] * dim, name="inp1")
            lv = R.nn.pad(
                inp1,
                pad_value=R.const(0, "float32"),
                pad_width=[0, 0, 0, 0] + [4, 4] * dim,
                pad_mode="constant",
            )
            lv1 = op(lv, pool_size=[3] * dim, strides=[1] * dim)
            out = bb.emit_output(lv1)
        bb.emit_func_output(out, [inp1])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    graph = gen_case_graph_unary_attrs(
        "max_pool",
        dim + 2,
        attrs=f", size = {[1, 1] + [3] * dim},"
        f" stride = {[1, 1] + [1] * dim},"
        f" padding = {([(0, 0)] * 2) + [(4, 4)] * dim},"
        f" border = 'constant'",
    )
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims_conv)
def test_avg_pool(dim):
    if dim == 1:
        op = R.nn.avg_pool1d
    elif dim == 2:
        op = R.nn.avg_pool2d
    else:
        op = R.nn.avg_pool3d

    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16] + [32] * dim, name="inp1")
            lv = op(inp1, pool_size=[3] * dim, strides=[2] * dim, padding=[1, 1] * dim)
            out = bb.emit_output(lv)
        bb.emit_func_output(out, [inp1])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    graph = gen_case_graph_unary_attrs(
        "avg_pool",
        dim + 2,
        attrs=f", size = {[1, 1] + [3] * dim},"
        f" stride = {[1, 1] + [2] * dim},"
        f" padding = {([(0, 0)] * 2) + [(1, 1)] * dim},"
        f" border = 'ignore'",
    )
    verify_model_struct(graph, {}, expected)

    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16] + [32] * dim, name="inp1")
            lv = op(
                inp1,
                pool_size=[3] * dim,
                strides=[1] * dim,
                padding=[1, 1] * dim,
                count_include_pad=True,
            )
            out = bb.emit_output(lv)
        bb.emit_func_output(out, [inp1])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    graph = gen_case_graph_unary_attrs(
        "avg_pool",
        dim + 2,
        attrs=f", size = {[1, 1] + [3] * dim},"
        f" stride = {[1, 1] + [1] * dim},"
        f" padding = {([(0, 0)] * 2) + [(1, 1)] * dim},"
        f" border = 'constant'",
    )
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims_conv)
def test_rms_pool(dim):
    if dim == 1:
        avg_pool = R.nn.avg_pool1d
    elif dim == 2:
        avg_pool = R.nn.avg_pool2d
    else:
        avg_pool = R.nn.avg_pool3d

    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16] + [32] * dim, name="inp1")
            lv = R.power(inp1, R.const(2, "float32"))
            lv1 = avg_pool(lv, pool_size=[3] * dim, strides=[2] * dim, padding=[1, 1] * dim)
            gv = R.sqrt(lv1)
            out = bb.emit_output(gv)
        bb.emit_func_output(out, [inp1])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    graph = gen_case_graph_unary_attrs(
        "rms_pool",
        dim + 2,
        attrs=f", size = {[1, 1] + [3] * dim},"
        f" stride = {[1, 1] + [2] * dim},"
        f" padding = {([(0, 0)] * 2) + [(1, 1)] * dim},"
        f" border = 'ignore'",
    )
    verify_model_struct(graph, {}, expected)

    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16] + [32] * dim, name="inp1")
            lv = R.power(inp1, R.const(2, "float32"))
            lv1 = avg_pool(lv, pool_size=[3] * dim, strides=[2] * dim, padding=[1, 1] * dim)
            gv = R.sqrt(lv1)
            out = bb.emit_output(gv)
        bb.emit_func_output(out, [inp1])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    graph = gen_case_graph_unary_attrs(
        "rms_pool",
        dim + 2,
        attrs=f", size = {[1, 1] + [3] * dim},"
        f" stride = {[1, 1] + [2] * dim},"
        f" padding = {([(0, 0)] * 2) + [(1, 1)] * dim},"
        f" border = 'ignore'",
    )
    verify_model_struct(graph, {}, expected)


def test_local_response_normalization():
    def method(in1):
        bb = relax.BlockBuilder.current()
        return bb.emit_te(topi.nn.lrn, in1, 5, 1, 1.0, 0.5, 1.0)

    expected = get_unary_mod(method, 4)
    graph = gen_case_graph_unary_attrs(
        "local_response_normalization", 4, attrs=", size = [1, 5, 1, 1]"
    )
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims_conv)
def test_local_mean_normalization(dim):
    if dim == 1:
        conv = R.nn.conv1d
    elif dim == 2:
        conv = R.nn.conv2d
    else:
        conv = R.nn.conv3d

    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16] + [32] * dim, name="inp1")
            lv = R.full(
                R.shape([16, 1] + [3] * dim), R.const(1 / (3**dim), "float32"), dtype="float32"
            )
            lv1 = conv(inp1, lv, padding=[1, 1] * dim, groups=16)
            gv = R.subtract(inp1, lv1)
            out = bb.emit_output(gv)
        bb.emit_func_output(out, [inp1])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    graph = gen_case_graph_unary_attrs(
        "local_mean_normalization", dim + 2, attrs=f", size = {[1, 1] + [3] * dim}"
    )
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims_conv)
def test_local_variance_normalization(dim):
    if dim == 1:
        conv = R.nn.conv1d
    elif dim == 2:
        conv = R.nn.conv2d
    else:
        conv = R.nn.conv3d

    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16] + [32] * dim, name="inp1")
            lv = R.power(inp1, R.const(2, "float32"))
            lv1 = R.full(
                R.shape([16, 1] + [3] * dim), R.const(1 / (3**dim), "float32"), dtype="float32"
            )
            lv2 = conv(lv, lv1, padding=[1, 1] * dim, groups=16)
            lv3 = R.sqrt(lv2)
            lv4 = R.add(lv3, R.const(0, "float32"))
            lv5 = R.maximum(lv4, R.const(0, "float32"))
            gv = R.divide(inp1, lv5)
            out = bb.emit_output(gv)
        bb.emit_func_output(out, [inp1])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    graph = gen_case_graph_unary_attrs(
        "local_variance_normalization", dim + 2, attrs=f", size = {[1, 1] + [3] * dim}"
    )
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims_conv)
def test_local_contrast_normalization(dim):
    if dim == 1:
        conv = R.nn.conv1d
        avg_pool = R.nn.avg_pool1d
    elif dim == 2:
        conv = R.nn.conv2d
        avg_pool = R.nn.avg_pool2d
    else:
        conv = R.nn.conv3d
        avg_pool = R.nn.avg_pool3d

    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16] + [32] * dim, name="inp1")
            lv = R.full(
                R.shape([16, 1] + [3] * dim), R.const(1 / (3**dim), "float32"), dtype="float32"
            )
            lv1 = conv(inp1, lv, padding=[1, 1] * dim, groups=16)
            lv2 = R.subtract(inp1, lv1)
            lv3 = R.power(lv2, R.const(2, "float32"))
            lv4 = R.permute_dims(lv3, axes=[1, 0] + list(range(2, dim + 2)))
            lv5 = avg_pool(
                lv4,
                pool_size=[3] * dim,
                strides=[1] * dim,
                padding=[1, 1] * dim,
                count_include_pad=True,
            )
            lv6 = R.permute_dims(lv5, axes=[1, 0] + list(range(2, dim + 2)))
            lv7 = R.sqrt(lv6)
            lv8 = R.add(lv7, R.const(0, "float32"))
            lv9 = R.maximum(lv8, R.const(0, "float32"))
            gv = R.divide(lv2, lv9)
            out = bb.emit_output(gv)
        bb.emit_func_output(out, [inp1])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    graph = gen_case_graph_unary_attrs(
        "local_contrast_normalization", dim + 2, attrs=f", size = {[1, 1] + [3] * dim}"
    )
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_l1_normalization(dim):
    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16] + [32] * (dim - 2), name="inp1")
            lv = R.abs(inp1)
            lv1 = R.sum(lv, axis=[1], keepdims=True)
            lv2 = R.add(lv1, R.const(0, "float32"))
            lv3 = R.maximum(lv2, R.const(0, "float32"))
            gv = R.divide(inp1, lv3)
            out = bb.emit_output(gv)
        bb.emit_func_output(out, [inp1])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    graph = gen_case_graph_unary_attrs("l1_normalization", dim, attrs=", axes=[1]")
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_l2_normalization(dim):
    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16] + [32] * (dim - 2), name="inp1")
            lv = R.power(inp1, R.const(2, "float32"))
            lv1 = R.sum(lv, axis=[1], keepdims=True)
            lv2 = R.sqrt(lv1)
            lv3 = R.add(lv2, R.const(0, "float32"))
            lv4 = R.maximum(lv3, R.const(0, "float32"))
            gv = R.divide(inp1, lv4)
            out = bb.emit_output(gv)
        bb.emit_func_output(out, [inp1])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 1})
    graph = gen_case_graph_unary_attrs("l2_normalization", dim, attrs=", axes=[1]")
    verify_model_struct(graph, {}, expected)


@pytest.mark.parametrize("dim", dims)
def test_batch_norm(dim):
    bb = relax.BlockBuilder()
    with bb.function("main"):
        with bb.dataflow():
            inp1 = nn.Placeholder([4, 16] + [32] * dim, name="inp1")
            mean = nn.Placeholder([1, 16], name="mean")
            variance = nn.Placeholder([1, 16], name="variance")
            offset = nn.Placeholder([1, 16], name="offset")
            scale = nn.Placeholder([1, 16], name="scale")
            lv = R.squeeze(scale, axis=[0])
            lv = bb.normalize(lv)
            lv1 = R.squeeze(offset, axis=[0])
            lv1 = bb.normalize(lv1)
            lv2 = R.squeeze(mean, axis=[0])
            lv2 = bb.normalize(lv2)
            lv3 = R.squeeze(variance, axis=[0])
            lv3 = bb.normalize(lv3)
            lv4 = bb.emit_te(topi.nn.batch_norm, inp1, lv, lv1, lv2, lv3, 1, 0.5)
            gv = lv4[0]
            out = bb.emit_output(gv)
        bb.emit_func_output(out, [inp1, mean, variance, offset, scale])
    expected = bb.get()
    expected["main"] = expected["main"].with_attrs({"num_input": 5})
    graph = f"""
        version 1.0;

        graph G( input, mean, variance, offset, scale ) -> ( output )
        {{
            input = external(shape = {[4, 16] + [32] * dim});
            mean = external(shape = [1,16]);
            variance = external(shape = [1,16]);
            offset = external(shape = [1,16]);
            scale = external(shape = [1,16]);
            output = batch_normalization(input, mean, variance, offset, scale, epsilon=0.5);
        }}
        """

    verify_model_struct(graph, {}, expected)
