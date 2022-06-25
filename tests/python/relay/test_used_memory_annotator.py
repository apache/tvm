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
# pylint: disable=invalid-name

"""
Testing for the pass that annotates used memory for each primitive
Relay function.
"""

import pytest

import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprVisitor


def AnnotateUsedMemory():
    return relay.transform._ffi_api.AnnotateUsedMemory()


class CheckUsedMemoryAnnotation(ExprVisitor):
    """
    Check that the annotations on each function in the graph match
    what is expected.
    """

    def __init__(self, expected_annotations, expected_io_annotation):
        self.expected_annotations = expected_annotations
        self.expected_io_annotation = expected_io_annotation
        super().__init__()

    def visit_function(self, fn):
        if "Primitive" in fn.attrs:
            assert (
                "used_memory" in fn.attrs
            ), "Primitive function does not have used_memory annotation."

            assert len(self.expected_annotations) > 0, "Not all expected annotations were compared"

            expected_mem = self.expected_annotations.pop(0)
            actual_mem = [int(x) for x in fn.attrs["used_memory"]]
            assert expected_mem == actual_mem, (
                f"Expected used memory annotation {expected_mem} "
                f"did not match actual annotation {actual_mem}"
            )
        super().visit_function(fn)

    def __call__(self, fn):
        assert (
            fn.attrs["io_used_memory"] == self.expected_io_annotation
        ), "Expected IO annotation did not match."
        self.visit(fn.body)


def _check_used_memory_annotations(mod, expected_annotations, expected_io_annotation):
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.ToANormalForm()(mod)
    mod = relay.transform.InferType()(mod)
    mod = AnnotateUsedMemory()(mod)

    CheckUsedMemoryAnnotation(expected_annotations, expected_io_annotation)(mod["main"])


def _create_primitive_function(expr):
    func = relay.Function(relay.analysis.free_vars(expr), expr)
    func = func.with_attr("Primitive", 1)
    return func


def test_simple():
    """
    Test simple graph with one primitive function.
    """

    def get_inner_func():
        x = relay.var("x", shape=(1, 2, 2, 4), dtype="int8")
        x = relay.nn.max_pool2d(x)
        x = _create_primitive_function(x)
        return x

    ifm = relay.var("input", shape=(1, 2, 2, 4), dtype="int8")
    call = relay.Call(get_inner_func(), [ifm])
    mod = tvm.IRModule.from_expr(call)

    expected_annotations = [
        [2 * (1 * 2 * 2 * 4)],
    ]
    expected_io_annotation = 2 * (1 * 2 * 2 * 4)
    _check_used_memory_annotations(mod, expected_annotations, expected_io_annotation)


def test_multiple_functions():
    """
    Test a graph with multiple primitive functions.
    """

    def get_inner_func(ifm_shape):
        x = relay.var("x", shape=ifm_shape, dtype="int8")
        x = relay.nn.max_pool2d(x, pool_size=(2, 2), layout="NHWC")
        x = _create_primitive_function(x)
        return x

    ifm = relay.var("input", shape=(1, 8, 8, 2), dtype="int8")
    x = get_inner_func((1, 8, 8, 2))
    x = relay.Call(x, [ifm])
    y = get_inner_func((1, 7, 7, 2))
    y = relay.Call(y, [x])
    z = get_inner_func((1, 6, 6, 2))
    z = relay.Call(z, [y])
    mod = tvm.IRModule.from_expr(z)

    expected_annotations = [
        [(1 * 8 * 8 * 2) + (1 * 7 * 7 * 2)],
        [(1 * 7 * 7 * 2) + (1 * 6 * 6 * 2)],
        [(1 * 6 * 6 * 2) + (1 * 5 * 5 * 2)],
    ]
    expected_io_annotation = (1 * 8 * 8 * 2) + (1 * 5 * 5 * 2)
    _check_used_memory_annotations(mod, expected_annotations, expected_io_annotation)


def test_mixed_data_types():
    """
    Test a graph with a primitive function that has mixed datatypes.
    """

    def get_inner_func():
        x = relay.var("x", shape=(1, 2, 2, 2), dtype="int16")
        x = relay.cast(x, dtype="uint32")
        x = _create_primitive_function(x)
        return x

    ifm = relay.var("input", shape=(1, 2, 2, 2), dtype="int16")
    x = get_inner_func()
    x = relay.Call(x, [ifm])
    mod = tvm.IRModule.from_expr(x)

    expected_annotations = [
        [(1 * 2 * 2 * 2) * 2 + (1 * 2 * 2 * 2) * 4],
    ]
    expected_io_annotation = (1 * 2 * 2 * 2) * 2 + (1 * 2 * 2 * 2) * 4
    _check_used_memory_annotations(mod, expected_annotations, expected_io_annotation)


def test_parallel_function_call():
    """
    Test a graph when the results of two functions are concatenated
    into a single result. The second function will also have the result
    of the first function alive so will be annotated with a larger
    "used memory" value.
    """

    def get_inner_func():
        x = relay.var("x", shape=(1, 4, 5, 6), dtype="int8")
        x = relay.reshape(x, newshape=(1, 4, 30))
        x = _create_primitive_function(x)
        return x

    ifm = relay.var("input", shape=(1, 4, 5, 6), dtype="int8")
    x = relay.Call(get_inner_func(), [ifm])
    y = relay.Call(get_inner_func(), [ifm])
    z = relay.concatenate([x, y], axis=0)
    mod = tvm.IRModule.from_expr(z)

    expected_annotations = [
        [(1 * 4 * 5 * 6) + (1 * 4 * 30)],
        # the output tensor from the previous function is also alive
        [(1 * 4 * 5 * 6) + (1 * 4 * 30) + (1 * 4 * 30)],
    ]
    expected_io_annotation = (1 * 4 * 5 * 6) + (1 * 4 * 60)
    _check_used_memory_annotations(mod, expected_annotations, expected_io_annotation)


def test_many_different_parallel_calls():
    """
    Test a graph that calls many different functions in parallel.

                    input
            /         |         \
    prim_func_1  prim_func_2  prim_func_3
           \         |         /
                 prim_func_4
    """

    def get_inner_func_1():
        x = relay.var("x", shape=(1, 4, 5, 6), dtype="int8")
        x = relay.tanh(x)
        x = _create_primitive_function(x)
        return x

    def get_inner_func_2():
        x = relay.var("x", shape=(1, 4, 5, 6), dtype="int8")
        x = relay.nn.max_pool2d(x, pool_size=(1, 1), layout="NHWC")
        x = _create_primitive_function(x)
        return x

    def get_inner_func_3():
        x = relay.var("x", shape=(1, 4, 5, 6), dtype="int8")
        x = relay.abs(x)
        x = relay.nn.relu(x)
        x = relay.exp(x)
        x = _create_primitive_function(x)
        return x

    def get_inner_func_4():
        x = relay.var("x", shape=(1, 4, 5, 6), dtype="int8")
        y = relay.var("y", shape=(1, 4, 5, 6), dtype="int8")
        z = relay.var("z", shape=(1, 4, 5, 6), dtype="int8")
        out = relay.concatenate([x, y, z], axis=3)
        out = _create_primitive_function(out)
        return out

    ifm = relay.var("input", shape=(1, 4, 5, 6), dtype="int8")
    x = relay.Call(get_inner_func_1(), [ifm])
    y = relay.Call(get_inner_func_2(), [ifm])
    z = relay.Call(get_inner_func_3(), [ifm])
    a = relay.Call(get_inner_func_4(), [x, y, z])
    mod = tvm.IRModule.from_expr(a)

    expected_annotations = [
        [(1 * 4 * 5 * 6) + (1 * 4 * 5 * 6)],
        # output from prim_func_1 is also still alive
        [(1 * 4 * 5 * 6) + (1 * 4 * 5 * 6) + (1 * 4 * 5 * 6)],
        # outputs from prim_func_1 and prim_func_2 are also still alive
        [(1 * 4 * 5 * 6) + (1 * 4 * 5 * 6) + (1 * 4 * 5 * 6) + (1 * 4 * 5 * 6)],
        [(1 * 4 * 5 * 6) + (1 * 4 * 5 * 6) + (1 * 4 * 5 * 6) + (1 * 4 * 5 * 18)],
    ]
    expected_io_annotation = (1 * 4 * 5 * 6) + (1 * 4 * 5 * 18)
    _check_used_memory_annotations(mod, expected_annotations, expected_io_annotation)


def test_nested_branches():
    """
    Tests a graph with branches that also branch.

             input
            /     \
          /        \
    prim_func_1  prim_func_2
                   /     \
                  /       \
            prim_func_3   prim_func_4
    """

    def get_generic_inner_func():
        x = relay.var("x", shape=(1, 2, 2, 4), dtype="int8")
        x = relay.nn.relu(x)
        return _create_primitive_function(x)

    ifm = relay.var("input", shape=(1, 2, 2, 4), dtype="int8")
    a = relay.Call(get_generic_inner_func(), [ifm])
    b = relay.Call(get_generic_inner_func(), [ifm])
    c = relay.Call(get_generic_inner_func(), [b])
    d = relay.Call(get_generic_inner_func(), [b])
    out = relay.concatenate([a, c, d], axis=3)
    mod = tvm.IRModule.from_expr(out)

    expected_annotations = [
        [(1 * 2 * 2 * 4) + (1 * 2 * 2 * 4)],
        # output from prim_func_1 is also still alive
        [(1 * 2 * 2 * 4) + (1 * 2 * 2 * 4) + (1 * 2 * 2 * 4)],
        # output from prim_func_1 is also still alive
        [(1 * 2 * 2 * 4) + (1 * 2 * 2 * 4) + (1 * 2 * 2 * 4)],
        # outputs from prim_func_1 and prim_func_3 are also still alive
        [(1 * 2 * 2 * 4) + (1 * 2 * 2 * 4) + (1 * 2 * 2 * 4) + (1 * 2 * 2 * 4)],
    ]
    expected_io_annotation = (1 * 2 * 2 * 4) + (1 * 2 * 2 * 12)
    _check_used_memory_annotations(mod, expected_annotations, expected_io_annotation)


def test_composite_inner_function():
    """
    Tests the typical BYOC use case where a primitive function
    contains a composite function.
    """

    def get_inner_func():
        x = relay.var("x", shape=(1, 2, 2, 4), dtype="int8")
        x = relay.nn.max_pool2d(x, pool_size=(2, 2), layout="NHWC")
        x = relay.Function(relay.analysis.free_vars(x), x)
        x = x.with_attr("Composite", "my_composite_func")

        y = relay.var("y", shape=(1, 2, 2, 4), dtype="int8")
        z = relay.Call(x, [y])
        return _create_primitive_function(z)

    ifm = relay.var("input", shape=(1, 2, 2, 4), dtype="int8")
    x = relay.Call(get_inner_func(), [ifm])
    mod = tvm.IRModule.from_expr(x)

    expected_annotations = [
        [(1 * 2 * 2 * 4) + (1 * 1 * 1 * 4)],
    ]
    expected_io_annotation = (1 * 2 * 2 * 4) + (1 * 1 * 1 * 4)
    _check_used_memory_annotations(mod, expected_annotations, expected_io_annotation)


def test_multiple_calls_to_same_function():
    """
    Tests the case when there are multiple calls to the same function.
    """

    def get_inner_func():
        x = relay.var("x", shape=(1, 2, 2, 4), dtype="int8")
        x = relay.nn.max_pool2d(x)
        x = _create_primitive_function(x)
        return x

    inner_func = get_inner_func()
    ifm = relay.var("input", shape=(1, 2, 2, 4), dtype="int8")
    call1 = relay.Call(inner_func, [ifm])
    call2 = relay.Call(inner_func, [call1])
    mod = tvm.IRModule.from_expr(call2)

    expected_annotations = [[2 * (1 * 2 * 2 * 4), 2 * (1 * 2 * 2 * 4)]]
    expected_io_annotation = 2 * (1 * 2 * 2 * 4)
    _check_used_memory_annotations(mod, expected_annotations, expected_io_annotation)


def test_parallel_calls_to_same_function():
    """
    Test parallel calls to the same function.
    """

    def get_inner_func():
        x = relay.var("x", shape=(1, 2, 2, 4), dtype="int8")
        x = relay.nn.max_pool2d(x)
        x = _create_primitive_function(x)
        return x

    inner_func = get_inner_func()
    ifm = relay.var("input", shape=(1, 2, 2, 4), dtype="int8")
    call1 = relay.Call(inner_func, [ifm])
    call2 = relay.Call(inner_func, [ifm])
    concat = relay.concatenate([call1, call2], axis=0)
    mod = tvm.IRModule.from_expr(concat)

    expected_annotations = [[2 * (1 * 2 * 2 * 4), 3 * (1 * 2 * 2 * 4)]]
    expected_io_annotation = 3 * (1 * 2 * 2 * 4)
    _check_used_memory_annotations(mod, expected_annotations, expected_io_annotation)


def test_parallel_calls_with_non_ifm_input():
    """
    Test a graph that calls many different functions in parallel where
    the input is not the input to the function.

                    y = f(x)
            /         |         \
       z0 = g0(y)    ...      zi = gi(y)
           \         |         /
                  concat
    """

    def get_inner_func_1():
        x = relay.var("x", shape=(1, 4, 5, 6), dtype="int8")
        x = relay.tanh(x)
        x = _create_primitive_function(x)
        return x

    def get_inner_func_2():
        x = relay.var("x", shape=(1, 4, 5, 6), dtype="int8")
        x = relay.nn.max_pool2d(x, pool_size=(2, 2))
        x = _create_primitive_function(x)
        return x

    ifm = relay.var("input", shape=(1, 4, 5, 6), dtype="int8")
    y = relay.Call(get_inner_func_1(), [ifm])
    g = get_inner_func_2()

    no_calls = 20
    z = [relay.Call(g, [y]) for _ in range(0, no_calls)]
    out = relay.concatenate(z, axis=3)
    mod = tvm.IRModule.from_expr(out)

    expected_annotations = [
        [(1 * 4 * 5 * 6) + (1 * 4 * 5 * 6)],
        [(1 * 4 * 5 * 6) + (1 * 4 * 4 * 5) * i for i in range(1, no_calls + 1)],
    ]
    expected_io_annotation = (1 * 4 * 5 * 6) + (1 * 4 * 4 * (5 * no_calls))
    _check_used_memory_annotations(mod, expected_annotations, expected_io_annotation)


def test_dynamic_io_tensor_not_supported():
    """
    Test to check dynamic IO tensor error.
    """

    def get_inner_func():
        x = relay.var("x", shape=(1, 2, 2, 4), dtype="int8")
        x = relay.nn.max_pool2d(x)
        x = _create_primitive_function(x)
        return x

    ifm = relay.var("input", shape=(1, 2, 2, relay.Any()), dtype="int8")
    call = relay.Call(get_inner_func(), [ifm])
    mod = tvm.IRModule.from_expr(call)

    err_rgx = r"AnnotateUsedMemory does not support dynamic shapes"
    with pytest.raises(tvm.TVMError, match=err_rgx):
        _check_used_memory_annotations(mod, [], [])


def test_dynamic_callsite_tensor_not_supported():
    """
    Test to check dynamic callsite tensor error.
    """

    def get_inner_func():
        x = relay.var("x", shape=(relay.Any(), 2, 2, 4), dtype="int8")
        x = relay.nn.max_pool2d(x)
        x = _create_primitive_function(x)
        return x

    ifm = relay.var("input", shape=(1, 2, 2, 4), dtype="int8")
    call = relay.Call(get_inner_func(), [ifm])
    mod = tvm.IRModule.from_expr(call)

    err_rgx = r"AnnotateUsedMemory does not support dynamic shapes"
    with pytest.raises(tvm.TVMError, match=err_rgx):
        _check_used_memory_annotations(mod, [], [])
