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

    def __init__(self, expected_annotations):
        self.expected_annotations = expected_annotations
        super().__init__()

    def visit_function(self, fn):
        if "Primitive" in fn.attrs:
            assert (
                "used_memory" in fn.attrs
            ), "Primitive function does not have used_memory annotation."

            assert len(self.expected_annotations) > 0, "Not all expected annotations were compared"

            expected_mem = self.expected_annotations.pop(0)
            actual_mem = fn.attrs["used_memory"]
            assert expected_mem == actual_mem, (
                f"Expected used memory annotation {expected_mem} "
                f"did not match actual annotation {actual_mem}"
            )
        super().visit_function(fn)


def _check_used_memory_annotations(mod, expected_annotations):
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.ToANormalForm()(mod)
    mod = relay.transform.InferType()(mod)
    mod = AnnotateUsedMemory()(mod)

    CheckUsedMemoryAnnotation(expected_annotations).visit(mod["main"].body)


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

    expected_annotations = [2 * (1 * 2 * 2 * 4)]
    _check_used_memory_annotations(mod, expected_annotations)


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
        (1 * 8 * 8 * 2) + (1 * 7 * 7 * 2),
        (1 * 7 * 7 * 2) + (1 * 6 * 6 * 2),
        (1 * 6 * 6 * 2) + (1 * 5 * 5 * 2),
    ]
    _check_used_memory_annotations(mod, expected_annotations)


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
        (1 * 2 * 2 * 2) * 2 + (1 * 2 * 2 * 2) * 4,
    ]
    _check_used_memory_annotations(mod, expected_annotations)


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
        (1 * 4 * 5 * 6) + (1 * 4 * 30),
        # the output tensor from the previous function is also alive
        (1 * 4 * 5 * 6) + (1 * 4 * 30) + (1 * 4 * 30),
    ]
    _check_used_memory_annotations(mod, expected_annotations)


def test_composite_inner_function():
    """
    Tests the typical BYOC use case where a primitive function
    contains a composite function.
    """

    def get_inner_func():
        x = relay.var("x", shape=(1, 2, 2, 4), dtype="int8")
        x = relay.nn.max_pool2d(x, pool_size=(2, 2))
        x = relay.Function(relay.analysis.free_vars(x), x)
        x = x.with_attr("Composite", "my_composite_func")

        y = relay.var("y", shape=(1, 2, 2, 4), dtype="int8")
        z = relay.Call(x, [y])
        z = _create_primitive_function(z)
        return x

    ifm = relay.var("input", shape=(1, 2, 2, 4), dtype="int8")
    x = relay.Call(get_inner_func(), [ifm])
    mod = tvm.IRModule.from_expr(x)

    expected_annotations = [(1 * 2 * 2 * 4) + (1 * 1 * 1 * 4)]
    _check_used_memory_annotations(mod, expected_annotations)
