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

import numpy as np

from tvm import relay, testing, transform
from tvm.relay.frontend.common import StrAttrsDict, set_span
from relay.utils.tag_span import _set_span, _create_span, _verify_structural_equal_with_span


def test_key_is_present():
    attrs = StrAttrsDict({"a": 1})
    assert attrs.has_attr("a")


def test_key_is_not_present():
    attrs = StrAttrsDict({"a": 1})
    assert not attrs.has_attr("b")


class TestSetSpan:
    def test_pass_ctx_switch(self):
        def _res(should_fill):
            if should_fill:
                with testing.enable_span_filling():
                    return set_span(relay.var("x", shape=(1, 64, 56, 56)), "x_var")
            else:
                with testing.disable_span_filling():
                    return set_span(relay.var("x", shape=(1, 64, 56, 56)), "x_var")

        disable = relay.var("x", shape=(1, 64, 56, 56))
        enable = relay.var("x", shape=(1, 64, 56, 56), span=_create_span("x_var"))

        _verify_structural_equal_with_span(_res(False), disable)
        _verify_structural_equal_with_span(_res(True), enable)

    # Should tag all exprs without span, and stop when expr is span-tagged
    def test_builtin_tuple(self):
        def _res():
            a = relay.const(np.ones([1, 1, 1]), dtype="int64", span=_create_span("a"))
            b = relay.const(np.zeros([1, 1, 1]), dtype="int64")
            return set_span(tuple([a, b]), "tuple")

        def _golden():
            a = relay.const(np.ones([1, 1, 1]), dtype="int64", span=_create_span("a"))
            b = relay.const(np.zeros([1, 1, 1]), dtype="int64", span=_create_span("tuple"))
            return tuple([a, b])

        res_tuple, golden_tuple = _res(), _golden()
        assert len(res_tuple) == len(golden_tuple)
        for i in range(len(res_tuple)):
            _verify_structural_equal_with_span(res_tuple[i], golden_tuple[i])

    def test_builtin_list(self):
        def _res():
            a = relay.const(np.ones([1, 1, 1]), dtype="int64", span=_create_span("a"))
            b = relay.const(np.zeros([1, 1, 1]), dtype="int64")
            t = relay.Tuple([a, b])
            t_a = relay.TupleGetItem(t, 0)
            t_b = relay.TupleGetItem(t, 1)
            return set_span([t_a, t_b], "list")

        def _golden():
            a = relay.const(np.ones([1, 1, 1]), dtype="int64", span=_create_span("a"))
            b = relay.const(np.zeros([1, 1, 1]), dtype="int64", span=_create_span("list"))
            t = relay.Tuple([a, b], span=_create_span("list"))
            t_a = relay.TupleGetItem(t, 0, span=_create_span("list"))
            t_b = relay.TupleGetItem(t, 1, span=_create_span("list"))
            return [t_a, t_b]

        res_list, golden_list = _res(), _golden()
        assert len(res_list) == len(golden_list)
        for i in range(len(res_list)):
            _verify_structural_equal_with_span(res_list[i], golden_list[i])

    def test_var(self):
        x = set_span(relay.var("x", shape=(1, 64, 56, 56)), "x_var")
        x_expected = relay.var("x", shape=(1, 64, 56, 56), span=_create_span("x_var"))
        _verify_structural_equal_with_span(x, x_expected)

    def test_constant(self):
        c = set_span(relay.const(np.ones([64, 64, 3, 3]), dtype="int64"), "const_c")
        c_expected = relay.const(
            np.ones([64, 64, 3, 3]), dtype="int64", span=_create_span("const_c")
        )
        _verify_structural_equal_with_span(c, c_expected)

    def test_call(self):
        def _res():
            x = set_span(relay.var("x", shape=(1, 64, 56, 56)), "x_var")
            w = relay.const(np.ones([64, 64, 3, 3]), dtype="int64")
            y = set_span(
                relay.nn.conv2d(x, w, channels=64, kernel_size=(3, 3), padding=(1, 1)), "conv2d"
            )
            return relay.Function([x], y)

        def _golden():
            x = relay.var("x", shape=(1, 64, 56, 56), span=_create_span("x_var"))
            w = relay.const(np.ones([64, 64, 3, 3]), dtype="int64", span=_create_span("conv2d"))
            y = _set_span(
                relay.nn.conv2d(x, w, channels=64, kernel_size=(3, 3), padding=(1, 1)), "conv2d"
            )
            return relay.Function([x], y)

        _verify_structural_equal_with_span(_res(), _golden())

    def test_tuple(self):
        def _res():
            a = set_span(relay.const(np.ones([1, 1, 1]), dtype="int64"), "a")
            b = relay.const(np.ones([1, 1, 1]), dtype="int64")
            t = set_span(relay.Tuple([a, b]), "t")
            return relay.Function([], t)

        def _golden():
            a = relay.const(np.ones([1, 1, 1]), dtype="int64", span=_create_span("a"))
            b = relay.const(np.ones([1, 1, 1]), dtype="int64", span=_create_span("t"))
            t = relay.Tuple([a, b], span=_create_span("t"))
            return relay.Function([], t)

        _verify_structural_equal_with_span(_res(), _golden())

    def test_tuple_getitem(self):
        def _res():
            a = set_span(relay.const(np.ones([1, 1, 1]), dtype="int64"), "a")
            b = relay.const(np.ones([1, 1, 1]), dtype="int64")
            t = relay.Tuple([a, b])
            i = set_span(relay.TupleGetItem(t, 0), "i")
            return relay.Function([], i)

        def _golden():
            a = relay.const(np.ones([1, 1, 1]), dtype="int64", span=_create_span("a"))
            b = relay.const(np.ones([1, 1, 1]), dtype="int64", span=_create_span("i"))
            t = relay.Tuple([a, b], span=_create_span("i"))
            i = relay.TupleGetItem(t, 0, span=_create_span("i"))
            return relay.Function([], i)

        _verify_structural_equal_with_span(_res(), _golden())

    def test_let(self):
        def _res():
            x = set_span(relay.Var("x"), "x_var")
            c_1 = relay.const(np.ones(10))
            add = relay.add(x, x)
            body = set_span(relay.Let(x, c_1, add), "let")

            c_2 = set_span(relay.const(np.zeros(10)), "zeros")
            y = set_span(relay.add(body, c_2), "add_2")
            return relay.Function([x], y)

        def _golden():
            x = relay.Var("x", span=_create_span("x_var"))
            c_1 = relay.const(np.ones(10), span=_create_span("let"))
            add = _set_span(relay.add(x, x), "let")
            body = relay.Let(x, c_1, add, span=_create_span("let"))

            c_2 = relay.const(np.zeros(10), span=_create_span("zeros"))
            y = _set_span(relay.add(body, c_2), "add_2")
            return relay.Function([x], y)

        _verify_structural_equal_with_span(_res(), _golden())

    def test_if(self):
        def _res():
            x = set_span(relay.var("x", shape=[], dtype="float32"), "x_var")
            y = set_span(relay.var("y", shape=[], dtype="float32"), "y_var")
            eq = relay.equal(x, y)

            true_branch = set_span(relay.add(x, y), "true_branch")
            false_branch = relay.subtract(x, y)
            ife = set_span(relay.If(eq, true_branch, false_branch), "if")
            return relay.Function([x, y], ife)

        def _golden():
            x = relay.var("x", shape=[], dtype="float32", span=_create_span("x_var"))
            y = relay.var("y", shape=[], dtype="float32", span=_create_span("y_var"))
            eq = _set_span(relay.equal(x, y), "if")

            true_branch = _set_span(relay.add(x, y), "true_branch")
            false_branch = _set_span(relay.subtract(x, y), "if")
            ife = relay.If(eq, true_branch, false_branch, span=_create_span("if"))
            return relay.Function([x, y], ife)

        _verify_structural_equal_with_span(_res(), _golden())

    def test_fn(self):
        def _res():
            x = set_span(relay.var("x", shape=(1, 64, 56, 56)), "x_var")
            w = relay.const(np.ones([64, 64, 3, 3]), dtype="int64")
            y = relay.nn.conv2d(x, w, channels=64, kernel_size=(3, 3), padding=(1, 1))
            f = set_span(relay.Function([x], y), "func")
            return f

        def _golden():
            x = relay.var("x", shape=(1, 64, 56, 56), span=_create_span("x_var"))
            w = relay.const(np.ones([64, 64, 3, 3]), dtype="int64", span=_create_span("func"))
            y = _set_span(
                relay.nn.conv2d(x, w, channels=64, kernel_size=(3, 3), padding=(1, 1)), "func"
            )
            f = relay.Function([x], y, span=_create_span("func"))
            return f

        _verify_structural_equal_with_span(_res(), _golden())


if __name__ == "__main__":
    testing.main()
