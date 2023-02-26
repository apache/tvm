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

import pytest
import tvm
from tvm import relay
from tvm.relay.analysis import check_basic_block_normal_form


def test_one_block():
    x = relay.var("x")
    y = relay.add(x, x)
    z = relay.add(x, y)
    check_basic_block_normal_form(z)


def test_let():
    x = relay.var("x")
    y = relay.var("y")
    body = relay.Let(y, x, y)
    check_basic_block_normal_form(body)


@pytest.mark.xfail(raises=tvm.error.TVMError)
def test_invalid_if():
    cond = relay.var("cond", dtype="bool", shape=())
    shared = relay.var("shared")
    true_branch = shared
    false_branch = relay.add(shared, shared)
    body = relay.If(cond, true_branch, false_branch)
    """
    The program below violates basic block normal form, as the scope of %shared
    is ambiguous and should not be in that of true branch.

    free_var %cond: bool
    if (%cond) {
      free_var %shared
      %shared
    } else {
      add(%shared, %shared)
    }
    """
    check_basic_block_normal_form(body)


def test_valid_if():
    cond = relay.var("cond", dtype="bool", shape=())
    shared = relay.var("shared")
    true_branch = shared
    false_branch = relay.add(shared, shared)
    body = relay.If(cond, true_branch, false_branch)
    shared_bound = relay.var("shared_bound", shape=(1,), dtype="float32")
    body = relay.Let(shared, shared_bound, body)
    """
    The program below uses let binding to control the scope of %shared, which
    follows the basic block normal form.

    free_var %shared_bound: Tensor[(1), float32]
    let %shared = %shared_bound;
    free_var %cond: bool
    if (%cond) {
      %shared
    } else {
      add(%shared, %shared)
    }
    """
    check_basic_block_normal_form(body)


@pytest.mark.xfail(raises=tvm.error.TVMError)
def test_invalid_if2():
    """
    fn (%x: float32) {
      %0 = equal(%x, 2f);
      if (%0) {
        %1 = add(%x, 1f);
        multiply(%1, 2f)
      } else {
        multiply(%1, 1f)
      }
    }
    """
    x = relay.var("x", shape=(), dtype="float32")
    one = relay.const(1, dtype="float32")
    two = relay.const(2, dtype="float32")
    v1 = relay.add(x, one)
    v2 = relay.equal(x, two)
    true_branch = relay.multiply(v1, two)
    false_branch = relay.multiply(v1, one)
    body = relay.If(v2, true_branch, false_branch)
    func = relay.Function([x], body)
    check_basic_block_normal_form(func)


def test_valid_if2():
    """
    fn (%x: float32) {
      let %v1 = add(%x, 1f);
      %0 = equal(%x, 2f);
      if (%0) {
        multiply(%v1, 2f)
      } else {
        multiply(%v1, 1f)
      }
    }
    """
    x = relay.var("x", shape=(), dtype="float32")
    one = relay.const(1, dtype="float32")
    two = relay.const(2, dtype="float32")
    v1 = relay.var("v1")
    v2 = relay.equal(x, two)
    true_branch = relay.multiply(v1, two)
    false_branch = relay.multiply(v1, one)
    body = relay.If(v2, true_branch, false_branch)
    body = relay.Let(v1, relay.add(x, one), body)
    func = relay.Function([x], body)
    check_basic_block_normal_form(func)


@pytest.mark.xfail(raises=tvm.error.TVMError)
def test_func():
    x = relay.var("x", shape=(1,), dtype="float32")  # , a)
    y = relay.var("y", shape=(1,), dtype="float32")  # , a)
    z = relay.var("z", shape=(1,), dtype="float32")  # , a)
    x2 = relay.add(x, x)
    func_a = relay.Function([y], relay.add(x2, y))  # , a, [a])
    func_b = relay.Function([z], relay.add(x2, z))  # , a, [a])
    body = relay.Tuple([func_a, func_b])
    body = relay.Function([x], body)
    """
    fn (%x: Tensor[(1), float32]) {
      %1 = fn (%y: Tensor[(1), float32]) {
        %0 = add(%x, %x);
        add(%0, %y)
      };
      %2 = fn (%z: Tensor[(1), float32]) {
        add(%0, %z)
      };
      (%1, %2)
    }
    """
    check_basic_block_normal_form(body)


@pytest.mark.xfail(raises=tvm.error.TVMError)
def test_higher_order_return():
    x = relay.var("x", shape=(1,), dtype="float32")  # , a)
    y = relay.var("y", shape=(1,), dtype="float32")  # , a)
    z = relay.var("z", shape=(1,), dtype="float32")  # , a)
    x2 = relay.add(x, x)
    func_a = relay.Function([y], relay.add(x2, y))  # , a, [a])
    func_b = relay.Function([z], relay.add(x2, z))  # , a, [a])
    body = relay.Tuple([func_a, func_b])
    body = relay.Function([x], body)
    """
    fn (%x: Tensor[(1), float32]) {
      %1 = fn (%y: Tensor[(1), float32]) {
        %0 = add(%x, %x);
        add(%0, %y)
      };
      %2 = fn (%z: Tensor[(1), float32]) {
        add(%0, %z)
      };
      (%1, %2)
    }
    """
    check_basic_block_normal_form(body)


@pytest.mark.xfail(raises=tvm.error.TVMError)
def test_higher_order_nested():
    x = relay.var("x", dtype="float32", shape=(1,))
    s = relay.var("s", dtype="float32", shape=(1,))
    shared = relay.add(s, s)
    func_true = relay.Function([x], relay.add(x, shared))
    choice_t = relay.FuncType([], relay.scalar_type("bool"))
    f = relay.Var("f", choice_t)
    z = relay.Var("z")
    body = relay.If(f(), func_true, relay.Function([z], relay.add(z, shared)))
    top = relay.Function([f, s], body)
    """
    fn (%f: fn () -> bool, %s: Tensor[(1), float32]) {
      %0 = %f();
      if (%0) {
        fn (%x: Tensor[(1), float32]) {
          %1 = add(%s, %s);
          add(%x, %1)
        }
      } else {
        fn (%z) {
          add(%z, %1)
        }
      }
    }
    """
    check_basic_block_normal_form(top)


if __name__ == "__main__":
    tvm.testing.main()
