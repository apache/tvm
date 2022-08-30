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
import tvm
from tvm import relay
from tvm.relay import testing

import pytest
import numpy as np


def make_rel(name, args, num_inputs=None, attrs=None):
    func = tvm.ir.EnvFunc.get("tvm.relay.type_relation." + name)
    if num_inputs is None:
        num_inputs = len(args) - 1
    return relay.ty.TypeRelation(func, args, num_inputs, attrs)


def make_solver():
    solver = relay.analysis._ffi_api._test_type_solver()
    solver.Solve = solver("Solve")
    solver.Unify = solver("Unify")
    solver.Resolve = solver("Resolve")
    solver.AddConstraint = solver("AddConstraint")

    def gen_type(name, args, out=None):
        out = out if out else relay.ty.IncompleteType()
        solver.AddConstraint(make_rel(name, args + [out]))
        return out

    solver.gen_type = gen_type
    return solver


def test_bcast():
    solver = make_solver()
    t0 = relay.ty.TensorType((10, 20), "float32")
    t1 = relay.ty.TensorType((10, 1), "float32")
    tc = relay.ty.TensorType((10, 1, 1), "float32")
    t2 = solver.gen_type("Broadcast", [t0, t1])
    t3 = solver.gen_type("Identity", [t2])
    t4 = solver.gen_type("Broadcast", [t3, tc])
    assert solver.Solve()
    assert solver.Resolve(t2) == relay.ty.TensorType((10, 20), "float32")
    assert solver.Resolve(t4) == relay.ty.TensorType((10, 10, 20), "float32")


def test_backward_solving():
    solver = make_solver()
    t0 = relay.ty.TensorType((10, 20), "float32")
    tc = relay.ty.TensorType((10, 1, 1), "float32")
    t1 = relay.ty.IncompleteType()
    t3 = solver.gen_type("Broadcast", [t0, t1])
    t2 = solver.gen_type("Identity", [t1], out=tc)
    assert solver.Solve()
    assert solver.Resolve(t3) == relay.ty.TensorType((10, 10, 20), "float32")


def test_unify_tuple():
    solver = make_solver()
    t1 = relay.ty.IncompleteType()
    t2 = relay.ty.IncompleteType()
    t3 = relay.ty.TensorType((10, 20), "float32")

    tup1 = relay.ty.TupleType([t1, t2])
    tup2 = relay.ty.TupleType([t3, t3])

    unified = solver.Unify(tup1, tup2)
    assert unified == tup2


def test_unify_global_type_var():
    # should only be able to unify if they're the same
    solver = make_solver()
    gtv = relay.GlobalTypeVar("gtv")
    unified = solver.Unify(gtv, gtv)
    assert unified == gtv


def test_unify_typecall():
    solver = make_solver()
    gtv = relay.GlobalTypeVar("gtv")

    # yeah, typecalls are shaped like tuples so the same
    # tests work out
    t1 = relay.ty.IncompleteType()
    t2 = relay.ty.IncompleteType()
    t3 = relay.ty.TensorType((10, 20), "float32")

    tc1 = relay.ty.TypeCall(gtv, [t1, t2])
    tc2 = relay.ty.TypeCall(gtv, [t3, t3])
    unified = solver.Unify(tc1, tc2)
    assert unified == tc2


def test_unify_functype():
    solver = make_solver()
    t1 = relay.ty.IncompleteType()
    t2 = relay.ty.IncompleteType()
    t3 = relay.ty.IncompleteType()

    unit = relay.ty.TupleType([])
    tensor1 = relay.ty.TensorType((10, 20), "float32")
    tensor2 = relay.ty.TensorType((10,), "float32")

    ft1 = relay.ty.FuncType([t1, t2], t3)
    ft2 = relay.ty.FuncType([tensor1, tensor2], unit)

    unified = solver.Unify(ft1, ft2)
    assert unified == ft2


def test_recursive_unify():
    solver = make_solver()
    t1 = relay.ty.IncompleteType()
    t2 = relay.ty.IncompleteType()
    t3 = relay.ty.IncompleteType()

    tensor1 = relay.ty.TensorType((10, 10, 20), "float32")
    tensor2 = relay.ty.TensorType((10, 20), "float32")
    tensor3 = relay.ty.TensorType((10,), "float32")

    tup1 = relay.ty.TupleType([relay.ty.TupleType([t1, t2]), t2])
    tup2 = relay.ty.TupleType([relay.ty.TupleType([tensor1, tensor2]), tensor2])

    ft1 = relay.ty.FuncType([tup1, t3], t3)
    ft2 = relay.ty.FuncType([tup2, tensor3], tensor3)

    unified = solver.Unify(ft1, ft2)
    assert unified == ft2


def test_unify_vars_under_tuples():
    solver = make_solver()
    t1 = relay.ty.IncompleteType()

    tup1 = relay.ty.TupleType([t1, t1])
    unified = solver.Unify(tup1, tup1)
    assert unified == tup1

    t2 = relay.ty.IncompleteType()
    tup2 = relay.ty.TupleType([t2, t2])

    tup3 = relay.ty.TupleType([t1, t2])
    tup4 = relay.ty.TupleType([t2, t1])
    unified = solver.Unify(tup3, tup4)
    assert unified == tup1 or unified == tup2


def test_binding_over_typevars():
    solver = make_solver()

    t1 = relay.ty.IncompleteType()
    t2 = relay.ty.IncompleteType()

    a = relay.ty.TypeVar("a")
    b = relay.ty.TypeVar("b")
    c = relay.ty.TypeVar("c")
    d = relay.ty.TypeVar("d")

    ft1 = relay.ty.FuncType([t1], t2, [c, d])
    ft2 = relay.ty.FuncType([a], b, [a, b])
    unified = solver.Unify(ft1, ft2)
    assert unified == solver.Resolve(ft1)


def test_recursive_backward_solving():
    solver = make_solver()

    tensor1 = relay.ty.TensorType((10, 20), "float32")
    tensor2 = relay.ty.TensorType((10, 1, 1), "float32")
    tensor3 = relay.ty.TensorType((10,), "float32")

    t1 = relay.ty.IncompleteType()
    t2 = relay.ty.IncompleteType()
    t3 = relay.ty.IncompleteType()

    tup1 = relay.ty.TupleType([relay.ty.TupleType([tensor1, tensor2]), tensor3])
    tup2 = relay.ty.TupleType([relay.ty.TupleType([t1, t2]), t3])
    solver.gen_type("Identity", [tup1], out=tup2)

    assert solver.Solve()
    assert solver.Resolve(tup2) == tup1


def test_backward_solving_after_child_update():
    solver = make_solver()

    tensor1 = relay.ty.TensorType((10, 20), "float32")
    tensor2 = relay.ty.TensorType((10, 1, 1), "float32")

    t1 = relay.ty.IncompleteType()
    t2 = relay.ty.IncompleteType()
    t3 = relay.ty.IncompleteType()

    tup1 = relay.ty.TupleType([t1, t2])
    tup2 = relay.ty.TupleType([t1, t3])

    tup_concrete = relay.ty.TupleType([tensor1, tensor2])

    t4 = solver.gen_type("Identity", [tup1])
    t5 = solver.gen_type("Identity", [tup2])

    solver.gen_type("Identity", [t4], out=t5)
    assert solver.Solve()
    assert solver.Resolve(t3) == t3 or solver.Resolve(t3) == t2
    assert solver.Resolve(t4) == tup1 or solver.Resolve(t4) == tup2
    assert solver.Resolve(t5) == tup1 or solver.Resolve(t5) == tup2

    # updating the variables *inside* tup1 and tup2 should update t4 and t5
    solver.gen_type("Identity", [t1], out=tensor1)
    solver.gen_type("Identity", [t2], out=tensor2)
    assert solver.Solve()
    assert solver.Resolve(t4) == tup_concrete
    assert solver.Resolve(t5) == tup_concrete


def test_unify_quantified_funcs():
    solver = make_solver()
    a, b, c = relay.TypeVar("a"), relay.TypeVar("b"), relay.TypeVar("c")
    ft1 = relay.FuncType([a, b], c, [a, b, c])
    ft2 = relay.FuncType([a, a], a, [a])
    unified = solver.Unify(ft1, ft2)
    assert unified == ft2

    ft3 = relay.FuncType([a], a, [a])
    ft4 = relay.FuncType([b], c, [b, c])
    unified = solver.Unify(ft3, ft4)
    assert unified == ft3


def test_unify_quantified_func_and_concrete():
    solver = make_solver()
    a, b = relay.TypeVar("a"), relay.TypeVar("b")
    ft1 = relay.FuncType([a], b, [a, b])
    ft2 = relay.FuncType([b], relay.TupleType([]), [b])
    unified = solver.Unify(ft1, ft2)
    assert unified == ft2


def test_unify_quantified_funcs_nesting():
    solver = make_solver()
    a, b, c = relay.TypeVar("a"), relay.TypeVar("b"), relay.TypeVar("c")

    ft1 = relay.FuncType([a, relay.TupleType([b, c])], relay.TupleType([a, b, c]), [a, b, c])
    ft2 = relay.FuncType([a, relay.TupleType([a, a])], relay.TupleType([a, a, a]), [a])
    unified = solver.Unify(ft1, ft2)
    assert unified == ft2


def test_unify_quantified_funcs_var_order():
    solver = make_solver()
    a, b, c = relay.TypeVar("a"), relay.TypeVar("b"), relay.TypeVar("c")

    ft1 = relay.FuncType([a, relay.TupleType([b, c])], relay.TupleType([a, b, c]), [a, b, c])
    ft2 = relay.FuncType([a, relay.TupleType([a, c])], relay.TupleType([a, a, c]), [a, c])
    # unified = solver.Unify(ft1, ft2) # crashes here but it shouldn't
    # assert unified == ft2


@pytest.mark.xfail(raises=tvm._ffi.base.TVMError)
def test_incompatible_tuple_unification():
    solver = make_solver()
    t1 = relay.ty.IncompleteType()
    t2 = relay.ty.IncompleteType()

    tensor1 = relay.ty.TensorType((1, 2, 3), "float32")
    tensor2 = relay.ty.TensorType((2, 3), "float32")
    tensor3 = relay.ty.TensorType((3,), "float32")

    tup1 = relay.ty.TupleType([relay.ty.TupleType([t1, t1]), t2])
    tup2 = relay.ty.TupleType([relay.ty.TupleType([tensor1, tensor2]), tensor3])
    solver.Unify(tup1, tup2)


@pytest.mark.xfail(raises=tvm._ffi.base.TVMError)
def test_bad_recursive_unification():
    solver = make_solver()
    t1 = relay.ty.IncompleteType()
    solver.Unify(t1, relay.ty.TupleType([t1, t1]))


@pytest.mark.xfail(raises=tvm._ffi.base.TVMError)
def test_unify_invalid_global_typevars():
    solver = make_solver()
    gtv1 = relay.GlobalTypeVar("gtv1")
    gtv2 = relay.GlobalTypeVar("gtv2")
    solver.Unify(gtv1, gtv2)


@pytest.mark.xfail(raises=tvm._ffi.base.TVMError)
def test_incompatible_typecall_var_unification():
    solver = make_solver()
    gtv1 = relay.GlobalTypeVar("gtv1")
    gtv2 = relay.GlobalTypeVar("gtv2")

    t1 = relay.IncompleteType()
    t2 = relay.IncompleteType()

    tc1 = relay.TypeCall(gtv1, [t1])
    tc2 = relay.TypeCall(gtv2, [t2])
    solver.Unify(tc1, tc2)


@pytest.mark.xfail(raises=tvm._ffi.base.TVMError)
def test_incompatible_typecall_args_unification():
    solver = make_solver()
    gtv = relay.GlobalTypeVar("gtv1")
    t1 = relay.IncompleteType()
    t2 = relay.IncompleteType()

    tensor1 = relay.TensorType((1, 2, 3), "float32")
    tensor2 = relay.TensorType((2, 3), "float32")
    tensor3 = relay.TensorType((3,), "float32")

    tc1 = relay.TypeCall(gtv, [relay.TupleType([t1, t1]), t2])
    tc2 = relay.TypeCall(gtv, [relay.TupleType([tensor1, tensor2]), tensor3])
    solver.Unify(tc1, tc2)


@pytest.mark.xfail(raises=tvm._ffi.base.TVMError)
def test_incompatible_quantified_func_unification():
    solver = make_solver()
    a, b, c = relay.TypeVar("a"), relay.TypeVar("b"), relay.TypeVar("c")

    ft1 = relay.FuncType([a, b], c, [a, b, c])
    ft2 = relay.FuncType([b, c], relay.TupleType([a]), [a, b, c])
    solver.Unify(ft1, ft2)


def test_integer_compatibility_in_layout_transform():
    x = relay.var("data", shape=(2, 3, 48, 48), dtype="float32")
    conv_out = relay.nn.conv2d(
        x,
        relay.var("weight", shape=(1, 3, 1, 1), dtype="float32"),
        strides=[47, 47],
        channels=1,
        kernel_size=[1, 1],
    )
    bias_out = relay.nn.bias_add(conv_out, relay.var("bias"))
    broadcast_out = relay.op.broadcast_to(bias_out, relay.const([2, 1, 2, 2], dtype="int64"))
    y = relay.add(bias_out, broadcast_out)

    mod, _ = testing.create_workload(y)
    with tvm.transform.PassContext(opt_level=3):
        with tvm.target.Target("llvm"):
            mod = relay.transform.CanonicalizeOps()(mod)
            mod = relay.transform.AlterOpLayout()(mod)


if __name__ == "__main__":
    test_bcast()
    test_backward_solving()
    test_unify_tuple()
    test_unify_typecall()
    test_unify_functype()
    test_recursive_unify()
    test_unify_vars_under_tuples()
    test_recursive_backward_solving()
    test_backward_solving_after_child_update()
    test_unify_quantified_funcs()
    test_unify_quantified_func_and_concrete()
    test_unify_quantified_funcs_nesting()
    test_unify_quantified_funcs_var_order()
    test_incompatible_tuple_unification()
    test_bad_recursive_unification()
    test_incompatible_typecall_var_unification()
    test_incompatible_typecall_args_unification()
    test_incompatible_quantified_func_unification()
    test_integer_compatibility_in_layout_transform()
