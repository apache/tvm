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
import tvm
from tvm import relay
from tvm.relay.analysis import detect_feature
from tvm.relay.transform import to_cps, un_cps
from tvm.relay.analysis import Feature
from tvm.relay.prelude import Prelude
from tvm.relay.testing import add_nat_definitions, make_nat_expr, rand, run_infer_type, run_opt_pass
from tvm.relay import create_executor
from tvm.relay import transform


def test_id():
    x = relay.var("x", shape=[])
    id = run_infer_type(relay.Function([x], x))
    id_cps = run_infer_type(to_cps(id))


def test_double():
    t = relay.TypeVar("t")
    x = relay.var("x", t)
    f = relay.var("f", relay.FuncType([t], t))
    double = run_infer_type(relay.Function([f, x], f(f(x)), t, [t]))
    double_cps = run_infer_type(to_cps(double))


# make sure cps work for recursion.
def test_recursion():
    mod = tvm.IRModule()
    p = Prelude(mod)
    add_nat_definitions(p)
    shape = (10, 10)
    dtype = 'float32'
    t = relay.TensorType(shape, dtype)
    x = relay.var("x", t)
    double = relay.Function([x], x + x)
    i = relay.var("i", t)
    func = relay.Function([i], p.nat_iterate(double, make_nat_expr(p, 3))(i))
    mod["main"] = func
    mod["main"] = to_cps(mod["main"], mod=mod)
    mod["main"] = un_cps(mod["main"])
    ex = create_executor(mod=mod)
    i_nd = rand(dtype, *shape)
    forward = ex.evaluate()(i_nd)
    tvm.testing.assert_allclose(forward.asnumpy(), 8 * i_nd.asnumpy())


# This serve as an integration test.
# It test that, given a program with reference,
# cps and pe can completely eliminate the allocation of reference.
def test_cps_pe():
    def destroy_ref(x):
        x = run_infer_type(x)
        x = to_cps(x)
        x = run_infer_type(x)
        y = un_cps(x)
        y = run_infer_type(y)
        x = run_opt_pass(x, tvm.transform.Sequential(
            [transform.PartialEvaluate(), transform.DeadCodeElimination(inline_once=True)]))
        assert Feature.fRefCreate not in detect_feature(x)
    unit = relay.Function([], relay.const(0., dtype='float32'))
    f_ref = relay.Var("f_ref")

    one = relay.const(1., dtype='float32')
    two = relay.const(2., dtype='float32')
    cond = relay.var(shape=(), dtype='uint1', name_hint='cond')
    true_branch = relay.RefWrite(f_ref, relay.Function([], one))
    false_branch = relay.RefWrite(f_ref, relay.Function([], two))
    if_expr = relay.If(cond, true_branch, false_branch)

    stmt = relay.Let(f_ref, relay.RefCreate(unit),
                     relay.Let(relay.Var("x"), if_expr,
                               relay.Call(relay.RefRead(f_ref), [])))

    F = relay.Function([cond], stmt)
    destroy_ref(F)

    G = relay.Function([cond], relay.If(cond, one, two))
    G = run_infer_type(G)
    G = relay.transform.gradient(G)
    destroy_ref(G)

    x = relay.var("x", shape=(1, 16))
    y = relay.var("y", shape=(1, 16))
    z = relay.var("z", shape=(1, 16))
    cond = relay.var("cond", shape=(), dtype='uint1')
    H = relay.If(cond, x, y)
    H = relay.add(H, z)
    H = relay.Function([cond,x,y,z], H)
    H = run_infer_type(H)
    H = relay.transform.gradient(H)
    destroy_ref(H)


if __name__ == '__main__':
    test_recursion()
    test_cps_pe()
