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
from tvm import te
from tvm import relay
from tvm.relay.analysis import well_formed
from tvm.relay.prelude import Prelude


def test_let():
    x = relay.Var("x")
    assert well_formed(x)
    v = relay.Constant(tvm.nd.array(10))
    ty = None
    let = relay.Let(x, v, x)
    assert well_formed(let)
    assert not well_formed(relay.Let(x, v, let))
    f = relay.Function([x], x, ty)
    assert well_formed(f)
    assert well_formed(relay.Let(relay.Var("y"), f, relay.Let(relay.Var("z"), f, v)))


def test_tuple():
    x = relay.Var("x")
    assert well_formed(x)
    v = relay.Constant(tvm.nd.array(10))
    let = relay.Let(x, v, x)
    assert well_formed(let)
    assert well_formed(relay.Tuple([v, v]))
    assert not well_formed(relay.Tuple([let, relay.Let(x, v, x)]))


def test_tuple_get_item():
    t = relay.Var("t")
    assert well_formed(relay.TupleGetItem(t, 2))


def test_adt():
    mod = tvm.IRModule()
    p = Prelude(mod)
    _, none, some = p.mod.get_type("Option")
    x = relay.Var("x")
    some_case = relay.Clause(relay.PatternConstructor(some, [relay.PatternVar(x)]), x)
    default_case = relay.Clause(relay.PatternVar(x), x)
    m0 = relay.Match(none(), [default_case])
    m1 = relay.Match(none(), [some_case, default_case])
    assert well_formed(m0)
    assert not well_formed(m1)


if __name__ == "__main__":
    test_let()
    test_tuple()
    test_tuple_get_item()
    test_adt()
