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
from tvm import te
from tvm import relay
from tvm.relay import transform
from tvm.relay.prelude import Prelude


def test_remove_all_prelude_functions():
    mod = tvm.IRModule()
    p = Prelude(mod)
    x = relay.var("x", shape=(1, 16))
    mod["main"] = relay.Function([x], x)
    mod = relay.transform.RemoveUnusedFunctions()(mod)
    l = set([x[0].name_hint for x in mod.functions.items()])
    assert l == set(["main"])


def test_remove_all_prelude_functions_but_referenced_functions():
    mod = tvm.IRModule()
    p = Prelude(mod)
    x = relay.var("x", shape=(1, 16))
    id_func = relay.Function([x], x)
    id_name = relay.GlobalVar("id_func")
    mod[id_name] = id_func

    mod["main"] = relay.Function([x], id_name(x))
    mod = relay.transform.RemoveUnusedFunctions()(mod)
    l = set([x[0].name_hint for x in mod.functions.items()])
    assert l == set(["id_func", "main"])


def test_keep_only_referenced_prelude_functions():
    mod = tvm.IRModule()
    p = Prelude(mod)
    _, cons, nil = p.mod.get_type("List")
    hd = p.mod.get_global_var("hd")
    tl = p.mod.get_global_var("tl")
    l = nil()
    for i in [4, 3, 2, 1, 0]:
        l = cons(relay.const(i), l)
    body = hd(tl(tl(l)))
    mod["main"] = relay.Function([], body)
    mod = relay.transform.RemoveUnusedFunctions()(mod)
    l = set([x[0].name_hint for x in mod.functions.items()])
    assert l == set(["tl", "hd", "main"])


def test_multiple_entry_functions():
    mod = tvm.IRModule()
    p = Prelude(mod)
    _, cons, nil = p.mod.get_type("List")
    hd = p.mod.get_global_var("hd")
    tl = p.mod.get_global_var("tl")
    l = nil()
    for i in [4, 3, 2, 1, 0]:
        l = cons(relay.const(i), l)
    body = hd(tl(tl(l)))
    mod["main1"] = relay.Function([], body)

    x = relay.var("x", shape=(1, 16))
    id_func = relay.Function([x], x)
    id_name = relay.GlobalVar("id_func")
    mod[id_name] = id_func
    mod["main2"] = relay.Function([x], id_name(x))
    mod = relay.transform.RemoveUnusedFunctions(["main1", "main2"])(mod)
    l = set([x[0].name_hint for x in mod.functions.items()])
    assert l == set(["tl", "hd", "main2", "id_func", "main1"])


def test_globalvar_as_call_arg():
    mod = tvm.IRModule()
    p = Prelude(mod)
    tensor_array = p.get_global_var("tensor_array", "int32")
    tensor1 = p.get_ctor(p.get_name("tensor_t", "int32"), "tensor1", "int32")
    write = p.get_global_var("tensor_array_write", "int32")
    stack = p.get_global_var("tensor_array_stack", "int32")
    v = relay.var("v")
    init_tensor_array = tensor_array(relay.const(3))
    tensor_array1 = write(init_tensor_array, relay.const(0), tensor1(v))
    tensor_array2 = stack(tensor_array1)
    mod["main"] = relay.Function([v], tensor_array2)
    mod = relay.transform.RemoveUnusedFunctions()(mod)
    l = set([x[0].name_hint for x in mod.functions.items()])
    assert "tensor_array_int32" in l


def test_call_globalvar_without_args():
    def get_mod():
        mod = tvm.IRModule({})
        fn1 = relay.Function([], relay.const(1))
        fn2 = relay.Function([], relay.const(2))
        g1 = relay.GlobalVar("g1")
        g2 = relay.GlobalVar("g2")
        mod[g1] = fn1
        mod[g2] = fn2
        p = relay.var("p", "bool")
        mod["main"] = relay.Function([p], relay.Call(relay.If(p, g1, g2), []))
        return mod

    mod = get_mod()
    ref_mod = get_mod()
    mod = relay.transform.RemoveUnusedFunctions()(mod)
    assert tvm.ir.structural_equal(mod, ref_mod, map_free_vars=True)


if __name__ == "__main__":
    tvm.testing.main()
