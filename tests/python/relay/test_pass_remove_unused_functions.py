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
from tvm.relay import transform
from tvm.relay.prelude import Prelude

def test_remove_all_prelude_functions():
    mod = relay.Module()
    p = Prelude(mod)
    x = relay.var("x", shape=(1, 16))
    mod["main"] = relay.Function([x], x)
    mod = relay.transform.RemoveUnusedFunctions()(mod)
    # Keep: main
    assert len(mod.functions) == 1

def test_remove_all_prelude_functions_but_referenced_functions():
    mod = relay.Module()
    p = Prelude(mod)
    x = relay.var("x", shape=(1, 16))
    id_func = relay.Function([x], x)
    id_name = relay.GlobalVar('id_func')
    mod[id_name] = id_func

    mod["main"] = relay.Function([x], id_name(x))
    mod = relay.transform.RemoveUnusedFunctions()(mod)
    # Keep: id_func, main
    assert len(mod.functions) == 2

def test_keep_only_referenced_prelude_functions():
    mod = relay.Module()
    p = Prelude(mod)
    l = p.nil()
    for i in [4, 3, 2, 1, 0]:
        l = p.cons(relay.const(i), l)
    body = p.hd(p.tl(p.tl(l)))
    mod["main"] = relay.Function([], body)
    mod = relay.transform.RemoveUnusedFunctions()(mod)
    # Keep: hd, tl, main
    assert len(mod.functions) == 3

if __name__ == '__main__':
    pytest.main()
