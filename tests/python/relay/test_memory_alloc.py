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
# under the License
import tvm
import numpy as np
from tvm import relay
from tvm.relay import memory_alloc

def check_vm_alloc(func, check_fn):
    mod = relay.Module()
    mod['main'] = func
    ex = relay.create_executor('vm', mod)
    args = []
    for param in func.params:
        param = param.type_annotation
        sh = [int(sh) for sh in param.shape]
        data = np.random.rand(*sh).astype(param.dtype)
        args.append(tvm.nd.array(data))
    result = ex.evaluate(mod['main'])(*args)
    py_res = check_fn(*[arg.asnumpy() for arg in args])
    np.testing.assert_allclose(result.asnumpy(), py_res)

def storage_type(mod):
    return relay.TypeCall(mod.get_global_type_var("Storage"), [])

def test_tyck_alloc_storage():
    mod = relay.Module()
    mod.import_from_std("core.rly")

def test_tyck_alloc_tensor():
    mod = relay.Module()
    mod.import_from_std("core.rly")
    sto = relay.Var("x", storage_type(mod))
    sh = relay.const(np.array([1, 2]), dtype="int64")
    at = relay.op.memory.alloc_tensor(sto, sh)
    mod['main'] = relay.Function([sto], at)
    relay.transform.InferType()(mod)


def check_add(x):
    return x + x

def test_add():
    x = relay.var('x', shape=(2,))
    z = x + x
    func = relay.Function([x,], z)
    check_vm_alloc(func, check_add)


def check_add_sub(x, y):
    z = x + x
    return z - y

def test_add_sub():
    x = relay.var('x', shape=(10,))
    y = relay.var('y', shape=(10,))
    z = x + x
    z = z - y
    func = relay.Function([x, y], z)
    check_vm_alloc(func, check_add_sub)

if __name__ == "__main__":
    test_tyck_alloc_tensor()
    test_add()
    test_add_sub()
