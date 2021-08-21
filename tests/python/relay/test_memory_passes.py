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
from tvm import te
import numpy as np
from tvm import relay


def check_memory_plan(func, check_fn):
    # Build Module
    mod = tvm.IRModule().from_expr(func)

    # Convert arguments.
    args = []
    for param in func.params:
        param = param.type_annotation
        sh = [int(sh) for sh in param.shape]
        data = np.random.rand(*sh).astype(param.dtype)
        args.append(tvm.nd.array(data))

    # TODO(mbs): Why does the executor need to be shared? Seems wrong.
    ex = relay.create_executor("vm", mod)

    # Compute without memory planning.
    no_plan_result = ex.evaluate()(*args)

    # Compute with memory planning.
    with tvm.transform.PassContext(opt_level=1, disabled_pass=["MemoryPlan"]):
        plan_result = ex.evaluate()(*args)

    # Compute Python result.
    py_res = check_fn(*[arg.numpy() for arg in args])

    # First check that the two VM results agree.
    np.testing.assert_allclose(no_plan_result.numpy(), plan_result.numpy())

    # Finally check that the results match the Python result.
    np.testing.assert_allclose(plan_result.numpy(), py_res)


def storage_type(mod):
    return relay.TypeCall(mod.get_global_type_var("Storage"), [])


def test_tyck_alloc_storage():
    mod = tvm.IRModule()
    mod.import_from_std("core.rly")


def test_tyck_alloc_tensor():
    mod = tvm.IRModule()
    mod.import_from_std("core.rly")
    sto = relay.Var("x", storage_type(mod))
    sh = relay.const(np.array([1, 2]), dtype="int64")
    at = relay.op.memory.alloc_tensor(sto, relay.const(0, dtype="int64"), sh)
    mod["main"] = relay.Function([sto], at)
    relay.transform.InferType()(mod)


def check_add(x):
    return x + x


def test_add():
    x = relay.var("x", shape=(2,))
    z = x + x
    func = relay.Function(
        [
            x,
        ],
        z,
    )
    check_memory_plan(func, check_add)


def check_add_sub(x, y):
    z = x + x
    return z - y


def test_add_sub():
    x = relay.var("x", shape=(10,))
    y = relay.var("y", shape=(10,))
    z = x + x
    z = z - y
    func = relay.Function([x, y], z)
    check_memory_plan(func, check_add_sub)


def check_no_fuse(x, y, w):
    z = x + y
    return np.matmul(z, np.transpose(w))


def test_no_fuse():
    x = relay.var("x", shape=(5, 1))
    y = relay.var("y", shape=(5, 1))
    w = relay.var("w", shape=(5, 1))
    z = x + y
    out = relay.op.nn.dense(z, w)
    func = relay.Function([x, y, w], out)
    check_memory_plan(func, check_no_fuse)


if __name__ == "__main__":
    test_tyck_alloc_tensor()
    test_add()
    test_add_sub()
