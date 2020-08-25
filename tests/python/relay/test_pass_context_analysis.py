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
# pylint: disable=no-else-return,invalid-name,len-as-condition,too-many-nested-blocks

import numpy as np

import tvm
from tvm import relay
from tvm.relay import expr as _expr, transform
from tvm.relay.analysis import context_analysis


def test_device_copy():
    if not tvm.runtime.enabled("cuda") or not tvm.gpu(0).exist:
        return

    mod = tvm.IRModule()
    x = relay.var("x", shape=(2, 3))
    copy = relay.op.device_copy(x, tvm.cpu(), tvm.gpu())
    out = copy + relay.const(np.random.rand(2, 3))
    glb_var = relay.GlobalVar("main")
    mod[glb_var] = relay.Function([x], out)
    ca = context_analysis(mod, tvm.cpu())
    ca.visit(mod[glb_var])
    ca_res = ca.results()

    for expr, dev in ca_res.items():
        if isinstance(expr, _expr.Call):
            assert dev.device_type == tvm.gpu().device_type
        elif isinstance(expr, _expr.Var):
            assert dev.device_type == tvm.cpu().device_type
        elif isinstance(expr, _expr.Constant):
            assert dev.device_type == tvm.gpu().device_type


def test_shape_func():
    pass


def test_vm_shape_of():
    pass


def test_alloc_storage():
    pass


def test_alloc_tensor():
    pass


def test_dynamic_input():
    if not tvm.runtime.enabled("cuda") or not tvm.gpu(0).exist:
        return
    mod = tvm.IRModule()
    dtype = "float32"
    data_shape = (relay.Any(), 4)
    tensor_type = relay.TensorType(data_shape, dtype)
    tuple_type = relay.TupleType([tensor_type, tensor_type])
    data0 = relay.var("d0", type_annotation=relay.TupleType([tuple_type, tensor_type]))
    data1 = relay.var("d1", shape=(relay.Any(), 4), dtype=dtype)
    data_tuple = relay.expr.TupleWrapper(data0, 2)
    nested_data_tuple = relay.expr.TupleWrapper(data_tuple[0], 2)
    y = nested_data_tuple[1] * data_tuple[1] + data1
    mod["main"] = relay.Function([data0, data1], y)
    compiler = relay.vm.VMCompiler()
    # mod, _ = compiler.optimize(mod, target="cuda")
    # ca = context_analysis(mod, tvm.cpu())

if __name__ == "__main__":
    pass
