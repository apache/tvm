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
from tvm import te
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing import run_infer_type, create_workload, ctx_list


def run_opt_pass(expr, opt_pass):
    assert isinstance(opt_pass, tvm.transform.Pass)

    mod = tvm.IRModule.from_expr(expr)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body

def verify_func(func, data, ref_res):
    assert isinstance(data, list)
    for target, ctx in ctx_list():
        #TODO(mbrookhart): enable Cuda tests onces the VM supports dynamic shapes
        if "llvm" not in target: continue
        for kind in ["graph", "vm", "debug"]:
            mod = tvm.ir.IRModule.from_expr(func)
            intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
            op_res = intrp.evaluate()(*data)
            tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)

def test_dynamic_to_static_reshape():
    def verify_reshape(shape, newshape, oshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.var("y", relay.TensorType(newshape, "float32"))
        z = relay.dynamic.reshape(x, relay.shape_of(y))
        func = run_infer_type(relay.Function([x, y], z))
        func2 = run_opt_pass(run_opt_pass(func, transform.DynamicToStatic()), transform.InferType())

        zz = func2.body
        assert isinstance(zz, relay.Call)
        assert zz.op == relay.op.get("reshape")
        assert "newshape=" in zz.astext()
        assert zz.checked_type == relay.ty.TensorType(oshape, "float32")

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        y_data = np.random.uniform(low=-1, high=1, size=newshape).astype("float32")
        ref_res = np.reshape(x_data, oshape)
        verify_func(func2, [x_data, y_data], ref_res)

    verify_reshape((2, 3, 4), (8, 3), (8, 3))
    verify_reshape((4, 7), (2, 7, 2), (2, 7, 2))

def test_dynamic_to_static_double_reshape():
    def verify_reshape(shape, newshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.var("y", relay.TensorType(newshape, "float32"))
        z = relay.dynamic.reshape(x, relay.shape_of(y))
        z = relay.dynamic.reshape(z, relay.shape_of(x))
        func = run_infer_type(relay.Function([x, y], z))
        func2 = run_opt_pass(run_opt_pass(func, transform.DynamicToStatic()), transform.InferType())

        zz = func2.body
        assert isinstance(zz, relay.Call)
        assert zz.op == relay.op.get("reshape")
        assert "newshape=" in zz.astext()
        assert zz.checked_type == relay.ty.TensorType(shape, "float32")

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        y_data = np.random.uniform(low=-1, high=1, size=newshape).astype("float32")
        verify_func(func2, [x_data, y_data], x_data)

    verify_reshape((2, 3, 4), (8, 3))
    verify_reshape((4, 7), (2, 7, 2))

def test_dynamic_to_static_quad_reshape():
    def verify_reshape(shape, newshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.var("y", relay.TensorType(newshape, "float32"))
        z1 = relay.dynamic.reshape(x, relay.shape_of(y))
        z2 = relay.dynamic.reshape(z1, relay.shape_of(x))
        z3 = relay.dynamic.reshape(z2, relay.shape_of(z1))
        z4 = relay.dynamic.reshape(z3, relay.shape_of(z2))
        func = run_infer_type(relay.Function([x, y], z4))
        func2 = run_opt_pass(run_opt_pass(func, transform.DynamicToStatic()), transform.InferType())

        zz = func2.body
        assert isinstance(zz, relay.Call)
        assert zz.op == relay.op.get("reshape")
        assert "newshape=" in zz.astext()
        assert zz.checked_type == relay.ty.TensorType(shape, "float32")

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        y_data = np.random.uniform(low=-1, high=1, size=newshape).astype("float32")
        verify_func(func2, [x_data, y_data], x_data)

    verify_reshape((2, 3, 4), (8, 3))
    verify_reshape((4, 7), (2, 7, 2))

if __name__=="__main__":
    test_dynamic_to_static_reshape()
    test_dynamic_to_static_double_reshape()
    test_dynamic_to_static_quad_reshape()

