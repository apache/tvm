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
"""
Support level10 operator test cases.

"""

import numpy as np
import tvm
from tvm import relay
from tvm.relay.testing import run_infer_type
import tvm.topi.testing
import random
import tvm.testing

executor_kind = tvm.testing.parameter("debug", "vm")


@tvm.testing.uses_gpu
def test_broadcast_to(executor_kind):
    def verify_more_dynamic_broadcast_to(x_shape, out_shape):
        rank = len(out_shape)
        dtype = "float32"
        shape_type = "int64"
        reshape_shape = relay.Var("shape", relay.ty.TensorType((len(x_shape),), shape_type))
        broadcast_shape = relay.Var("shape", relay.ty.TensorType((rank,), shape_type))
        x = relay.Var("x", relay.ty.TensorType((np.prod(x_shape),), dtype))
        r = relay.reshape(x, reshape_shape)
        z = relay.broadcast_to(r, broadcast_shape)

        func = relay.Function([x, reshape_shape, broadcast_shape], z)

        x = np.random.uniform(size=np.prod(x_shape)).astype(dtype)
        ref_res = np.broadcast_to(np.reshape(x, x_shape), out_shape)
        for target, dev in tvm.testing.enabled_targets():
            mod = tvm.ir.IRModule.from_expr(func)
            op_res = relay.create_executor(
                executor_kind, mod=mod, device=dev, target=target
            ).evaluate(func)(
                x, np.array(x_shape).astype(shape_type), np.array(out_shape).astype(shape_type)
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

    verify_more_dynamic_broadcast_to((4, 3), (3, 4, 3))

    def verify_broadcast_to(x_shape, out_shape):
        rank = len(out_shape)
        dtype = "float32"
        shape_type = "int64"
        dyn_shape = relay.Var("shape", relay.ty.TensorType((rank,), shape_type))
        x = relay.Var("x", relay.ty.TensorType(x_shape, dtype))
        z = relay.broadcast_to(x, dyn_shape)
        zz = run_infer_type(z)

        assert zz.checked_type == relay.ty.TensorType((relay.Any(),) * rank, dtype)

        func = relay.Function([x, dyn_shape], z)

        x = np.random.uniform(size=x_shape).astype(dtype)
        ref_res = np.broadcast_to(x, out_shape)
        for target, dev in tvm.testing.enabled_targets():
            mod = tvm.ir.IRModule.from_expr(func)
            op_res = relay.create_executor(
                executor_kind, mod=mod, device=dev, target=target
            ).evaluate(func)(x, np.array(out_shape).astype(shape_type))
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

    verify_broadcast_to((1,), (1, 1, 1))
    verify_broadcast_to((1, 1), (4, 1, 1))
    verify_broadcast_to((4, 1), (1, 4, 3))


@tvm.testing.uses_gpu
def test_dyn_broadcast_to(executor_kind):
    dtype = "uint8"
    rank = 3
    shape_type = "int64"
    dyn_shape = relay.Var("shape", relay.ty.TensorType((rank,), shape_type))
    x_shape = (1,)
    x = relay.Var("x", relay.ty.TensorType(x_shape, dtype))
    z = relay.broadcast_to(x, dyn_shape)
    zz = run_infer_type(z)

    assert zz.checked_type == relay.ty.TensorType((relay.Any(),) * rank, dtype)

    func = relay.Function([x, dyn_shape], z)

    x = np.random.uniform(size=x_shape).astype(dtype)
    dyn_shape = (1,) * rank
    ref_res = np.broadcast_to(x, dyn_shape)
    for target, dev in tvm.testing.enabled_targets():
        mod = tvm.ir.IRModule.from_expr(func)
        op_res = relay.create_executor(executor_kind, mod=mod, device=dev, target=target).evaluate(
            func
        )(x, np.array(dyn_shape).astype(shape_type))
        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)


@tvm.testing.uses_gpu
def test_dyn_one_hot(executor_kind):
    def _get_oshape(indices_shape, depth, axis):
        oshape = []
        true_axis = len(indices_shape) if axis == -1 else axis
        ndim = len(indices_shape) + 1
        indices_index = 0
        for i in range(0, ndim):
            if i == true_axis:
                oshape.append(depth)
            else:
                oshape.append(indices_shape[indices_index])
                indices_index += 1

        return oshape

    def _verify(indices_shape, depth, on_value, off_value, axis, dtype):
        indices = relay.var("indices", relay.TensorType(indices_shape, "int32"))
        depth_var = relay.var("depth", relay.TensorType((), "int32"))
        on_value_const = relay.const(on_value)
        off_value_const = relay.const(off_value)
        out = relay.one_hot(indices, on_value_const, off_value_const, depth_var, axis, dtype)
        func = relay.Function([indices, depth_var], out)
        indices_np = np.random.randint(0, depth, size=indices_shape).astype("int32")
        out_np = tvm.topi.testing.one_hot(indices_np, on_value, off_value, depth, axis, dtype)
        for target, dev in tvm.testing.enabled_targets():
            mod = tvm.ir.IRModule.from_expr(func)
            out_relay = relay.create_executor(
                executor_kind, mod=mod, device=dev, target=target
            ).evaluate()(indices_np, np.array(depth).astype("int32"))
            tvm.testing.assert_allclose(out_relay.numpy(), out_np)

    _verify((3,), 3, 1, 0, -1, "int32")
    _verify((3,), 3, 1.0, 0.0, -1, "float32")
    _verify((2, 2), 5, 2, -2, 0, "int32")
    _verify((2, 2), 5, 0.5, -0.5, 1, "float32")
    _verify((3, 2, 4, 5), 6, 1, 0, 1, "int32")
    _verify((3, 2, 4, 5), 6, 1.0, 0.0, 0, "float32")


if __name__ == "__main__":
    tvm.testing.main()
