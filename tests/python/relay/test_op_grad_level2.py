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
import topi
import topi.testing
from tvm import relay
from tvm.relay.transform import gradient
from tvm.relay.testing import ctx_list
from tvm.relay.testing import run_infer_type


def verify_max_pool2d_grad(x_shape, pool_size, strides, padding, ceil_mode):
    x = relay.var("x", relay.TensorType(x_shape, "float32"))
    y = tvm.relay.nn.max_pool2d(x, pool_size=pool_size, strides=strides, padding=padding,
                                ceil_mode=ceil_mode)

    fwd_func = relay.Function([x], y)
    fwd_func = run_infer_type(fwd_func)
    bwd_func = run_infer_type(gradient(fwd_func))

    data = np.random.rand(*x_shape).astype("float32")
    ph, pw = padding
    y_shape = topi.util.get_const_tuple(fwd_func.ret_type.shape)
    out_grad = np.ones(shape=y_shape)
    ref_grad = topi.testing.pool_grad_nchw(data, out_grad, pool_size=pool_size, strides=strides,
                                           padding=[ph, pw, ph, pw],
                                           pool_type='max', ceil_mode=ceil_mode)

    for target, ctx in ctx_list():
        intrp = relay.create_executor(ctx=ctx, target=target)
        op_res, (op_grad, ) = intrp.evaluate(bwd_func)(data)
        np.testing.assert_allclose(op_grad.asnumpy(), ref_grad, rtol=0.01)


def test_max_pool2d_grad():
    verify_max_pool2d_grad((1, 4, 16, 16), pool_size=(2, 2), strides=(2, 2), padding=(0, 0),
                           ceil_mode=False)
    verify_max_pool2d_grad((1, 4, 16, 16), pool_size=(1, 1), strides=(1, 1), padding=(1, 1), ceil_mode=False)


def verify_avg_pool2d_grad(x_shape, pool_size, strides, padding, ceil_mode, count_include_pad):
    x = relay.var("x", relay.TensorType(x_shape, "float32"))
    y = tvm.relay.nn.avg_pool2d(x, pool_size=pool_size, strides=strides, padding=padding,
                                ceil_mode=ceil_mode, count_include_pad=count_include_pad)

    fwd_func = relay.Function([x], y)
    fwd_func = run_infer_type(fwd_func)
    bwd_func = run_infer_type(gradient(fwd_func))

    data = np.random.rand(*x_shape).astype("float32")
    ph, pw = padding
    y_shape = topi.util.get_const_tuple(fwd_func.ret_type.shape)
    out_grad = np.ones(shape=y_shape)
    ref_grad = topi.testing.pool_grad_nchw(data, out_grad, pool_size=pool_size, strides=strides,
                                           padding=[ph, pw, ph, pw],
                                           pool_type='avg', ceil_mode=ceil_mode)

    for target, ctx in ctx_list():
        intrp = relay.create_executor(ctx=ctx, target=target)
        op_res, (op_grad, ) = intrp.evaluate(bwd_func)(data)
        np.testing.assert_allclose(op_grad.asnumpy(), ref_grad, rtol=0.01)


def test_avg_pool2d_grad():
    verify_avg_pool2d_grad((1, 4, 16, 16), pool_size=(2, 2), strides=(2, 2), padding=(0, 0),
                           ceil_mode=False, count_include_pad=True)
    verify_avg_pool2d_grad((1, 4, 16, 16), pool_size=(1, 1), strides=(1, 1), padding=(1, 1),
                           ceil_mode=False, count_include_pad=False)


if __name__ == "__main__":
    test_max_pool2d_grad()
    test_avg_pool2d_grad()
