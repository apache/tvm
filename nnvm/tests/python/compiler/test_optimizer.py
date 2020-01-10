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
import nnvm
import nnvm.compiler.optimizer as optimizer
import nnvm.compiler.lr_scheduler as lr_scheduler

from nnvm.testing.config import ctx_list
from tvm.contrib import graph_runtime


def helper(symbol, inputs, params, update_func, run_times, target, ctx, dtype="float32"):
    ishapes = {}
    np_inputs = {}
    params_dict = {}
    for (name, shape, s) in inputs:
        ishapes.update({name: shape})
        np_inputs.update({name: np.random.uniform(size=shape).astype(dtype)})
    for (name, shape, s) in params:
        np_inputs.update({name: np.random.uniform(size=shape).astype(dtype)})
        params_dict.update({name: np_inputs[name]})

    graph, lib, rt_params = nnvm.compiler.build(symbol, target, shape=ishapes)
    m = graph_runtime.create(graph, lib, ctx)
    m.set_input(**np_inputs)
    m.set_input(**rt_params)
    for _ in range(run_times):
        m.run()
    y_np = update_func(**np_inputs)
    out = m.get_output(0, tvm.nd.empty(y_np.shape, dtype))
    tvm.testing.assert_allclose(out.asnumpy(), y_np, atol=1e-5, rtol=1e-5)


def test_sgd():
    for target, ctx in ctx_list():
        data = nnvm.sym.Variable("data")
        weight = nnvm.sym.Variable("weight")
        out = nnvm.sym.elemwise_mul(data, weight ** 2)

        dshape = (1, 2, 3)
        wshape = dshape

        base_lr = 0.1
        lr_factor = 0.5
        rescale_grad = 0.2
        wd = 0.1
        clip_gradient = 0.25

        scheduler = lr_scheduler.FactorScheduler(base_lr=base_lr, step=1, factor=lr_factor)
        opt = optimizer.SGD(learning_rate=base_lr, lr_scheduler=scheduler,
                            rescale_grad=rescale_grad, clip_gradient=clip_gradient,
                            wd=wd)
        opt_sym = opt.minimize(out, var=weight)

        inputs = [("data", dshape, data)]
        params = [("weight", wshape, weight)]

        def update_func(data, weight):
            gradient_0 = data * 2 * weight * rescale_grad
            gradient_0 = np.clip(gradient_0, -clip_gradient, clip_gradient)
            weight_0 = weight - base_lr * lr_factor * (gradient_0 + wd * weight)
            gradient_1 = data * 2 * weight_0 * rescale_grad
            gradient_1 = np.clip(gradient_1, -clip_gradient, clip_gradient)
            weight_1 = weight_0 - base_lr * (lr_factor ** 2) * (gradient_1 + wd * weight_0)
            return weight_1

        helper(opt_sym, inputs, params, update_func, 2, target, ctx)



def test_adam():
    for target, ctx in ctx_list():
        data = nnvm.sym.Variable("data")
        weight = nnvm.sym.Variable("weight")
        out = nnvm.sym.elemwise_mul(data, weight ** 2)

        dshape = (1, 2, 3)
        wshape = dshape

        base_lr = 0.1
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        lr_factor = 0.5
        rescale_grad = 0.2
        wd = 0.1
        clip_gradient = 0.25

        scheduler = lr_scheduler.FactorScheduler(base_lr=base_lr, step=1, factor=lr_factor)
        opt = optimizer.Adam(learning_rate=base_lr, beta1=beta1, beta2=beta2, epsilon=epsilon,
                             lr_scheduler=scheduler, rescale_grad=rescale_grad,
                             clip_gradient=clip_gradient, wd=wd)
        opt_sym = opt.minimize(out, var=weight)

        inputs = [("data", dshape, data)]
        params = [("weight", wshape, weight)]

        def update_func(data, weight):
            rate_0 = np.sqrt(1 - beta2) / (1 - beta1)
            lr_0 = base_lr * lr_factor * rate_0
            gradient_0 = data * 2 * weight * rescale_grad
            gradient_0 = np.clip(gradient_0, -clip_gradient, clip_gradient)
            m_0 = (1 - beta1) * gradient_0
            v_0 = (1 - beta2) * (gradient_0 ** 2)
            weight_0 = weight - lr_0 * (m_0 / (np.sqrt(v_0) + epsilon) + wd * weight)
            rate_1 = np.sqrt(1 - beta2 ** 2) / (1 - beta1 ** 2)
            lr_1 = base_lr * (lr_factor ** 2) * rate_1
            gradient_1 = data * 2 * weight_0 * rescale_grad
            gradient_1 = np.clip(gradient_1, -clip_gradient, clip_gradient)
            m_1 = beta1 * m_0 + (1 - beta1) * gradient_1
            v_1 = beta2 * v_0 + (1 - beta2) * (gradient_1 ** 2)
            weight_1 = weight_0 - lr_1 * (m_1 / (np.sqrt(v_1) + epsilon) + wd * weight_0)
            return weight_1

        helper(opt_sym, inputs, params, update_func, 2, target, ctx)

if __name__ == "__main__":
    test_sgd()
    test_adam()
