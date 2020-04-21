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
#pylint: disable=invalid-name
"""Utilities for testing and benchmarks"""
from __future__ import absolute_import as _abs
import numpy as np

import tvm
from tvm import te
import tvm.relay as relay
import tvm.relay.op as op


from . import mlp
from . import resnet
from . import dqn
from . import dcgan
from . import mobilenet
from . import lstm
from . import inception_v3
from . import squeezenet
from . import vgg
from . import densenet
from . import yolo_detection
from . import temp_op_attr

from .config import ctx_list
from .init import create_workload
from .nat import add_nat_definitions, count, make_nat_value, make_nat_expr
from .py_converter import to_python, run_as_python
from ..transform import gradient

def run_opt_pass(expr, opt_pass):
    assert isinstance(opt_pass, tvm.transform.Pass)
    mod = tvm.IRModule.from_expr(expr)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def run_infer_type(expr):
    return run_opt_pass(expr, relay.transform.InferType())


def _np_randn_from_type(t, scale=1, mean=0):
    return (mean + (scale * np.random.randn(*(int(d) for d in t.shape)))).astype(t.dtype)


def check_grad(func, inputs=None, eps=1e-6, atol=1e-5, rtol=1e-3, scale=None, mean=0):
    """Perform numerical gradient checking given a relay function.

    Compare analytical gradients to numerical gradients derived from two-sided approximation. Note
    that this test may fail if your function input types are not of high enough precision.

    Parameters
    ----------
    func : tvm.relay.Function
        The relay function to test.

    inputs: List[np.array]
        Optional user-provided input parameters to use. If not given, will generate random normal
        inputs scaled to be close to the chosen epsilon value to avoid numerical precision loss.

    eps: float
        The epsilon value to use for computing numerical gradient approximation.

    atol: float
        The absolute tolerance on difference between numerical and analytical gradients. Note that
        this needs to be scaled appropriately relative to the chosen eps and inputs.

    rtol: float
        The relative tolerance on difference between numerical and analytical gradients. Note that
        this needs to be scaled appropriately relative to the chosen eps.

    scale: float
        The standard deviation of the inputs.

    mean: float
        The mean of the inputs.
    """

    fwd_func = run_infer_type(func)
    bwd_func = run_infer_type(gradient(fwd_func))

    if scale is None:
        scale = 10 * eps

    if inputs is None:
        params = fwd_func.params
        # Generate random inputs on the same scale as epsilon to avoid numerical precision loss.
        inputs = [_np_randn_from_type(x.checked_type, scale=scale, mean=mean) for x in params]

    for target, ctx in ctx_list():
        intrp = relay.create_executor(ctx=ctx, target=target)

        # Get analytic gradients.
        _, grads = intrp.evaluate(bwd_func)(*inputs)
        grads = [grad.asnumpy().astype("float64") for grad in grads]

        # Get numeric gradients for each dimension of each param, using two-sided approximation.
        approx_grads = []
        for x in inputs:
            approx_grad = np.zeros(x.shape)
            for i in np.ndindex(*x.shape):
                x_i = x[i]
                x[i] = x_i + eps
                fwd_plus = intrp.evaluate(fwd_func)(*inputs).asnumpy().astype("float64")
                x[i] = x_i - eps
                fwd_minus = intrp.evaluate(fwd_func)(*inputs).asnumpy().astype("float64")
                x[i] = x_i
                approx_grad[i] = np.sum((fwd_plus - fwd_minus) / (2 * eps))
            approx_grads.append(approx_grad)

        # Compare gradients by checking that relative difference is below tolerance.
        for grad, approx_grad in zip(grads, approx_grads):
            np.testing.assert_allclose(grad, approx_grad, atol=atol, rtol=rtol)


def rand(dtype, *shape):
    return tvm.nd.array(np.random.rand(*shape).astype(dtype))
