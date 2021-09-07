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
# pylint: disable=invalid-name
"""Utilities for testing and benchmarks"""
from __future__ import absolute_import as _abs
import collections
import numpy as np

import tvm
from tvm import te
from tvm import relay
from tvm.relay import op
from tvm.relay.prelude import Prelude
from tvm.testing import enabled_targets

from . import mlp
from . import resnet
from . import resnet_3d
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
from . import synthetic

from .init import create_workload
from .nat import count, make_nat_value, make_nat_expr
from .py_converter import to_python, run_as_python
from ..transform import gradient


def run_opt_pass(expr, opt_pass, import_prelude=False):
    assert isinstance(opt_pass, tvm.transform.Pass)
    mod = tvm.IRModule.from_expr(expr)
    if import_prelude:
        Prelude(mod)
    mod = relay.transform.InferType()(mod)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def run_infer_type(expr):
    return run_opt_pass(expr, relay.transform.InferType())


def _np_randn_from_type(t, scale=1, mean=0):
    res = mean + (scale * np.random.randn(*(int(d) for d in t.shape)))
    # if t.shape == (), then randn returns a scalar so we need to wrap for dtype conversion
    if np.isscalar(res):
        res = np.array(res)
    return res.astype(t.dtype)


def check_grad(
    func,
    inputs=None,
    test_inputs=None,
    eps=1e-6,
    atol=1e-5,
    rtol=1e-3,
    scale=None,
    mean=0,
    mode="higher_order",
    target_devices=None,
):
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

    test_inputs: List[np.array]
        The inputs to test for gradient matching. Useful in cases where some inputs are not
        differentiable, such as symbolic inputs to dynamic ops. If not given, all inputs are
        tested.

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

    target_devices: Optional[List[Tuple[tvm.target.Target, tvm.runtime.Device]]]
        A list of targets/devices on which the gradient should be
        tested.  If not specified, will default to `tvm.testing.enabled_targets()`.

    """

    fwd_func = run_infer_type(func)
    bwd_func = run_infer_type(gradient(fwd_func, mode=mode))

    if scale is None:
        scale = 10 * eps

    if inputs is None:
        params = fwd_func.params
        # Generate random inputs on the same scale as epsilon to avoid numerical precision loss.
        inputs = [_np_randn_from_type(x.checked_type, scale=scale, mean=mean) for x in params]

    if test_inputs is None:
        test_inputs = inputs

    if target_devices is None:
        target_devices = enabled_targets()

    for target, dev in target_devices:
        # Eval the backward and forward functions
        # TODO(mbs): Evaluate a pair of functions so can share preparation between them.
        bwd_func_compiled = relay.create_executor(device=dev, target=target).evaluate(bwd_func)
        fwd_func_compiled = relay.create_executor(device=dev, target=target).evaluate(fwd_func)

        # Get analytic gradients.
        _, grads = bwd_func_compiled(*inputs)
        grads = [grad.numpy().astype("float64") for grad in grads]

        # Throw out gradients we aren't testing
        if inputs != test_inputs:
            tmp = []
            # find the gradient that corresponds to every test input
            for test_input in test_inputs:
                for i, grad in enumerate(grads):
                    if inputs[i] is test_input:
                        tmp.append(grad)
                        break
            grads = tmp

        assert len(grads) > 0, "You must test at least one gradient."

        # Get numeric gradients for each dimension of each param, using two-sided approximation.
        approx_grads = []
        for x in test_inputs:
            approx_grad = np.zeros(x.shape)
            for i in np.ndindex(*x.shape):
                x_i = x[i]
                x[i] = x_i + eps
                fwd_plus = fwd_func_compiled(*inputs).numpy().astype("float64")
                x[i] = x_i - eps
                fwd_minus = fwd_func_compiled(*inputs).numpy().astype("float64")
                x[i] = x_i
                approx_grad[i] = np.sum((fwd_plus - fwd_minus) / (2 * eps))
            approx_grads.append(approx_grad)
        # Compare gradients by checking that relative difference is below tolerance.
        for grad, approx_grad in zip(grads, approx_grads):
            np.testing.assert_allclose(grad, approx_grad, atol=atol, rtol=rtol)


def rand(dtype, *shape):
    return tvm.nd.array(np.random.rand(*shape).astype(dtype))


def count_ops(expr):
    """count number of times a given op is called in the graph"""

    class OpCounter(tvm.relay.ExprVisitor):
        """OpCounter"""

        def visit_call(self, call):
            if hasattr(call, "op"):
                self.node_counter[call.op.name] += 1
            return super().visit_call(call)

        def count(self, expr):
            self.node_set = {}
            self.node_counter = collections.Counter()
            self.visit(expr)
            return self.node_counter

    return OpCounter().count(expr)
