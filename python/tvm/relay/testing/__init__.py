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

import tvm
import tvm.relay as relay
import tvm.relay.op as op
from tvm.relay import transform
from tvm.relay import Function, GlobalVar, ScopeBuilder, Tuple, TupleGetItem, create_executor
from tvm.relay import TensorType, TupleType
import numpy as np

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

from .config import ctx_list
from .init import create_workload
from .nat import add_nat_definitions, count, make_nat_value, make_nat_expr
from .py_converter import to_python, run_as_python
from ..transform import gradient

def run_opt_pass(expr, opt_pass):
    assert isinstance(opt_pass, transform.Pass)
    mod = relay.Module.from_expr(expr)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def run_infer_type(expr):
    return run_opt_pass(expr, transform.InferType())


def rand_from_type(t):
    return relay.Constant(rand(t.dtype, *[int(d) for d in t.shape]))


CHECK_GRAD_COUNTER = 0
def check_grad(func, mod=None):
    """
    Test that directional gradient calculated by reverse mode
    is close to the one calculated by finite difference.
    """
    global CHECK_GRAD_COUNTER
    if mod is None:
        mod = relay.Module()
    def make(name):
        return GlobalVar(name + str(CHECK_GRAD_COUNTER))
    func_name = make("func_")
    back_func_name = make("back_func_")
    finite_difference_func_name = make("finite_difference_")
    reverse_mode_func_name = make("reverse_mode_")
    check_func_name = make("check_func_")
    CHECK_GRAD_COUNTER = CHECK_GRAD_COUNTER + 1
    epsilon = relay.const(0.01)
    mod[func_name] = func
    mod[back_func_name] = gradient(mod[func_name], mod=mod)
    params = mod[func_name].params
    directions = [rand_from_type(x.checked_type) for x in params]
    ft = TensorType(())
    sb = ScopeBuilder()
    def get_reverse_mode_result(e, d, t):
        assert isinstance(t, TensorType)
        return op.cast(e * d, 'float32')
    bf = sb.let("bf", TupleGetItem(back_func_name(*params), 1))
    reverse_mode_results = [get_reverse_mode_result(TupleGetItem(bf, i),
                                                    directions[i],
                                                    x.checked_type)
                            for i, x in enumerate(params)]
    reverse_mode_result = relay.const(0.0)
    for x in reverse_mode_results:
        reverse_mode_result = reverse_mode_result + op.reduce.sum(x)
    sb.ret(reverse_mode_result)
    reverse_mode_result = sb.get()
    mod[reverse_mode_func_name] = Function(params,
                                           reverse_mode_result,
                                           ft,
                                           mod[func_name].type_params,
                                           mod[func_name].attrs)
    finite_difference_result = op.reduce.sum((func_name(*[x + epsilon * y for x, y in
                                                          zip(params, directions)]) -
                                              func_name(*params)) /
                                             epsilon)

    mod[finite_difference_func_name] = Function(params,
                                                finite_difference_result,
                                                ft,
                                                mod[func_name].type_params,
                                                mod[func_name].attrs)
    check_func_result = op.abs(reverse_mode_func_name(*params) -
                               finite_difference_func_name(*params))
    mod[check_func_name] = Function(params,
                                    check_func_result,
                                    ft,
                                    mod[func_name].type_params,
                                    mod[func_name].attrs)
    ex = create_executor(mod=mod)
    res = ex.evaluate(check_func_name(*[rand_from_type(x.checked_type) for x in params]))
    assert res.data.asnumpy() < 0.001

def rand(dtype, *shape):
    return tvm.nd.array(np.random.rand(*shape).astype(dtype))
