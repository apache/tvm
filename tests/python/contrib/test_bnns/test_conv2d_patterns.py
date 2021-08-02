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
"""BNNS pattern detection check"""

import tvm
from tvm import relay
import numpy as np

from tvm.relay.op.contrib.bnns import partition_for_bnns

fp32 = "float32"


def partition(exp):
    """Apply BNNS specific partitioning transformation"""
    mod = tvm.IRModule.from_expr(exp)
    with tvm.transform.PassContext(opt_level=3):
        mod = partition_for_bnns(mod)
    return mod


def is_op_fused(func, op_name):
    is_fused = False

    def visit(op):
        if (
            isinstance(op, tvm.relay.function.Function)
            and op_name in op.attrs["PartitionedFromPattern"]
        ):
            nonlocal is_fused
            is_fused = True

    tvm.relay.analysis.post_order_visit(func.body, visit)
    return is_fused


def test_pattern_conv2d_with_bias_add():
    for axis in (1, 2):
        a = relay.var("a", shape=(2, 7, 8, 8), dtype=fp32)
        w = relay.const(np.random.uniform(-10, 10, (8, 7, 3, 3)).astype(fp32))
        res = relay.nn.conv2d(a, w, kernel_size=(3, 3), padding=(1, 1), channels=8, out_dtype=fp32)
        b = relay.const(np.random.uniform(-10, 10, 8).astype(fp32))
        res = relay.nn.bias_add(res, b, axis=axis)

        mod = partition(res)
        bias_is_fused = is_op_fused(mod["tvmgen_default_bnns_main_0"], "nn.bias_add")

        assert bias_is_fused if axis == 1 else not bias_is_fused


def test_pattern_conv2d_with_add():
    workloads = {8: False, (8, 1): False, (8, 1, 1): True, (1, 8, 1, 1): True}

    for b_shape, should_be_fused in workloads.items():
        a = relay.var("a", shape=(2, 7, 8, 8), dtype=fp32)
        w = relay.const(np.random.uniform(-10, 10, (8, 7, 3, 3)).astype(fp32))
        res = relay.nn.conv2d(a, w, kernel_size=(3, 3), padding=(1, 1), channels=8, out_dtype=fp32)
        b = relay.const(np.random.uniform(-10, 10, b_shape).astype(fp32))
        res = relay.add(res, b)

        mod = partition(res)
        bias_is_fused = is_op_fused(mod["tvmgen_default_bnns_main_0"], "add")

        assert bias_is_fused == should_be_fused


def test_pattern_conv2d_with_non_cons_weights():
    for const_weights in (True, False):
        a = relay.var("a", shape=(2, 7, 8, 8), dtype=fp32)
        if const_weights:
            w = relay.const(np.random.uniform(-10, 10, (8, 7, 3, 3)).astype(fp32))
        else:
            w = relay.var("w", shape=(8, 7, 3, 3), dtype=fp32)

        res = relay.nn.conv2d(a, w, kernel_size=(3, 3), padding=(1, 1), channels=8, out_dtype=fp32)

        mod = partition(res)
        use_bnns = len(mod.get_global_vars()) == 2  # GlobalVar: "main" and "bnns_0"

        assert use_bnns == const_weights


def test_pattern_conv2d_with_non_cons_bias():
    a = relay.var("a", shape=[2, 7, 8, 8], dtype=fp32)
    w = relay.const(np.random.uniform(-10, 10, (8, 7, 3, 3)).astype(fp32))
    res = relay.nn.conv2d(a, w, kernel_size=(3, 3), padding=(1, 1), channels=8, out_dtype=fp32)
    b = relay.var("b", shape=[8], dtype=fp32)
    res = relay.nn.bias_add(res, b, axis=1)

    mod = partition(res)
    bias_is_fused = is_op_fused(mod["tvmgen_default_bnns_main_0"], "nn.bias_add")

    assert not bias_is_fused
