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
"""Patterns supported LIBXSMM."""
import numpy as np

import tvm
from tvm import relay
from tvm.ir.transform import Sequential
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name

from ...dataflow_pattern import wildcard, is_op, is_constant
from .register import register_pattern_table


def get_root_call(call, root_op_name):
    if not isinstance(call, relay.Call):
        return None
    if str(call.op) == root_op_name:
        return call
    return get_root_call(call.args[0], root_op_name)


def check_dense_shape(call):
    dense = get_root_call(call, "nn.dense")
    data = dense.args[0].checked_type
    weight = dense.args[1].checked_type
    m = int(data.shape[0])
    n = int(weight.shape[0])
    k = int(data.shape[1])

    # Conditions to enable libxsmm BYOC.
    # Note: currently we enable libxsmm when cube_root(m * n * k ) <= 256 since it has significant performance improvement.
    return bool(np.cbrt(m * n * k) <= 256)


@tvm.ir.register_op_attr("nn.dense", "target.libxsmm")
def dense(expr):
    return check_dense_shape(expr)


def make_dense_pattern(with_bias=False, eltwise=None):
    data = wildcard()
    weight = is_constant()
    bias = wildcard()
    dense = is_op("nn.dense")(data, weight)
    pattern_name = "libxsmm.dense"
    if with_bias:
        dense_out = is_op("nn.bias_add")(dense, bias)
        pattern_name += "_bias"
    else:
        dense_out = dense
    if eltwise:
        dense_out = is_op(eltwise)(dense_out)
        pattern_name += "_" + eltwise.split(".")[-1]
    return [pattern_name, dense_out, check_dense_shape]


@register_pattern_table("libxsmm")
def pattern_table():
    elt_list = ["nn.relu", None]
    libxsmm_patterns = []
    for with_bias in [True, False]:
        for elt in elt_list:
            libxsmm_patterns.append(make_dense_pattern(with_bias, elt))
    return libxsmm_patterns


def partition_for_libxsmm(mod, params=None):
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.MergeComposite(pattern_table()),
            transform.AnnotateTarget(["libxsmm"]),
            transform.PartitionGraph(),
        ]
    )

    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)

    return mod
