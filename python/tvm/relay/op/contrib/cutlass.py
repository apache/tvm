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
from tvm.relay import transform
from ...dataflow_pattern import wildcard, is_op, is_constant
from .register import register_pattern_table


def make_gemm_pattern(with_bias=True, with_act=None):
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    gemm = is_op("nn.dense")(data, weight)
    if with_bias:
        add_or_bias_add = is_op("add") | is_op("nn.bias_add")
        gemm_out = add_or_bias_add(gemm, bias)
    else:
        gemm_out = gemm

    if with_act is None:
        return gemm_out
    elif isinstance(with_act, str) and with_act == "nn.relu":
        return is_op(with_act)(gemm_out)


def get_pattern_table():
    dense_pat = ("cutlass.dense", make_gemm_pattern(False, None))
    dense_bias_pat = ("cutlass.dense_bias", make_gemm_pattern(True, None))
    dense_bias_relu_pat = ("cutlass.dense_bias_relu", make_gemm_pattern(True, "nn.relu"))
    cutlass_patterns = [
        dense_bias_relu_pat,
        dense_bias_pat,
        dense_pat,
    ]
    return cutlass_patterns


def partition_for_cutlass(mod):
    mod = transform.MergeComposite(get_pattern_table())(mod)
    mod = transform.AnnotateTarget(["cutlass"])(mod)
    mod = transform.PartitionGraph()(mod)
    return mod
