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
"""Patterns supported CUTLASS."""
from tvm.relay import transform
from ...dataflow_pattern import wildcard, is_op, is_constant


def make_gelu_pattern(bias_out, out_dtype="float16"):
    mul = is_op("multiply")(bias_out, is_constant() | wildcard())
    if out_dtype == "float16":
        erf = is_op("cast")(is_op("erf")(is_op("cast")(mul)))
    else:
        erf = is_op("erf")(mul)
    mul_half = is_op("multiply")(erf, is_constant() | wildcard())
    add = is_op("add")(mul_half, is_constant() | wildcard())
    return is_op("multiply")(add, bias_out)


def make_gemm_pattern(with_bias=True, with_act=None, out_dtype="float16"):
    """Create a pattern for dense op followed by activations."""
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
    if isinstance(with_act, str) and with_act == "relu":
        return is_op("nn.relu")(gemm_out)

    assert isinstance(with_act, str) and with_act == "gelu"
    return make_gelu_pattern(gemm_out, out_dtype)


def make_batch_matmul_pattern():
    return is_op("nn.batch_matmul")(wildcard(), wildcard())


def partition_for_cutlass(mod):
    """Partition the input module into CUTLASS-supported subgraphs."""
    dense_pat = ("cutlass.dense", make_gemm_pattern(False, None))
    dense_bias_pat = ("cutlass.dense_bias", make_gemm_pattern(True, None))
    dense_bias_relu_pat = ("cutlass.dense_bias_relu", make_gemm_pattern(True, "relu"))
    dense_bias_gelu_fp16_pat = ("cutlass.dense_bias_gelu_fp16", make_gemm_pattern(True, "gelu"))
    dense_bias_gelu_fp32_pat = (
        "cutlass.dense_bias_gelu_fp32",
        make_gemm_pattern(True, "gelu", out_dtype="float32"),
    )
    cutlass_patterns = [
        dense_bias_gelu_fp16_pat,
        dense_bias_gelu_fp32_pat,
        dense_bias_relu_pat,
        dense_bias_pat,
        dense_pat,
        ("cutlass.batch_matmul", make_batch_matmul_pattern()),
    ]
    mod = transform.MergeComposite(cutlass_patterns)(mod)
    mod = transform.AnnotateTarget(["cutlass"])(mod)
    mod = transform.PartitionGraph()(mod)
    return mod
