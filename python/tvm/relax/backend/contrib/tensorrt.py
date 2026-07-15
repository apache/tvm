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

"""Pattern table and partitioning for the TensorRT BYOC backend.

The composite name of each pattern is "tensorrt.<op>", matching the runtime
converter registered under the same name (the converters are keyed by
"tensorrt." + op_name). ``partition_for_tensorrt`` carves the matched subgraphs
out of the module and annotates them for the ``tensorrt`` codegen.
"""

from collections.abc import Mapping

from tvm.ir import IRModule
from tvm.relax.dpl.pattern import DFPattern, is_op, wildcard
from tvm.relax.transform import FuseOpsByPattern, MergeCompositeFunctions

from ..pattern_registry import get_patterns_with_prefix, register_patterns

Pattern = tuple[str, DFPattern, Mapping[str, DFPattern]]


def _op_pattern(composite_name: str, op_name: str, num_args: int) -> Pattern:
    """A pattern matching a single op called with ``num_args`` wildcard arguments."""
    args = [wildcard() for _ in range(num_args)]
    return (composite_name, is_op(op_name)(*args), {})


def _tensorrt_patterns() -> list[Pattern]:
    patterns: list[Pattern] = []

    # Activations and unary elementwise ops (single tensor argument).
    for composite, op in [
        ("tensorrt.nn.relu", "relax.nn.relu"),
        ("tensorrt.sigmoid", "relax.sigmoid"),
        ("tensorrt.nn.silu", "relax.nn.silu"),
        ("tensorrt.tanh", "relax.tanh"),
        ("tensorrt.exp", "relax.exp"),
        ("tensorrt.log", "relax.log"),
        ("tensorrt.sqrt", "relax.sqrt"),
        ("tensorrt.abs", "relax.abs"),
        ("tensorrt.negative", "relax.negative"),
        ("tensorrt.sin", "relax.sin"),
        ("tensorrt.cos", "relax.cos"),
        ("tensorrt.atan", "relax.atan"),
        ("tensorrt.ceil", "relax.ceil"),
        ("tensorrt.floor", "relax.floor"),
        ("tensorrt.erf", "relax.erf"),
        ("tensorrt.nn.softmax", "relax.nn.softmax"),
        ("tensorrt.nn.batch_flatten", "relax.nn.batch_flatten"),
        ("tensorrt.expand_dims", "relax.expand_dims"),
        ("tensorrt.squeeze", "relax.squeeze"),
        ("tensorrt.transpose", "relax.permute_dims"),
        ("tensorrt.layout_transform", "relax.layout_transform"),
        ("tensorrt.nn.max_pool2d", "relax.nn.max_pool2d"),
        ("tensorrt.nn.avg_pool2d", "relax.nn.avg_pool2d"),
        ("tensorrt.nn.max_pool3d", "relax.nn.max_pool3d"),
        ("tensorrt.nn.avg_pool3d", "relax.nn.avg_pool3d"),
        ("tensorrt.nn.adaptive_avg_pool2d", "relax.nn.adaptive_avg_pool2d"),
        ("tensorrt.sum", "relax.sum"),
        ("tensorrt.prod", "relax.prod"),
        ("tensorrt.max", "relax.max"),
        ("tensorrt.min", "relax.min"),
        ("tensorrt.mean", "relax.mean"),
        ("tensorrt.concatenate", "relax.concat"),
        ("tensorrt.split", "relax.split"),
    ]:
        patterns.append(_op_pattern(composite, op, 1))

    # Binary elementwise ops (two tensor arguments).
    for composite, op in [
        ("tensorrt.add", "relax.add"),
        ("tensorrt.subtract", "relax.subtract"),
        ("tensorrt.multiply", "relax.multiply"),
        ("tensorrt.divide", "relax.divide"),
        ("tensorrt.power", "relax.power"),
        ("tensorrt.maximum", "relax.maximum"),
        ("tensorrt.minimum", "relax.minimum"),
    ]:
        patterns.append(_op_pattern(composite, op, 2))

    # image.resize2d (data + target-size shape argument).
    patterns.append(_op_pattern("tensorrt.image.resize2d", "relax.image.resize2d", 2))

    # Convolutions and matmul (data + weight).
    for composite, op in [
        ("tensorrt.nn.conv1d", "relax.nn.conv1d"),
        ("tensorrt.nn.conv2d", "relax.nn.conv2d"),
        ("tensorrt.nn.conv3d", "relax.nn.conv3d"),
        ("tensorrt.nn.conv2d_transpose", "relax.nn.conv2d_transpose"),
        ("tensorrt.nn.conv3d_transpose", "relax.nn.conv3d_transpose"),
        ("tensorrt.nn.batch_matmul", "relax.matmul"),
        ("tensorrt.reshape", "relax.reshape"),
    ]:
        patterns.append(_op_pattern(composite, op, 2))

    # layer_norm (data, gamma, beta) and clip (data, min, max).
    patterns.append(_op_pattern("tensorrt.nn.layer_norm", "relax.nn.layer_norm", 3))
    patterns.append(_op_pattern("tensorrt.clip", "relax.clip", 3))

    # strided_slice is called either with or without the optional strides argument.
    patterns.append(_op_pattern("tensorrt.strided_slice", "relax.strided_slice", 5))
    patterns.append(_op_pattern("tensorrt.strided_slice", "relax.strided_slice", 4))

    return patterns


register_patterns(_tensorrt_patterns())


def partition_for_tensorrt(mod: IRModule) -> IRModule:
    """Partition the module, offloading TensorRT-supported subgraphs.

    Parameters
    ----------
    mod : tvm.ir.IRModule
        The module to partition. Bind model parameters (e.g. via
        ``relax.transform.BindParams``) before calling this so that weights are
        available to TensorRT as constants.

    Returns
    -------
    mod : tvm.ir.IRModule
        The module with TensorRT-supported subgraphs grouped into composite
        functions annotated for the ``tensorrt`` codegen.
    """
    patterns = get_patterns_with_prefix("tensorrt")
    mod = FuseOpsByPattern(patterns, bind_constants=True, annotate_codegen=False)(mod)
    mod = MergeCompositeFunctions(["tensorrt"])(mod)
    return mod
