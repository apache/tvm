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
# pylint: disable=invalid-name, unused-argument
"""Arm(R) CMSIS-NN supported operators for Cortex-M."""
import tvm.ir
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name

from ...dataflow_pattern import is_constant, is_op, wildcard
from .register import register_pattern_table


def partition_for_cmsisnn(mod, params=None, **opts):
    """Partition the graph greedily offloading supported
    operators on Cortex-M using CMSIS-NN

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.

    Returns
    -------
    ret : annotated and partitioned module.
    """
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.MergeComposite(pattern_table()),
            transform.AnnotateTarget("cmsisnn"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )

    return seq(mod)


@register_pattern_table("cmsisnn")
def pattern_table():
    """Get the cmsisnn compiler pattern table."""

    def softmax_pattern():
        pattern = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())
        pattern = is_op("nn.softmax")(pattern)
        pattern = is_op("qnn.quantize")(pattern, is_constant(), is_constant())
        return pattern

    def check_quantized_softmax(extract):
        """Check if softmax is supported by CMSIS-NN."""

        # check for dtypes of quantize and dequantize
        return (
            extract.attrs.out_dtype == "int8"
            and extract.args[0].args[0].args[0].checked_type.dtype == "int8"
        )

    return [
        ("cmsisnn.qnn_softmax", softmax_pattern(), check_quantized_softmax),
    ]
