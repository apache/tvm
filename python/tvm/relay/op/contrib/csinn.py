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
"""CSINN library supported operators.
There are two ways to registering a function for an op to indicate if it is
supported by CSINN.

- The first and simplest way is to use the helper so that
users only need to provide the operator name and a boolean value to indicate if
it is supported. For example:

    .. code-block:: python

      add = _register_external_op_helper("add")
      add = _register_external_op_helper("add", True)
      add = _register_external_op_helper("add", False)

- The other way is to implement the function by themselves to
check the attributes of the op and decide if it should be offloaded to DNNL.
"""
import tvm.ir
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from ...dataflow_pattern import is_constant, wildcard, is_op
from .register import register_pattern_table


def is_csinn_runtime_enabled():
    """Check if the CSINN runtime is present.

    Returns
    -------
    ret: bool
        True if present, False if not.
    """
    check_enabled = tvm.get_global_func("relay.op.is_csinn_runtime_enabled", True)
    if check_enabled:
        return check_enabled()
    return False


def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by CSINN.

    Paramters
    ---------
    op_name : Str
        The name of operator that will be registered.

    Returns
    -------
    f : callable
        A function that returns if the operator is supported by DNNL.
    """

    @tvm.ir.register_op_attr(op_name, "target.csinn")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper


_register_external_op_helper("nn.conv2d")


@register_pattern_table("csinn")
def csinn_pattern_table():
    """Get the csinn pattern table."""

    def conv_pattern():
        """Create a convolution pattern.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern.
        """
        pattern = is_op("nn.conv2d")(wildcard(), is_constant())
        pattern = is_op("nn.bias_add")(pattern, is_constant())
        return pattern

    return [
        ("csinn.conv2d", conv_pattern()),
    ]


def partition_for_csinn(mod, params=None, **opts):
    """Partition the graph greedily offloading supported operators to CSINN.

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
            transform.MergeComposite(csinn_pattern_table()),
            transform.AnnotateTarget("csinn", False),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )

    return seq(mod)
