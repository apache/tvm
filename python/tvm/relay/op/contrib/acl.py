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
"""ACL library supported operators."""
import tvm
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name

from ...dataflow_pattern import wildcard, is_op, is_constant
from .register import register_pattern_table


def is_acl_runtime_present():
    """Check if the ACL graph runtime is present.

    Returns
    -------
    ret: bool
        True if present, False if not.
    """
    return tvm.get_global_func("relay.op.is_acl_runtime_enabled", True)


def partition_for_acl(mod, params=None):
    """Partition the graph greedily offloading supported
    operators to ACL.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : dict[str, NDArray]
        Constant input parameters.

    Returns
    -------
    ret : annotated and partitioned module.
    """
    if params:
        mod['main'] = bind_params_by_name(mod['main'], params)

    seq = tvm.transform.Sequential([transform.MergeComposite(pattern_table()),
                                    transform.AnnotateTarget('acl'),
                                    transform.PartitionGraph()])

    return seq(mod)


@register_pattern_table("acl")
def pattern_table():
    """Get the ACL pattern table."""

    def conv_pattern():
        """Create a convolution pattern.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern.
        """
        pattern = is_op('nn.pad')(wildcard()) | wildcard()
        pattern = is_op('nn.conv2d')(pattern, is_constant())
        pattern = pattern.optional(lambda x: is_op('nn.bias_add')(x, is_constant()))
        pattern = pattern.optional(is_op('nn.relu'))
        return pattern

    def check_conv(extract):
        """Check conv pattern is supported by ACL."""
        call = extract
        while call.op.name != "nn.conv2d":
            call = call.args[0]
        return conv2d(call.attrs, call.args)

    return [('acl.conv2d', conv_pattern(), check_conv)]


def _register_external_op_helper(op_name, supported=True):
    @tvm.ir.register_op_attr(op_name, "target.acl")
    def _func_wrapper(attrs, args):
        return supported

    return _func_wrapper


_register_external_op_helper("reshape")


@tvm.ir.register_op_attr("nn.conv2d", "target.acl")
def conv2d(attrs, args):
    """Check if the external ACL codegen for conv2d should be used."""

    # ACL only supports group size of 1
    if attrs.groups != 1:
        return False

    # ACL only supports NHWC layout
    if attrs.data_layout != "NHWC":
        return False

    return True


@tvm.ir.register_op_attr("nn.max_pool2d", "target.acl")
def max_pool2d(attrs, args):
    """Check if the external ACL codegen for maxpool2d should be used."""

    # ACL only supports NHWC layout
    if attrs.layout != "NHWC":
        return False

    return True
