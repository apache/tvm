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
"""Arm Compute Library supported operators."""
import tvm
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name

from ...dataflow_pattern import wildcard, is_op, is_constant
from .register import register_pattern_table


def is_arm_compute_runtime_enabled():
    """Check if the ACL graph runtime is present.

    Returns
    -------
    ret: bool
        True if present, False if not.
    """
    check_enabled = tvm.get_global_func("relay.op.is_arm_compute_runtime_enabled", True)
    if check_enabled:
        return check_enabled()
    return False


def partition_for_arm_compute_lib(mod, params=None):
    """Partition the graph greedily offloading supported
    operators to Arm Compute Library.

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
        mod['main'] = bind_params_by_name(mod['main'], params)

    seq = tvm.transform.Sequential([transform.MergeComposite(arm_compute_lib_pattern_table()),
                                    transform.AnnotateTarget('arm_compute_lib'),
                                    transform.PartitionGraph()])

    return seq(mod)


@register_pattern_table("arm_compute_lib")
def arm_compute_lib_pattern_table():
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

    def qnn_conv_pattern():
        """Create a quantized convolution pattern.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern.
        """
        pattern = is_op('nn.pad')(wildcard()) | wildcard()
        pattern = is_op('qnn.conv2d')(
            pattern, is_constant(), is_constant(), is_constant(), is_constant(), is_constant())
        pattern = pattern.optional(lambda x: is_op('nn.bias_add')(x, is_constant()))
        pattern = pattern.optional(is_op('nn.relu'))
        pattern = is_op('qnn.requantize')(
            pattern, wildcard(), wildcard(), is_constant(), is_constant())
        return pattern

    def check_conv(extract):
        """Check conv pattern is supported by ACL."""
        call = extract
        while call.op.name != "nn.conv2d":
            call = call.args[0]
        return conv2d(call.attrs, call.args)

    def check_qnn_conv(extract):
        """Check qnn conv pattern is supported by ACL."""
        if extract.attrs.out_dtype != "uint8":
            return False
        call = extract
        while call.op.name != "qnn.conv2d":
            call = call.args[0]
        return qnn_conv2d(call.attrs, call.args)

    return [('arm_compute_lib.conv2d', conv_pattern(), check_conv),
            ('arm_compute_lib.qnn_conv2d', qnn_conv_pattern(), check_qnn_conv)]


def _register_external_op_helper(op_name, supported=True):
    @tvm.ir.register_op_attr(op_name, "target.arm_compute_lib")
    def _func_wrapper(attrs, args):
        return supported

    return _func_wrapper


_register_external_op_helper("reshape")


@tvm.ir.register_op_attr("nn.conv2d", "target.arm_compute_lib")
def conv2d(attrs, args):
    """Check if the external ACL codegen for conv2d should be used."""
    if attrs.groups != 1:
        return False
    if attrs.data_layout != "NHWC":
        return False
    if attrs.out_dtype != "float32" and attrs.out_dtype != "":
        return False
    data_typ = args[0].checked_type
    if len(data_typ.shape) != 4 or data_typ.shape[0] != 1 or data_typ.dtype != "float32":
        return False
    kernel_typ = args[1].checked_type
    if len(kernel_typ.shape) != 4 or kernel_typ.dtype != "float32":
        return False
    return True


def qnn_conv2d(attrs, args):
    """Check if the external ACL codegen for qnn.conv2d should be used."""
    if attrs.groups != 1:
        return False
    if attrs.data_layout != "NHWC":
        return False
    if attrs.out_dtype != "int32" and attrs.out_dtype != "":
        return False
    data_typ = args[0].checked_type
    if len(data_typ.shape) != 4 or data_typ.shape[0] != 1 or data_typ.dtype != "uint8":
        return False
    kernel_typ = args[1].checked_type
    if len(kernel_typ.shape) != 4 or kernel_typ.dtype != "uint8":
        return False
    return True


@tvm.ir.register_op_attr("nn.max_pool2d", "target.arm_compute_lib")
def max_pool2d(attrs, args):
    """Check if the external ACL codegen for maxpool2d should be used."""
    if attrs.layout != "NHWC":
        return False
    typ = args[0].checked_type
    if typ.dtype not in ["float32", "uint8"]:
        return False
    return True
