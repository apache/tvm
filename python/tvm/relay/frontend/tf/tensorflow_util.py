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
# pylint: disable=import-self, invalid-name, unused-argument, too-many-lines, len-as-condition, broad-except
# pylint: disable=import-outside-toplevel, redefined-builtin
"""TF: Tensorflow frontend utils"""
import numpy as np
import tvm

from ... import expr as _expr


def get_pad_pair(input1d, kernel1d, stride1d):
    """Extract pad values"""

    if input1d % stride1d == 0:
        pad = max(kernel1d - stride1d, 0)
    else:
        pad = max(kernel1d - (input1d % stride1d), 0)

    pad_before = pad // 2
    pad_after = pad - pad_before

    return [pad_before, pad_after]


def math_name_picker(surfix):
    """Make math function name"""

    def _impl(attr):
        return "broadcast_" + surfix

    return _impl


def dimension_picker(prefix, surfix=""):
    """Extracts dimensions"""

    def _impl(attr):
        kernel = attr["kernel_shape"]
        if len(kernel) == 2:
            return prefix + "2d" + surfix
        if len(kernel) == 3:
            return prefix + "3d" + surfix
        raise tvm.error.OpAttributeInvalid(
            "Only 2D or 3D kernels are supported for operator {}".format(prefix + "2d or 3d")
        )

    return _impl


def dimension_constraint():
    """Check dimentions validity"""

    def _dim_check(attrs):
        if len(attrs["kernel_shape"]) in (2, 3):
            return True
        return False

    return _dim_check, "Only 2d or 3d kernel supported."


def get_param(params, input_node):
    """Get param"""

    if isinstance(input_node, _expr.Constant):
        return np.atleast_1d(input_node.data.asnumpy())
    return params[input_node.name_hint].asnumpy()


def get_num_param(params, input_node):
    """Get neumeric params"""

    return get_param(params, input_node).item()


def get_list_param(params, input_node):
    """Get list param"""

    return get_param(params, input_node).tolist()


def get_tuple_param(params, input_node):
    """get tule param"""

    return tuple(get_param(params, input_node))


def need_prelude_for_shape_inference(op):
    """Prelude check"""

    return "TensorArray" in op


def get_more_static_shape(shape0, shape1):
    """Compare two shapes with the same rank,
    and return the one with fewer symbolic dimension.
    """

    assert len(shape0) == len(shape1)
    num_sym_dim0 = 0
    num_sym_dim1 = 0
    for dim0, dim1 in zip(list(shape0), list(shape1)):
        if not isinstance(dim0, int):
            num_sym_dim0 += 1
        if not isinstance(dim1, int):
            num_sym_dim1 += 1

    if num_sym_dim0 < num_sym_dim1:
        return shape0
    return shape1
