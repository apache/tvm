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
"""DNNL library supported operators.
There are two ways to registering a function for an op to indicate if it is
supported by DNNL.

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
from ... import expr as _expr
from ... import op as _op
from .register import register_pattern_table


def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by DNNL.

    Paramters
    ---------
    op_name : Str
        The name of operator that will be registered.

    Returns
    -------
    f : callable
        A function that returns if the operator is supported by DNNL.
    """
    @_op.register(op_name, "target.dnnl")
    def _func_wrapper(attrs, args):
        return supported

    return _func_wrapper


_register_external_op_helper("nn.batch_norm")
_register_external_op_helper("nn.conv2d")
_register_external_op_helper("nn.dense")
_register_external_op_helper("nn.relu")
_register_external_op_helper("add")
_register_external_op_helper("subtract")
_register_external_op_helper("multiply")


def make_pattern(with_bias=True):
    data = _expr.var("data")
    weight = _expr.var("weight")
    bias = _expr.var("bias")
    conv = _op.nn.conv2d(data, weight)
    if with_bias:
        conv_out = _op.add(conv, bias)
    else:
        conv_out = conv
    return _op.nn.relu(conv_out)


@register_pattern_table("dnnl")
def pattern_table():
    conv2d_bias_relu_pat = ("dnnl.conv2d_bias_relu", make_pattern(with_bias=True))
    conv2d_relu_pat = ("dnnl.conv2d_relu", make_pattern(with_bias=False))
    dnnl_patterns = [conv2d_bias_relu_pat, conv2d_relu_pat]
    return dnnl_patterns
