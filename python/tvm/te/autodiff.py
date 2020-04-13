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

"""Automatic differentiation of tensor expressions."""
from . import _ffi_api


def gradient(output, inputs, head=None):
    """Perform reverse-mode automatic differentiation.

    Parameters
    ----------
    output : Tensor
        The tensor to differentiate.

    inputs : List[Tensor]
        The list of input tensors to be differentiated wrt.

    head : Tensor
        The adjoint of the output, in other words, some tensor, by which the Jacobians
        will be multiplied. Its shape must be of the form `prefix + output.shape`.
        If `None` is passed, the identity tensor of shape `output.shape + output.shape`
        will be used.

    Returns
    -------
    tensors: List[Tensor]
        The result gradient, in the same order as the inputs

    Example
    -------
    .. code-block:: python

        x = tvm.placeholder((32, 3, 28, 28), name='x')
        w1 = tvm.placeholder((10, 3, 3, 3), name='w1')
        w2 = tvm.placeholder((10, 10, 3, 3), name='w2')
        z1 = topi.nn.conv2d(x, w1, 1, 1, 1)
        z2 = topi.nn.conv2d(z1, w2, 1, 1, 1)
        y = topi.sum(z2)

        # produce gradients
        [dw1, dw2] = tvm.gradient(y, [w1, w2])

        # produce Jacobians
        [jw1, jw2] = tvm.gradient(z2, [w1, w2])

        # produce gradients, the head adjoint for z2 is provided manually
        [dw1, dw2] = tvm.gradient(z2, [w1, w2], topi.full_like(z2, 1.0))

    """
    if not isinstance(inputs, list):
        inputs = [inputs]
    return _ffi_api.Gradient(output, inputs, head)
