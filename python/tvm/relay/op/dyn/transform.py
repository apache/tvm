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

# pylint: disable=import-outside-toplevel
"""Dynamic Transform operators."""

from . import _make


def reshape(data, newshape):
    """Reshape the input array based on the values of the newshape tensor.

    To give user more convenience in without doing manual shape inference,
    some dimensions of the shape can take special values from the set {0, -1, -3}.
    The significance of each is explained below:

    ``0`` copy this dimension from the input to the output shape.

        .. code-block:: python

            data.shape = (2,3,4), newshape = (4,0,2), result.shape = (4,3,2)
            data.shape = (2,3,4), newshape = (2,0,0), result.shape = (2,3,4)

    ``-1`` infers the dimension of the output shape by using the remainder of
    the input dimensions keeping the size of the new array same as that of the input array.
    At most one dimension of shape can be -1.

        .. code-block:: python

            data.shape = (2,3,4), newshape = (6,1,-1), result.shape = (6,1,4)
            data.shape = (2,3,4), newshape = (3,-1,8), result.shape = (3,1,8)
            data.shape = (2,3,4), newshape = (-1,), result.shape = (24,)

    ``-3`` use the product of two consecutive dimensions of the input shape
    as the output dimension.

        .. code-block:: python

            data.shape = (2,3,4), newshape = (-3,4), result.shape = (6,4)
            data.shape = (2,3,4,5), newshape = (-3,-3), result.shape = (6,20)
            data.shape = (2,3,4), newshape = (0,-3), result.shape = (2,12)

    Special values -2 and -4 from the standard reshape op would introduce dynamic rank
    in this op. Thus, they are not permitted.

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    newshape : relay.Expr
        The new shape. Should be compatible with the original shape.

    Returns
    -------
    result : relay.Expr
        The reshaped result.
    """
    return _make.reshape(data, newshape)

def broadcast_to(data, shape):

    """Return a scalar value array with the same type, broadcast to
    the provided shape.

    Parameters
    ----------
    data : relay.Expr
        The input tensor.

    shape : a relay.Expr, cannot be a tuple of consts
        Provide the shape to broadcast to.

    Returns
    -------
    result : relay.Expr
        The resulting tensor.
    """
    return _make.broadcast_to(data, shape)
