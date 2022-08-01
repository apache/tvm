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
#
# pylint: disable=invalid-name, too-many-arguments, too-many-nested-blocks, no-else-raise
"""Interpolate operator"""
from tvm.te import hybrid


@hybrid.script
def _interpolate_1d(x, xp, fp):

    lenx = x.shape[0]
    lenxp = xp.shape[0]
    minxp = xp[0]
    maxxp = xp[lenxp - 1]

    out = output_tensor(x.shape, fp.dtype)

    for i in range(lenx):
        for j in range(lenxp):
            if x[i] <= minxp:
                out[i] = fp[0]
            elif x[i] >= maxxp:
                out[i] = fp[lenxp - 1]
            elif j > 0 and x[i] >= xp[j - 1] and x[i] < xp[j]:
                out[i] = fp[j - 1] + (x[i] - xp[j - 1]) * (
                    (fp[j] - fp[j - 1]) / (xp[j] - xp[j - 1])
                )

    return out


@hybrid.script
def _interpolate_1d_tensor(x, xp, fp):

    lenx = x.shape[0]
    lenxp = xp.shape[0]
    minxp = xp[0][0]
    maxxp = xp[lenxp - 1][0]

    out = output_tensor(x.shape, x.dtype)

    for i in range(lenx):
        for j in range(lenxp):
            if x[i][0] <= minxp:
                out[i][0] = fp[0][0]
            elif x[i][0] >= maxxp:
                out[i][0] = fp[lenxp - 1][0]
            elif j > 0 and x[i][0] >= xp[j - 1][0] and x[i][0] < xp[j][0]:
                out[i][0] = fp[j - 1][0] + (x[i][0] - xp[j - 1][0]) * (
                    (fp[j][0] - fp[j - 1][0]) / (xp[j][0] - xp[j - 1][0])
                )

    return out


def interpolate(x, xp, fp):
    """Calculates piecewise interpolant to a function with given discrete data points
    and evaluated at given indices.

    .. note::
        Similar to ``numpy.interp``.

    Parameters
    ----------
    x : relay.Expr
        The indices at which to evaluate the interpolated values.

    xp : relay.Expr
        The indices corresponding to the reference data points.

    fp : relay.Expr
        The values of the reference data points.

    Returns
    -------
    ret : relay.Expr
        The computed result.

    Examples
    --------
    .. code-block:: python

        x = [0, 1, 1.5, 2.72, 3.14]
        xp = [1, 2, 3]
        fp = [3, 2, 0]
        f = relay.interpolate(x, xp, fp)
        f = [3.  , 3.  , 2.5 , 0.56, 0.  ]
    """
    if len(x.shape) > 1 and x.shape[1] > 1:
        raise ValueError("Interpolate currently only supports 1d linear interpolation.")
    elif len(x.shape) > 1 and x.shape[1] == 1:
        return _interpolate_1d_tensor(x, xp, fp)
    else:
        return _interpolate_1d(x, xp, fp)
