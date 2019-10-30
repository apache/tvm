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

"""Predefined dispatching functions for dynamic input shape."""

from tvm.expr import IntImm
from ..expr import const

def uniform_dispatcher(low=1, high=256, step=16):
    """Uniformly dispatch dynamic input shape.

    Always start splitting from 1 and end at high.
    Generated bucket will be truncated with respect to low and high.

    An extra bucket [high, -1] will be added to indicate
    the last bucket from high to positive infinite.

    Parameters
    ----------
    low : int
        Low end of symbolic axis

    high : int
        High end of symbolic axis

    step : int
        Step size.

    Returns
    -------
    out : Function
        Implementation function.
    """
    def _impl(input_shape):
        """Implementation function"""
        buckets = {}
        bucket = []

        left = low
        right = left
        for i in range(high):
            uniform_left = i * step
            uniform_right = (i + 1) * step
            if uniform_left <= left < uniform_right:
                right = min(high, uniform_right)
                break
        bucket.append((left, right))
        while right < high:
            left = right
            right = min(high, left + step)
            bucket.append((left, right))

        bucket.append((high, -1))

        for input_name, shape in input_shape.items():
            num_sym_axes = 0
            bucket_dict = {}
            for i, axis in enumerate(shape):
                if not isinstance(axis, (int, IntImm)):
                    num_sym_axes += 1
                    bucket_dict[i] = bucket
            buckets[input_name] = bucket_dict

        return buckets
    return _impl


def exp_dispatcher(low=1, high=256, base=2):
    """Dispatch dynamic input shape with [pow(n), pow(n+1)).

    Always start splitting from 1 and end at high.
    Generated bucket will be truncated with respect to low and high.

    An extra bucket [high, -1] will be added to indicate
    the last bucket from high to positive infinite.

    Parameters
    ----------
    low : int
        Low end of symbolic axis

    high : int
        High end of symbolic axis

    base : int
        Base of exponential function.

    Returns
    -------
    out : Function
        Implementation function.
    """
    def _impl(input_shape):
        """Implementation function"""
        buckets = {}
        bucket = []

        left = low
        right = left
        exp = 0
        for i in range(high):
            exp_left = pow(base, i)
            exp_right = pow(base, i + 1)
            if exp_left <= left < exp_right:
                right = min(high, exp_right)
                exp = i + 1
                break
        bucket.append((left, right))
        while right < high:
            exp += 1
            left = right
            right = min(high, pow(base, exp))
            bucket.append((left, right))

        bucket.append((min(bucket[-1][1], high), -1))

        for input_name, shape in input_shape.items():
            num_sym_axes = 0
            bucket_dict = {}
            for i, axis in enumerate(shape):
                if not isinstance(axis, (int, IntImm)):
                    num_sym_axes += 1
                    bucket_dict[i] = bucket
            buckets[input_name] = bucket_dict

        return buckets
    return _impl
