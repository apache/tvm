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
import pytest

pytest.importorskip("ethosu.vela")

from math import isclose
from tvm.contrib.ethosu.cascader import StripeConfig, Propagator


def test_propagator():
    transform = [
        [1, 0, 0, 0],
        [0, 1 / 2, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ]
    offset = [-1, 1, 2]
    propagator = Propagator(
        transform=transform,
        offset=offset,
    )
    assert list(propagator.offset) == offset
    for i, row in enumerate(transform):
        for j, value in enumerate(row):
            assert isclose(propagator.transform[i][j], value)


@pytest.mark.parametrize(
    ["propagator", "input_stripe_config", "output_stripe_config"],
    [
        (
            Propagator(
                transform=[
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 1 / 16, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 16],
                    [0, 0, 0, 0, 1],
                ],
                offset=[0, 0, 0, 0, 0],
            ),
            StripeConfig(
                shape=[1, 12, 14, 36],
                extent=[1, 24, 18, 72],
                strides=[1, 12, 14, 36],
                order=[1, 2, 3, 4],
                stripes=[1, 2, 2, 2],
                offset=[0, 0, 0, 0],
            ),
            StripeConfig(
                shape=[1, 12, 3, 14, 16],
                extent=[1, 24, 5, 18, 16],
                strides=[1, 12, 2.25, 14, 0],
                order=[1, 2, 4, 3, 0],
                stripes=[1, 2, 2, 2, 1],
                offset=[0, 0, 0, 0, 0],
            ),
        ),
        (
            Propagator(
                transform=[
                    [0.5, 0, 0],
                    [0, 0.5, 0],
                    [0, 0, 1],
                ],
                offset=[0, 0],
            ),
            StripeConfig(
                shape=[3, 5],
                extent=[27, 50],
                strides=[3, 5],
                order=[1, 2],
                stripes=[9, 10],
                offset=[0, 0],
            ),
            StripeConfig(
                shape=[2, 3],
                extent=[14, 25],
                strides=[1.5, 2.5],
                order=[1, 2],
                stripes=[9, 10],
                offset=[0, 0],
            ),
        ),
        (
            Propagator(
                transform=[
                    [2, 0, 0, 4],
                    [0, 1, 0, 2],
                    [0, 0, 0, 8],
                    [0, 0, 0, 1],
                ],
                offset=[-2, -1, 0],
            ),
            StripeConfig(
                shape=[4, 6, 32],
                extent=[48, 60, 64],
                strides=[4, 6, 32],
                order=[1, 2, 3],
                stripes=[12, 10, 2],
                offset=[0, 0, 0],
            ),
            StripeConfig(
                shape=[12, 8, 8],
                extent=[100, 62, 8],
                strides=[8, 6, 0],
                order=[1, 2, 0],
                stripes=[12, 10, 1],
                offset=[-2, -1, 0],
            ),
        ),
    ],
)
def test_propagate(propagator, input_stripe_config, output_stripe_config):
    result_stripe_config = propagator.propagate(input_stripe_config)
    assert result_stripe_config == output_stripe_config


if __name__ == "__main__":
    tvm.testing.main()
