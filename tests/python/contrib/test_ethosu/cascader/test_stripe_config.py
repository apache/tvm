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

from tvm.contrib.ethosu.cascader.stripe_config import StripeConfig, count_stripes


def test_stripe_config():
    shape = [1, 2, 3]
    extent = [2, 3, 4]
    strides = [3, 4, 5]
    order = [4, 5, 6]
    stripes = [5, 6, 7]
    offset = [6, 7, 8]
    hash_value = 3107995860559090954
    stripe_config = StripeConfig(
        shape=shape,
        extent=extent,
        strides=strides,
        order=order,
        stripes=stripes,
        offset=offset,
    )
    assert stripe_config.shape == shape
    assert stripe_config.extent == extent
    assert stripe_config.strides == strides
    assert stripe_config.order == order
    assert stripe_config.stripes == stripes
    assert stripe_config.offset == offset
    assert hash(stripe_config) == hash_value


@pytest.mark.parametrize(
    "mismatch", [None, "shape", "extent", "strides", "order", "stripes", "offset"]
)
def test_stripe_config_equal(mismatch):
    init_dict = {
        "shape": [1, 2, 3],
        "extent": [2, 3, 4],
        "strides": [3, 4, 5],
        "order": [4, 5, 6],
        "stripes": [5, 6, 7],
        "offset": [6, 7, 8],
    }
    stripe_config_a = StripeConfig(**init_dict)
    if mismatch:
        init_dict[mismatch] = [1, 1, 1]
    stripe_config_b = StripeConfig(**init_dict)
    if not mismatch:
        assert stripe_config_a == stripe_config_b
    else:
        assert stripe_config_a != stripe_config_b


@pytest.mark.parametrize(
    ["stripe_config", "expected_stripe_counts"],
    [
        (
            StripeConfig(
                shape=[3, 3, 3],
                extent=[9, 9, 9],
                strides=[3, 3, 3],
                order=[1, 2, 3],
                stripes=[3, 3, 3],
                offset=[0, 0, 0],
            ),
            {
                (3, 3, 3): 27,
            },
        ),
        (
            StripeConfig(
                shape=[3, 3],
                extent=[10, 10],
                strides=[2, 2],
                order=[1, 2],
                stripes=[5, 5],
                offset=[0, 0],
            ),
            {
                (3, 3): 16,
                (2, 3): 4,
                (3, 2): 4,
                (2, 2): 1,
            },
        ),
        (
            StripeConfig(
                shape=[3, 3, 9],
                extent=[9, 9, 9],
                strides=[3, 3, 0],
                order=[1, 2, 3],
                stripes=[3, 3, 1],
                offset=[0, 0, 0],
            ),
            {
                (3, 3, 9): 9,
            },
        ),
        (
            StripeConfig(
                shape=[5, 5],
                extent=[8, 8],
                strides=[5, 5],
                order=[1, 2],
                stripes=[2, 2],
                offset=[0, 0],
            ),
            {
                (5, 5): 1,
                (3, 5): 1,
                (5, 3): 1,
                (3, 3): 1,
            },
        ),
        (
            StripeConfig(
                shape=[5, 5],
                extent=[8, 8],
                strides=[5, 5],
                order=[1, 2],
                stripes=[2, 2],
                offset=[-1, -2],
            ),
            {
                (4, 3): 2,
                (4, 5): 2,
            },
        ),
        (
            StripeConfig(
                shape=[13, 7],
                extent=[128, 73],
                strides=[13, 7],
                order=[1, 2],
                stripes=[11, 12],
                offset=[-10, -5],
            ),
            {
                (3, 1): 1,
                (3, 2): 1,
                (8, 7): 10,
                (8, 2): 1,
                (13, 7): 90,
                (13, 1): 9,
                (8, 1): 1,
                (3, 7): 10,
                (13, 2): 9,
            },
        ),
    ],
)
def test_count_stripes(stripe_config, expected_stripe_counts):
    assert count_stripes(stripe_config) == expected_stripe_counts


@pytest.mark.parametrize(
    ["stripe_config", "expected_stripe_counts"],
    [
        (
            StripeConfig(
                shape=[4, 4],
                extent=[16, 16],
                strides=[2, 2],
                order=[1, 2],
                stripes=[7, 7],
                offset=[0, 0],
            ),
            {
                (4, 4): 1,
                (2, 4): 6,
                (4, 2): 6,
                (2, 2): 36,
            },
        ),
        (
            StripeConfig(
                shape=[4, 4],
                extent=[8, 8],
                strides=[2, 2],
                order=[1, 2],
                stripes=[6, 3],
                offset=[-5, 0],
            ),
            {
                (1, 4): 2,
                (2, 4): 3,
                (2, 2): 6,
                (1, 2): 4,
            },
        ),
    ],
)
def test_count_stripes_sliding_window(stripe_config, expected_stripe_counts):
    assert count_stripes(stripe_config, enable_sliding_window=True) == expected_stripe_counts


if __name__ == "__main__":
    tvm.testing.main()
