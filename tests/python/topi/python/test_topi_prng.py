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
import tvm
import tvm.relay
import tvm.testing
import tvm.topi
import numpy as np


def threefry_split(target, dev, gen):
    gen_placeholder = tvm.te.placeholder(gen.shape, name="gen", dtype="uint64")
    left_placeholder, right_placeholder = tvm.topi.random.threefry_split(gen_placeholder)
    s = tvm.topi.generic.schedule_extern([left_placeholder, right_placeholder])
    f = tvm.build(s, [gen_placeholder, left_placeholder, right_placeholder])
    left = tvm.nd.array(np.zeros(gen.shape, dtype="uint64"))
    right = tvm.nd.array(np.zeros(gen.shape, dtype="uint64"))
    f(tvm.nd.array(gen), left, right)
    return left.numpy(), right.numpy()


def threefry_generate(target, dev, gen, size):
    gen_placeholder = tvm.te.placeholder(gen.shape, name="gen", dtype="uint64")
    left_placeholder, right_placeholder = tvm.topi.random.threefry_generate(gen_placeholder, size)
    s = tvm.topi.generic.schedule_extern([left_placeholder, right_placeholder])
    f = tvm.build(s, [gen_placeholder, left_placeholder, right_placeholder])
    out_gen = tvm.nd.array(np.zeros(gen.shape, dtype="uint64"))
    rands = tvm.nd.array(np.zeros(size, dtype="uint64"))
    f(tvm.nd.array(gen), out_gen, rands)
    return out_gen.numpy(), rands.numpy()


def uniform(target, dev, gen, low, high, size, dtype):
    gen_placeholder = tvm.te.placeholder(gen.shape, name="gen", dtype="uint64")
    low_placeholder = tvm.te.placeholder(low.shape, name="low", dtype=dtype)
    high_placeholder = tvm.te.placeholder(high.shape, name="high", dtype=dtype)
    left_placeholder, right_placeholder = tvm.topi.random.uniform(
        gen_placeholder, low_placeholder, high_placeholder, size, dtype
    )
    s = tvm.topi.generic.schedule_extern([left_placeholder, right_placeholder])
    f = tvm.build(
        s, [gen_placeholder, low_placeholder, high_placeholder, left_placeholder, right_placeholder]
    )
    out_gen = tvm.nd.array(np.zeros(gen.shape, dtype="uint64"))
    rands = tvm.nd.array(np.zeros(size, dtype=dtype))
    f(tvm.nd.array(gen), tvm.nd.array(low), tvm.nd.array(high), out_gen, rands)
    return out_gen.asnumpy(), rands.asnumpy()


@tvm.testing.parametrize_targets
def test_threefry_split(target, dev):
    # test that results of split do not equal eachother or the input
    gen = tvm.relay.random.threefry_key(0).data.numpy()
    a, b = threefry_split(target, dev, gen)
    assert (a != b).any() and (
        a != gen
    ).any(), "Splitting a gen should result in different output gens"
    # unittest some split inputs
    assert (a == np.array([0, 0, 0, 0, 0, 0, 0, 0, 1 << 62, 0], dtype="uint64")).all()
    assert (b == np.array([0, 0, 0, 0, 1 << 63, 0, 0, 0, 1 << 62, 0], dtype="uint64")).all()

    # test enough splits to go over path length
    for i in range(129):
        a, b = threefry_split(target, dev, b)
    assert (a[0:4] == b[0:4]).all(), "State part of split should be the same"
    assert (b[0:4] != np.zeros(4, dtype="uint64")).any()

    # check that split then generate does not generate the same for both sides
    a, a_rands = threefry_generate(target, dev, a, (100,))
    b, b_rands = threefry_generate(target, dev, b, (100,))
    assert (
        a_rands != b_rands
    ).all(), "Numbers generated from different initial states should be different"

    # check repeatability
    _, rands1 = threefry_generate(target, dev, a, (100,))
    _, rands2 = threefry_generate(target, dev, a, (100,))
    assert (
        rands1 == rands2
    ).all(), "Numbers generated from the same initial state should be the same"

    a1, b1 = threefry_split(target, dev, a)
    a2, b2 = threefry_split(target, dev, a)
    assert (a1 == a2).all() and (
        b1 == b2
    ).all(), "Split called on the same input should return the same result"


@tvm.testing.parametrize_targets
def test_threefry_generate(target, dev):
    gen = tvm.relay.random.threefry_key(0).data.numpy()

    # check that we can generate some data
    a, rands = threefry_generate(target, dev, gen, (2048,))
    assert (
        rands.shape[0] == 2048 and len(rands.shape) == 1
    ), "Output shape should match requested shape"

    # check that gen out does not equal input
    assert (a != gen).any(), "Output generator should be different from input generator"

    # check that we can generate data whose total number of elements is not a multiple of 4.
    a, rands = threefry_generate(target, dev, gen, (7,))
    assert (
        rands.shape[0] == 7 and len(rands.shape) == 1
    ), "Output shape should match requested shape"

    # test enough generates to go over generate limit
    gen = np.array(
        [0, 0, 0, 0, 0, 0, 0, 2 ** 64 - 2, 1 << 63, 0], dtype="uint64"
    )  # make counter large
    a, rands = threefry_generate(target, dev, gen, (2048,))
    assert gen[4] != a[4], "Overflow of counter should trigger path change"
    assert a[7] == 2048, "Overflow of counter should still update counter"

    # check generate with path at length limit
    gen = np.array([0, 0, 0, 0, 0, 0, 0, 2 ** 64 - 2, 0, 0], dtype="uint64")  # make counter large
    a, rands = threefry_generate(target, dev, gen, (2048,))
    assert (
        gen[0:4] != a[0:4]
    ).any(), "Overflowing counter with no space left in path should change state"


@tvm.testing.parametrize_targets
def test_threefry_wrapping(target, dev):
    assert tvm.topi.random.threefry_test_wrapping(
        target, dev
    ), f"{target} does not suppport wrapping unsigned integer arithmetic"


@tvm.testing.parametrize_targets
def test_uniform(target, dev):
    gen = tvm.relay.random.threefry_key(0).data.asnumpy()
    m = 1024
    n = 1024
    dtypes = ["float32", "float64"]
    for dtype in dtypes:
        low = np.array(5.0, dtype=dtype)
        high = np.array(10.0, dtype=dtype)
        new_gen, rands = uniform(target, dev, gen, low, high, (m, n), dtype)
        assert (gen != new_gen).any()
        assert abs(np.mean(rands) - 7.5) < 1e-1
        assert np.min(rands) >= 5.0
        assert np.max(rands) <= 10.0


if __name__ == "__main__":
    test_threefry_split(tvm.target.Target("llvm"), tvm.device("cpu"))
    test_threefry_generate(tvm.target.Target("llvm"), tvm.device("cpu"))
    test_threefry_wrapping(tvm.target.Target("llvm"), tvm.device("cpu"))
    test_uniform(tvm.target.Target("llvm"), tvm.device("cpu"))
