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
import scipy.stats


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
        s,
        [gen_placeholder, low_placeholder, high_placeholder, left_placeholder, right_placeholder],
        target=target,
    )
    out_gen = tvm.nd.array(np.zeros(gen.shape, dtype="uint64"), device=dev)
    rands = tvm.nd.array(np.zeros(size, dtype=dtype), device=dev)
    f(
        tvm.nd.array(gen, device=dev),
        tvm.nd.array(low, device=dev),
        tvm.nd.array(high, device=dev),
        out_gen,
        rands,
    )
    return out_gen.numpy(), rands.asnumpy()


def multinomial(target, dev, gen, probs, num_samples):
    gen_placeholder = tvm.te.placeholder(gen.shape, name="gen", dtype="uint64")
    probs_placeholder = tvm.te.placeholder(probs.shape, name="probs", dtype="float32")
    new_gen_placeholder, indices_placeholder = tvm.topi.random.multinomial(
        gen_placeholder, probs_placeholder, num_samples
    )
    s = tvm.topi.generic.schedule_extern([new_gen_placeholder, indices_placeholder])
    f = tvm.build(
        s,
        [gen_placeholder, probs_placeholder, new_gen_placeholder, indices_placeholder],
        target=target,
    )
    out_gen = tvm.nd.array(np.zeros(gen.shape, dtype="uint64"), device=dev)
    indices = tvm.nd.array(np.zeros((*probs.shape[:-1], num_samples), dtype="int32"), device=dev)
    f(tvm.nd.array(gen), tvm.nd.array(probs), out_gen, indices)
    return out_gen.numpy(), indices.asnumpy()


@tvm.testing.parametrize_targets("llvm")
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


@tvm.testing.parametrize_targets("llvm")
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
        [0, 0, 0, 0, 0, 0, 0, 2**64 - 2, 1 << 63, 0], dtype="uint64"
    )  # make counter large
    a, rands = threefry_generate(target, dev, gen, (2048,))
    assert gen[4] != a[4], "Overflow of counter should trigger path change"
    assert a[7] == 2048, "Overflow of counter should still update counter"

    # check generate with path at length limit
    gen = np.array([0, 0, 0, 0, 0, 0, 0, 2**64 - 2, 0, 0], dtype="uint64")  # make counter large
    a, rands = threefry_generate(target, dev, gen, (2048,))
    assert (
        gen[0:4] != a[0:4]
    ).any(), "Overflowing counter with no space left in path should change state"


@tvm.testing.parametrize_targets("llvm")
def test_threefry_wrapping(target, dev):
    assert tvm.topi.random.threefry_test_wrapping(
        target, dev
    ), f"{target} does not suppport wrapping unsigned integer arithmetic"


@tvm.testing.parametrize_targets("llvm")
def test_uniform(target, dev):
    gen = tvm.relay.random.threefry_key(0).data.numpy()
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


@tvm.testing.parametrize_targets("llvm")
def test_multinomial(target, dev):
    def _verify_multinomial(size, num_samples, test_statistics=False):
        gen = tvm.relay.random.threefry_key(np.random.randint(0, 1e5)).data.numpy()
        probs = np.random.randint(low=-50, high=1000, size=size).astype("float32")
        new_gen, indices = multinomial(target, dev, gen, probs, num_samples)
        assert (gen != new_gen).any()
        assert np.min(indices) >= 0
        assert np.max(indices) < probs.shape[-1]
        # Note, only use test_statistics with sample size > 10,000.
        if test_statistics:
            # Clipped and normalized probabilities * number of samples
            # represents expected frequency of each category.
            # First upcast to float64 to remove numerical error.
            probs = probs.astype("float64")
            probs = np.reshape(probs, [-1, probs.shape[-1]])
            probs = np.maximum(probs, 0)
            probs = probs / np.expand_dims(np.sum(probs, axis=-1), axis=-1)
            # Multiply by number of samples and add epsilon to get non-zero expected samples per index.
            expected_frequency = probs * num_samples + np.finfo(float).eps
            # Do a small adjustment to make sure each row of expected_frequencies sums to exactly num_samples.
            expected_frequency = (
                np.expand_dims((num_samples / np.sum(expected_frequency, axis=-1)), axis=-1)
                * expected_frequency
            )
            # Reduce shape to a 2D matrix.
            indices = np.reshape(indices, [-1, indices.shape[-1]])
            # Split indendent rows of indices.
            index_list = [np.squeeze(x, 0) for x in np.split(indices, indices.shape[0], axis=0)]
            # Count frequency of selected indices in each row.
            observed_freqs = [np.bincount(samples, minlength=size[-1]) for samples in index_list]
            # Stack observed frequencies back into a matrix.
            observed_freqs = np.stack(observed_freqs, axis=0)
            # Test how closely observed samples match expectations.
            _, p_value = scipy.stats.chisquare(observed_freqs, expected_frequency, axis=-1)
            # If sampled correctly, p_value should be greater than 1e-6 almost all the time.
            assert np.all(p_value > 1e-6)

    # Test simple 1-D case.
    _verify_multinomial([3], 2)
    # Test 2-D case.
    _verify_multinomial([2, 10], 1)
    # Test 3-D case.
    _verify_multinomial([2, 3, 10], 4)
    # Test large sample size statistics.
    _verify_multinomial([3, 10], 10000, test_statistics=True)


if __name__ == "__main__":
    test_threefry_split(tvm.target.Target("llvm"), tvm.device("cpu"))
    test_threefry_generate(tvm.target.Target("llvm"), tvm.device("cpu"))
    test_threefry_wrapping(tvm.target.Target("llvm"), tvm.device("cpu"))
    test_uniform(tvm.target.Target("llvm"), tvm.device("cpu"))
    test_multinomial(tvm.target.Target("llvm"), tvm.device("cpu"))
