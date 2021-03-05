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


def threefry_split(target, ctx, gen):
    gen_placeholder = tvm.te.placeholder(gen.shape, name="gen", dtype="uint64")
    left_placeholder, right_placeholder = tvm.topi.random.threefry_split(gen_placeholder)
    s = tvm.topi.generic.schedule_extern([left_placeholder, right_placeholder])
    f = tvm.build(s, [gen_placeholder, left_placeholder, right_placeholder])
    left = tvm.nd.array(np.zeros(gen.shape, dtype="uint64"))
    right = tvm.nd.array(np.zeros(gen.shape, dtype="uint64"))
    f(tvm.nd.array(gen), left, right)
    return left.asnumpy(), right.asnumpy()


def threefry_generate(target, ctx, gen, size):
    gen_placeholder = tvm.te.placeholder(gen.shape, name="gen", dtype="uint64")
    left_placeholder, right_placeholder = tvm.topi.random.threefry_generate(gen_placeholder, size)
    s = tvm.topi.generic.schedule_extern([left_placeholder, right_placeholder])
    f = tvm.build(s, [gen_placeholder, left_placeholder, right_placeholder])
    out_gen = tvm.nd.array(np.zeros(gen.shape, dtype="uint64"))
    rands = tvm.nd.array(np.zeros(size, dtype="uint64"))
    f(tvm.nd.array(gen), out_gen, rands)
    return out_gen.asnumpy(), rands.asnumpy()


@tvm.testing.parametrize_targets
def test_threefry_split(target, ctx):
    # test that results of split do not equal eachother or the input
    gen = tvm.relay.random.threefry_key(0).data.asnumpy()
    a, b = threefry_split(target, ctx, gen)
    assert (a != b).any() and (
        a != gen
    ).any(), "Splitting a gen should result in different output gens"
    # unittest some split inputs
    assert (a == np.array([0, 0, 0, 0, 0, 0, 0, 0, 1 << 62, 0], dtype="uint64")).all()
    assert (b == np.array([0, 0, 0, 0, 1 << 63, 0, 0, 0, 1 << 62, 0], dtype="uint64")).all()

    # test enough splits to go over path length
    for i in range(129):
        a, b = threefry_split(target, ctx, b)
    assert (a[0:4] == b[0:4]).all(), "State part of split should be the same"
    assert (b[0:4] != np.zeros(4, dtype="uint64")).any()

    # check that split then generate does not generate the same for both sides
    a, a_rands = threefry_generate(target, ctx, a, (100,))
    b, b_rands = threefry_generate(target, ctx, b, (100,))
    assert (
        a_rands != b_rands
    ).all(), "Numbers generated from different initial states should be different"

    # check repeatability
    _, rands1 = threefry_generate(target, ctx, a, (100,))
    _, rands2 = threefry_generate(target, ctx, a, (100,))
    assert (
        rands1 == rands2
    ).all(), "Numbers generated from the same initial state should be the same"

    a1, b1 = threefry_split(target, ctx, a)
    a2, b2 = threefry_split(target, ctx, a)
    assert (a1 == a2).all() and (
        b1 == b2
    ).all(), "Split called on the same input should return the same result"


@tvm.testing.parametrize_targets
def test_threefry_generate(target, ctx):
    gen = tvm.relay.random.threefry_key(0).data.asnumpy()

    # check that we can generate some data
    a, rands = threefry_generate(target, ctx, gen, (100,))
    assert (
        rands.shape[0] == 100 and len(rands.shape) == 1
    ), "Output shape should match requested shape"

    # check that gen out does not equal input
    assert (a != gen).any(), "Output generator should be different from input generator"

    # test enough generates to go over generate limit
    gen = np.array(
        [0, 0, 0, 0, 0, 0, 0, 2 ** 64 - 2, 1 << 63, 0], dtype="uint64"
    )  # make counter large
    a, rands = threefry_generate(target, ctx, gen, (100,))
    assert gen[4] != a[4], "Overflow of counter should trigger path change"
    assert a[7] == 100, "Overflow of counter should still update counter"

    # check generate with path at length limit
    gen = np.array([0, 0, 0, 0, 0, 0, 0, 2 ** 64 - 2, 0, 0], dtype="uint64")  # make counter large
    a, rands = threefry_generate(target, ctx, gen, (100,))
    assert (
        gen[0:4] != a[0:4]
    ).any(), "Overflowing counter with no space left in path should change state"


@tvm.testing.parametrize_targets
def test_threefry_wrapping(target, ctx):
    assert tvm.topi.random.threefry_test_wrapping(
        target, ctx
    ), f"{target} does not suppport wrapping unsigned integer arithmetic"


if __name__ == "__main__":
    test_threefry_split(tvm.target.Target("llvm"), tvm.context("cpu"))
    test_threefry_generate(tvm.target.Target("llvm"), tvm.context("cpu"))
    test_threefry_wrapping(tvm.target.Target("llvm"), tvm.context("cpu"))
