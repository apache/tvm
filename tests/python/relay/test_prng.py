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
import tvm
import tvm.relay
import tvm.testing
from tvm.relay.testing import run_infer_type


@tvm.testing.parametrize_targets
def test_threefry_repeatability(target, dev):
    key1 = tvm.relay.random.threefry_key(1)
    rand1 = tvm.relay.random.threefry_generate(key1, (12,))
    out_key1, out1 = tvm.relay.create_executor(
        "vm", tvm.IRModule.from_expr(tvm.relay.Function([], rand1)), target=target, device=dev
    ).evaluate()()

    key2 = tvm.relay.random.threefry_key(1)
    rand2 = tvm.relay.random.threefry_generate(key2, (12,))
    out_key2, out2 = tvm.relay.create_executor(
        "vm", tvm.IRModule.from_expr(tvm.relay.Function([], rand2)), target=target, device=dev
    ).evaluate()()

    assert (
        out1.numpy() == out2.numpy()
    ).all(), "Generate on same seed should have the same output random numbers"

    assert (
        out_key1.numpy() == out_key2.numpy()
    ).all(), "Generate on same seed should have the same next keys"


@tvm.testing.parametrize_targets
def test_threefry_split(target, dev):
    key = tvm.relay.random.threefry_key(1)
    left, right = tvm.relay.TupleWrapper(tvm.relay.random.threefry_split(key), 2)
    _, rand1 = tvm.relay.TupleWrapper(tvm.relay.random.threefry_generate(left, (16,)), 2)
    _, rand2 = tvm.relay.TupleWrapper(tvm.relay.random.threefry_generate(right, (16,)), 2)
    out1, out2 = tvm.relay.create_executor(
        "vm",
        tvm.IRModule.from_expr(tvm.relay.Function([], tvm.relay.Tuple((rand1, rand2)))),
        target=target,
        device=dev,
    ).evaluate()()

    assert (
        out1.numpy() != out2.numpy()
    ).any(), "Generate after split should not have the same output"


@tvm.testing.parametrize_targets
def test_threefry_sequential_generate(target, dev):
    key = tvm.relay.random.threefry_key(1)
    key, rand1 = tvm.relay.TupleWrapper(tvm.relay.random.threefry_generate(key, (4,)), 2)
    _, rand2 = tvm.relay.TupleWrapper(tvm.relay.random.threefry_generate(key, (4,)), 2)
    out1, out2 = tvm.relay.create_executor(
        "vm",
        tvm.IRModule.from_expr(tvm.relay.Function([], tvm.relay.Tuple((rand1, rand2)))),
        target=target,
        device=dev,
    ).evaluate()()

    assert (
        out1.numpy() != out2.numpy()
    ).any(), "Sequential generates should not have the same output"


@tvm.testing.parametrize_targets
def test_threefry_sequential_generate_remaining(target, dev):
    key = tvm.relay.random.threefry_key(1)
    key, rand1 = tvm.relay.TupleWrapper(tvm.relay.random.threefry_generate(key, (7,)), 2)
    _, rand2 = tvm.relay.TupleWrapper(tvm.relay.random.threefry_generate(key, (7,)), 2)
    out1, out2 = tvm.relay.create_executor(
        "vm",
        tvm.IRModule.from_expr(tvm.relay.Function([], tvm.relay.Tuple((rand1, rand2)))),
        target=target,
        device=dev,
    ).evaluate()()

    assert (
        out1.numpy()[-3:] != out2.numpy()[-3:]
    ).any(), "Sequential generates should not have the same output"


def test_threefry_generate_infer():
    oshape = (12,)
    key_type = tvm.relay.TensorType([10], dtype="uint64")
    gen_type = tvm.relay.TensorType(oshape, dtype="uint64")
    expected_type = tvm.relay.TupleType([key_type, gen_type])

    key = tvm.relay.random.threefry_key(1)
    rand1 = tvm.relay.random.threefry_generate(key, oshape)
    f = tvm.relay.Function([], rand1)
    f = run_infer_type(f)
    tvm.ir.assert_structural_equal(f.ret_type, expected_type)


def test_threefry_split_infer():
    key_type = tvm.relay.TensorType([10], dtype="uint64")
    expected_type = tvm.relay.TupleType([key_type, key_type])

    key = tvm.relay.random.threefry_key(1)
    out_keys = tvm.relay.random.threefry_split(key)
    f = tvm.relay.Function([], out_keys)
    f = run_infer_type(f)
    tvm.ir.assert_structural_equal(f.ret_type, expected_type)


def test_uniform_infer():
    oshape = (12,)
    odtypes = ["float32", "float64"]
    for odtype in odtypes:
        key_type = tvm.relay.TensorType([10], dtype="uint64")
        gen_type = tvm.relay.TensorType(oshape, dtype=odtype)
        expected_type = tvm.relay.TupleType([key_type, gen_type])

        key = tvm.relay.random.threefry_key(1)
        rand1 = tvm.relay.random.uniform(key, oshape, odtype)
        f = tvm.relay.Function([], rand1)
        f = run_infer_type(f)
        tvm.ir.assert_structural_equal(f.ret_type, expected_type)


@pytest.mark.xfail(raises=tvm.error.TVMError)
def test_threefry_generate_infer_fail():
    # xfail: key size should be 10
    fake_key = tvm.relay.const([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype="uint64")
    rand1 = tvm.relay.random.threefry_generate(fake_key, (12,))
    f = tvm.relay.Function([], rand1)
    f = run_infer_type(f)


@pytest.mark.xfail(raises=tvm.error.TVMError)
def test_threefry_split_infer_fail():
    # xfail: key size should be 10
    fake_key = tvm.relay.const([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype="uint64")
    out_keys = tvm.relay.random.threefry_split(fake_key)
    f = tvm.relay.Function([], out_keys)
    f = run_infer_type(f)


@tvm.testing.requires_llvm
def test_threefry_generate_out_size():
    key = tvm.relay.random.threefry_key(1)
    key, rand1 = tvm.relay.TupleWrapper(tvm.relay.random.threefry_generate(key, (5,)), 2)
    out = tvm.relay.create_executor(
        "vm",
        tvm.IRModule.from_expr(tvm.relay.Function([], rand1)),
        target=tvm.target.Target("llvm"),
        device=tvm.device("cpu"),
    ).evaluate()()


if __name__ == "__main__":
    tvm.testing.main()
