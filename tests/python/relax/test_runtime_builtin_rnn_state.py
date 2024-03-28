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
# pylint: disable=missing-docstring,
from typing import Sequence, Union

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import dlight as dl
from tvm import tir
from tvm.runtime import ShapeTuple
from tvm.script import tir as T

# pylint: disable=invalid-name

np_zero = np.full((16, 16), 0.0, "float16")
np_one = np.full((32, 32), 1.0, "float32")
np_two = np.full((16, 16), 2.0, "float16")
np_three = np.full((32, 32), 3.0, "float32")

reserved_nseq = 4
max_history = 4
num_layers = 1
device = tvm.cuda()
# Note that kernels in this test file cannot support 1-dim states.
states = [((16, 16), "float16"), ((32, 32), "float32")]

f_clear = None
f_add_sequence = None
f_remove_sequence = None
f_fork_sequence = None
f_popn = None
f_begin_forward = None
f_end_forward = None
f_get = None
f_set = None
f_debug_get = None

f_tir_gets = []
f_tir_sets = []

# pylint: enable=invalid-name


def set_global_func():
    global f_clear, f_add_sequence, f_remove_sequence, f_fork_sequence, f_popn
    global f_begin_forward, f_end_forward, f_get, f_set, f_debug_get
    global f_tir_gets, f_tir_sets

    f_clear = tvm.get_global_func("vm.builtin.kv_state_clear")
    f_add_sequence = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
    f_remove_sequence = tvm.get_global_func("vm.builtin.kv_state_remove_sequence")
    f_fork_sequence = tvm.get_global_func("vm.builtin.kv_state_fork_sequence")
    f_popn = tvm.get_global_func("vm.builtin.kv_state_popn")
    f_begin_forward = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    f_end_forward = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    f_get = tvm.get_global_func("vm.builtin.rnn_state_get")
    f_set = tvm.get_global_func("vm.builtin.rnn_state_set")
    f_debug_get = tvm.get_global_func("vm.builtin.rnn_state_debug_get")

    target = tvm.target.Target("cuda")

    def _build(tir_func):
        mod = tvm.IRModule({"main": tir_func})
        with target:
            mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)  # pylint: disable=not-callable
        f = tvm.build(mod["main"], target=target)
        return f.entry_func

    _f_tir_gets, _f_tir_sets = [], []
    for state in states:
        shape, dtype = state
        _f_tir_gets.append(_build(rnn_state_get(shape, dtype)))
        _f_tir_sets.append(_build(rnn_state_set(shape, dtype)))

    f_tir_gets = _f_tir_gets
    f_tir_sets = _f_tir_sets


def create_rnn_state():
    f_create = tvm.get_global_func("vm.builtin.rnn_state_create")
    init_values = [tvm.nd.array(np_zero, device=device), tvm.nd.array(np_one, device=device)]
    return f_create(num_layers, reserved_nseq, max_history, f_tir_gets, f_tir_sets, init_values)


@pytest.fixture
def rnn_state():
    set_global_func()
    return create_rnn_state()


def verify_state(state, seq_ids, expected_values):
    layer_id = 0
    for seq_id in seq_ids:
        for state_id, expected_value in enumerate(expected_values[seq_id]):
            state_value = f_debug_get(state, layer_id, state_id, seq_id)
            tvm.testing.assert_allclose(state_value.numpy(), expected_value)


@tvm.testing.requires_cuda
def test_rnn_state_get(rnn_state):  # pylint: disable=redefined-outer-name
    state = rnn_state
    f_clear(state)
    f_add_sequence(state, 0)
    f_begin_forward(state, ShapeTuple([0]), ShapeTuple([1]))
    tvm_nd_0 = tvm.nd.array(np.empty((1, 16, 16), "float16"), device=device)
    tvm_nd_1 = tvm.nd.array(np.empty((1, 32, 32), "float32"), device=device)
    f_get(state, 0, 0, tvm_nd_0)
    f_get(state, 0, 1, tvm_nd_1)
    f_end_forward(state)
    tvm.testing.assert_allclose(tvm_nd_0.numpy(), np.zeros((1, 16, 16), "float16"))
    tvm.testing.assert_allclose(tvm_nd_1.numpy(), np.ones((1, 32, 32), "float32"))


@tvm.testing.requires_cuda
def test_rnn_state_set(rnn_state):  # pylint: disable=redefined-outer-name
    state = rnn_state
    f_clear(state)
    for seq_id in range(3):
        f_add_sequence(state, seq_id)
    f_begin_forward(state, ShapeTuple([0, 2]), ShapeTuple([1, 1]))

    f_set(state, 0, 0, tvm.nd.array(np.full((2, 16, 16), 2.0, "float16"), device=device))
    f_set(state, 0, 1, tvm.nd.array(np.full((2, 32, 32), 3.0, "float32"), device=device))
    f_end_forward(state)

    expected_values = [[np_two, np_three], [np_zero, np_one], [np_two, np_three]]
    verify_state(state, [0, 1, 2], expected_values)


@tvm.testing.requires_cuda
def test_rnn_state_popn(rnn_state):  # pylint: disable=redefined-outer-name
    state = rnn_state
    f_clear(state)

    f_add_sequence(state, 0)
    f_begin_forward(state, ShapeTuple([0]), ShapeTuple([1]))
    f_set(state, 0, 0, tvm.nd.array(np_two.reshape(1, 16, 16), device=device))
    f_set(state, 0, 1, tvm.nd.array(np_three.reshape(1, 32, 32), device=device))
    f_end_forward(state)

    verify_state(state, [0], [[np_two, np_three]])
    f_popn(state, 0, 1)
    verify_state(state, [0], [[np_zero, np_one]])
    with pytest.raises(tvm.error.TVMError):
        f_popn(state, 0, 1)  # no available history to pop


@tvm.testing.requires_cuda
def test_rnn_state_fork_sequence(rnn_state):  # pylint: disable=redefined-outer-name
    state = rnn_state
    f_clear(state)

    f_add_sequence(state, 0)
    f_begin_forward(state, ShapeTuple([0]), ShapeTuple([1]))
    f_set(state, 0, 0, tvm.nd.array(np_two.reshape(1, 16, 16), device=device))
    f_set(state, 0, 1, tvm.nd.array(np_three.reshape(1, 32, 32), device=device))
    f_end_forward(state)
    f_fork_sequence(state, 0, 1, -1)
    verify_state(state, [0, 1], [[np_two, np_three], [np_two, np_three]])
    # Verify popn for the forked sequence
    f_popn(state, 1, 1)
    verify_state(state, [0, 1], [[np_two, np_three], [np_zero, np_one]])


def rnn_state_get(
    shape: Sequence[int],
    dtype: str,
):
    # fmt: off
    @T.prim_func
    def _rnn_state_get(
        var_storage: T.handle,
        var_seq_slot_ids: T.handle,
        var_history_slot_ids: T.handle,
        var_output: T.handle,
    ):
        batch_size = T.int32(is_size_var=True)

        storage = T.match_buffer(var_storage, (reserved_nseq, max_history, *shape), dtype)
        seq_slot_ids = T.match_buffer(var_seq_slot_ids, (batch_size,), "int32")
        history_slot_ids = T.match_buffer(var_history_slot_ids, (batch_size,), "int32")
        output = T.match_buffer(var_output, (batch_size, *shape), dtype)

        for i in range(batch_size):
            for s in T.grid(*shape):
                with T.block("copy"):
                    vi, *vs = T.axis.remap("S" * (len(shape) + 1), [i, *s])
                    seq_id: T.int32 = seq_slot_ids[vi]
                    history_id: T.int32 = history_slot_ids[vi]
                    # The following line is equivalent to:
                    # `output[vi, *vs] = storage[seq_id, history_id, *vs]`
                    # However, unpacking operator in subscript requires Python 3.11 or newer
                    T.buffer_store(
                        output, T.BufferLoad(storage, [seq_id, history_id, *vs]), [vi, *vs]
                    )
    # fmt: on
    return _rnn_state_get


def rnn_state_set(
    shape: Sequence[Union[int, tir.Var]],
    dtype: str,
):
    # fmt: off
    @T.prim_func
    def _rnn_state_set(
        var_storage: T.handle,
        var_seq_slot_ids: T.handle,
        var_history_slot_ids: T.handle,
        var_data: T.handle,
    ):
        batch_size = T.int32(is_size_var=True)

        storage = T.match_buffer(var_storage, (reserved_nseq, max_history, *shape), dtype)
        seq_slot_ids = T.match_buffer(var_seq_slot_ids, (batch_size,), "int32")
        history_slot_ids = T.match_buffer(var_history_slot_ids, (batch_size,), "int32")
        data = T.match_buffer(var_data, (batch_size, *shape), dtype)

        for i in range(batch_size):
            for s in T.grid(*shape):
                with T.block("copy"):
                    vi, *vs = T.axis.remap("S" * (len(shape) + 1), [i, *s])
                    seq_id: T.int32 = seq_slot_ids[vi]
                    history_id: T.int32 = (history_slot_ids[vi] + 1) % T.cast(
                        max_history, "int32"
                    )
                    # The following line is equivalent to:
                    # `storage[seq_id, history_id, *vs] = data[vi, *vs]`
                    # However, unpacking operator in subscript requires Python 3.11 or newer
                    T.buffer_store(
                        storage, T.BufferLoad(data, [vi, *vs]), [seq_id, history_id, *vs]
                    )

    # fmt: on

    return _rnn_state_set


if __name__ == "__main__":
    set_global_func()
    rnn_state = create_rnn_state()
    test_rnn_state_get(rnn_state)
    test_rnn_state_set(rnn_state)
    test_rnn_state_popn(rnn_state)
    test_rnn_state_fork_sequence(rnn_state)
