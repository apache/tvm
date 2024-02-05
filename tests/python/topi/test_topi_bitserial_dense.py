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
"""Test code for bitserial_dense operator"""
import os
import numpy as np
import tvm
from tvm import te
from tvm import topi
import tvm.testing
import tvm.topi.testing
from tvm.topi.utils import get_const_tuple
from tvm.contrib.pickle_memoize import memoize

_bitserial_dense_implement = {
    "generic": (topi.nn.bitserial_dense, topi.generic.schedule_bitserial_dense),
    "cpu": (topi.x86.bitserial_dense, topi.x86.schedule_bitserial_dense),
    "arm_cpu": (topi.arm_cpu.bitserial_dense, topi.arm_cpu.schedule_bitserial_dense),
}


def generate_quantized_np(shape, bits, out_dtype):
    min_val = 0
    max_val = 1 << bits
    return np.random.randint(min_val, max_val, size=shape).astype(out_dtype)


def verify_bitserial_dense(batch, in_dim, out_dim, activation_bits, weight_bits, unipolar):
    out_dtype = "int16"

    def get_ref_data(a_shape, b_shape, input_dtype):
        a_np = generate_quantized_np(get_const_tuple(a_shape), activation_bits, input_dtype)
        b_np = generate_quantized_np(get_const_tuple(b_shape), weight_bits, input_dtype)
        if unipolar:
            b_ = np.copy(b_np).astype(out_dtype)
            for x in np.nditer(b_, op_flags=["readwrite"]):
                x[...] = 1 if x == 1 else -1
            c_np = np.dot(a_np, b_.T)
        else:
            c_np = np.dot(a_np, b_np.T)
        return a_np, b_np, c_np

    for target in ["llvm", "llvm -device=arm_cpu"]:
        target = tvm.target.Target(target)
        if "arm_cpu" in target.keys and "arm" not in os.uname()[4]:
            print("Skipped running code, not an arm device")
            continue
        input_dtype = "uint8" if "arm_cpu" in target.keys else "uint32"
        A = te.placeholder((batch, in_dim), dtype=input_dtype, name="A")
        B = te.placeholder((out_dim, in_dim), dtype=input_dtype, name="B")
        fcompute, fschedule = tvm.topi.testing.dispatch(target, _bitserial_dense_implement)
        C = fcompute(A, B, activation_bits, weight_bits, input_dtype, out_dtype, unipolar)
        s = fschedule([C])

        a_shape = get_const_tuple(A.shape)
        b_shape = get_const_tuple(B.shape)
        a_np, b_np, c_np = get_ref_data(a_shape, b_shape, input_dtype)

        dev = tvm.cpu(0)
        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(b_np, dev)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), dev)
        func = tvm.build(s, [A, B, C], target)
        func(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-5)


def test_bitserial_dense():
    verify_bitserial_dense(1, 1024, 1000, 1, 1, True)
    verify_bitserial_dense(1, 1024, 1000, 2, 1, True)

    verify_bitserial_dense(1, 1024, 1000, 1, 1, False)
    verify_bitserial_dense(1, 1024, 1000, 2, 1, False)


if __name__ == "__main__":
    test_bitserial_dense()
