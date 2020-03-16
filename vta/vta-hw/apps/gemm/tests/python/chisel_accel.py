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
from tvm import te
import numpy as np
import tsim
import sys

""" Vector Bit Slice and Pack Function
Parameters
----------
A : Vector to be sliced and packed
slice_width : slice width

Returns
---------
C: 2d matrix where each cloumn (because of bit packing) represents each bit slice of A
"""
def slice(A, slice_width):
    assert np.log2(slice_width) % 1 == 0, "only power of 2 is supported"
    dtype = type(A[0])
    row = 0
    # currently only supports uint
    if dtype is np.uint8: row = 8 // slice_width
    elif dtype is np.uint16: row = 16 // slice_width
    elif dtype is np.uint32: row = 32 // slice_width
    elif dtype is np.uint64: row = 64 // slice_width
    else: raise ValueError("datatype currently not supported")
    if (row >= 8):
        dtype = 'uint' + str(row)
    else:
        dtype = 'uint8'

    C = np.zeros((row, len(A))).astype(dtype) # sliced and transform

    # create mask
    slice_mask = 2**(slice_width)-1
    # slice and pack
    for x in range(len(A)):
        for y in range(row):
            C[y][x] = (np.uint64(A[x]) >> np.uint64(slice_width * y)) & np.uint64(slice_mask)
    return C

def slice_mat(A, slice_width):
    assert np.log2(slice_width) % 1 == 0, "only power of 2 is supported"
    dtype = type(A[0][0])
    row = 0
    # currently only supports uint
    if dtype is np.uint8: row = 8 // slice_width
    elif dtype is np.uint16: row = 16 // slice_width
    elif dtype is np.uint32: row = 32 // slice_width
    elif dtype is np.uint64: row = 64 // slice_width
    else: raise ValueError("datatype currently not supported")
    if (row >= 8):
        dtype = 'uint' + str(row)
    else:
        dtype = 'uint8'

    # 3d array (bits, row, clmn)
    C = np.zeros((row, A.shape[0], A.shape[1])).astype(dtype) # sliced and transform

    # create mask
    slice_mask = 2**(slice_width)-1
    # slice and pack
    for z in range(A.shape[0]):
        C[:, z, :] = slice(A[z], slice_width)
    return C

""" Matrix Multiplication Function
Parameters
----------
A : Matrix A
B: Matrix B
i_width : weight slice width
w_width : activation slice width

Returns
---------
C: result of A * B
"""
# A is a n*m matrix, B is a m*p matrix(not transposed yet)
def matrix_multiply(A, B, i_width, w_width):
    assert A.shape[1] == B.shape[0], "can't perform multiplication"
    BT = B.transpose()
    cycles = 0
    B_sliced = slice_mat(BT, w_width)
    C = np.zeros((A.shape[0], B.shape[1])).astype('uint64')
    for i in range(A.shape[0]):
        A_sliced = slice(A[i], i_width)
        test = test_accel(A_sliced, B_sliced, i_width, w_width)
        C[i] = test[0]
        cycles += test[1]
        np.testing.assert_array_equal(C[i], compute(A_sliced, B_sliced, i_width, w_width))
        print("PASS row " + str(i))

    np.testing.assert_array_equal(C, np.matmul(A.astype('uint64'),B))
    print("result: ")
    print(C)
    print("TEST PASSED, cycles: " + str(cycles))
    return C

""" Software Verification Function
Parameter Dimesions
---------
A (bits, y) and B (bits, y, x) (transposed)

Takes 1 vector and 1 matrix input (sliced and packed)

Returns
---------
Resulting vector
"""
def compute(A, B, i_width, w_width):
    assert A.shape[1] == B.shape[1], "sliced shape not match"
    # reset hardware accumulator
    accum = np.zeros(A.shape[1])
    for x in range(A.shape[0]):
        for y in range(B.shape[0]):
            accum += np.matmul(A[x].astype('uint64'), B[y].transpose()) << np.uint64(x*i_width + y*w_width)
    # get value from accumulator
    return accum

"""Testing Function for Matrix Vector Multiplication"""
def test_accel(A, B, i_width, w_width):
    assert A.shape[1] == B.shape[2], "sliced shape not match"
    dtype = A.dtype
    ctx = tvm.cpu(0)
    f = tsim.load_module()

    a_arr = []
    b_arr = []
    for i in range(A.shape[0]):
        list_a = np.zeros(A.shape[1]).astype(dtype)
        for j in range(A.shape[1]):
            list_a[j] = A[i][j]
        a_arr.append(tvm.nd.array(list_a.astype(dtype), ctx))

    for i in range(B.shape[0]):
        # transpose
        list_b = np.zeros((B.shape[2], B.shape[1])).astype(dtype)
        for j in range(B.shape[2]):
            for k in range(B.shape[1]):
                list_b[j][k] = B[i][j][k]
        b_arr.append(tvm.nd.array(list_b.astype(dtype), ctx))

    cycles = 0
    accum = tvm.nd.array(np.zeros(A.shape[1]).astype("uint32"), ctx)
    for i in range(len(a_arr)):
        for j in range(len(b_arr)):
            shift = np.uint8(i*i_width + j*w_width)
            if i == 0 and j == 0:
                cycles += f(b_arr[j], a_arr[i], shift, accum, np.uint32(1)) # reset accumulator
            else:
                cycles += f(b_arr[j], a_arr[i], shift, accum, np.uint32(0)) # no reset

    return (accum.asnumpy(), cycles)

""" Matrix Generator
Parameters
----------
dtype : String, datatype generated (supports only uint)
i_width : weight bit slices(needs to be less than actual bit width)
w_width : activation bit slices(needs to be less than actual bit width)
"""
def top_test(dtype, i_width, w_width):

    # only supports positive values (up to 2**(bits-1))
    rmax = 127
    # (m,16) * (16,16) GEMM
    rrow = np.random.randint(7) + 1
    clmn = 16
    A = np.random.randint(rmax, size=(rrow,clmn)).astype(dtype)
    B = np.random.randint(rmax, size=(clmn,clmn)).astype(dtype)

    print("A: " + str(A))
    print("B: " + str(B))
    # perform GEMM
    matrix_multiply(A, B, i_width, w_width)

if __name__ == "__main__":
    tsim.init("chisel")
    for i in range(1):
        # reg1 and reg2 bits in hardware/chisel/src/main/Compute.scala must be modified for slices greater than 8 bits
        if sys.argv[1] == 'serial':
          # generates a random uint8 GEMM with 2-bit(8/4) input and 4-bit(8/2) weight
          top_test("uint8", 4, 2)
        elif sys.argv[1] == 'parallel':
          # generates a random uint8 GEMM with 8-bit input and 8-bit weight (bit parallel)
          top_test('uint8', 8, 8)
