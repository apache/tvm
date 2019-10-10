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
import numpy as np
import tsim
import sys

""" Vector Bit Slice and Pack Function
Parameters
----------
A : Vector to be sliced and packed
slice_width : slice width

Returnsi
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
    else: raise ValueError("datatype " + str(dtype) + "currently not supported")
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

""" Matrix Multiplication Function
Parameters
----------
A : Matrix A
B: Matrix B
w_width : weight slice width
a_width : activation slice width

Returns
---------
C: result of A * B
"""
# A is a n*m matrix, B is a m*p matrix(not transposed yet)
def matrix_multiply(A, B, w_width, a_width):
    assert A.shape[1] == B.shape[0], "can't perform multiplication"
    BT = B.transpose()
    cycles = 0
    C = np.zeros((A.shape[0], B.shape[1])).astype('uint64')
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            # C[i, j] = A[i].dot(BT[j])
            A_sliced = slice(A[i], w_width)
            B_sliced = slice(BT[j], a_width)

            C[i, j] = compute(A_sliced, B_sliced, w_width, a_width)
            test = test_accel(A_sliced, B_sliced, w_width, a_width)
            cycles += test[1]
            np.testing.assert_equal(C[i,j], A[i].astype('uint64').dot(BT[j]))
            print("PASS SW serial & parallel")

            np.testing.assert_equal(test[0], C[i, j])
            print("PASS SW & HW bit serial")

            np.testing.assert_equal(test[0], A[i].astype('uint64').dot(BT[j]))
            print("PASS SW bit parallel & HW bit parallel")

    print("result: ")
    print(C)
    print("ALL TESTS PASSED, cycles: " + str(cycles))
    return C

""" Software Verification Function"""
# takes 2 matrix input (sliced and packed)
def compute(A, B, w_width, a_width):
    assert A.shape[1] == B.shape[1], "sliced shape not match"
    # reset hardware accumulator
    accum = 0
    for x in range(A.shape[0]):
        for y in range(B.shape[0]):
            # hardware implementation
            accum += np.uint64(A[x]).dot(np.uint64(B[y])) << np.uint64(x*w_width + y*a_width)
    # get value from accumulator
    return accum

"""Testing Function for Dot Product"""
def test_accel(A, B, w_width, a_width):
    assert A.shape[1] == B.shape[1], "sliced shape not match"

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
        list_b = np.zeros(B.shape[1]).astype(dtype)
        for j in range(B.shape[1]):
            list_b[j] = B[i][j]
        b_arr.append(tvm.nd.array(list_b.astype(dtype), ctx))

    cycles = 0

    accum = tvm.nd.array(np.array([0]).astype("uint64"), ctx)
    for i in range(len(a_arr)):
        for j in range(len(b_arr)):
            shift = np.uint8(i*w_width + j*a_width)
            if i == 0 and j == 0: 
                cycles += f(a_arr[i], b_arr[j], shift, accum, np.uint32(1)) # reset accumulator
            else: 
                cycles += f(a_arr[i], b_arr[j], shift, accum, np.uint32(0)) # no reset

    return (accum.asnumpy()[0], cycles)

""" Matrix Generator
Parameters
----------     
dtype : String, datatype generated (supports only uint)
w_width : weight bit slices(needs to be less than actual bit width)
a_width : activation bit slices(needs to be less than actual bit width)
"""
def top_test(dtype, w_width, a_width):

    rmax = np.random.randint(256)
    # random matrix generation (dimension up to 8)
    rrow = np.random.randint(7) + 1
    rclmn = np.random.randint(7) + 1
    rrow2 = np.random.randint(7) + 1 
    A = np.random.randint(rmax, size=(rrow,rclmn)).astype(dtype)
    B = np.random.randint(rmax, size=(rclmn,rrow2)).astype(dtype)

    print("A: ")
    print(A)
    print("\n")
    print("B: ")
    print(B)
    print("\n")
    matrix_multiply(A, B, w_width, a_width)


if __name__ == "__main__":
    tsim.init("chisel")
    for i in range(1):
        # reg1 and reg2 bits in Compute.scala must be modified for slices greater than 8 bits
        if sys.argv[1] == 'serial':
          # generates a random uint8 GEMM with 2-bit(8/4) weight and 4-bit(8/2) activation 
          top_test("uint8",4, 2)
        elif sys.argv[1] == 'parallel':
          # generates a random uint8 GEMM with 8-bit weight and 8-bit activation (bit parallel) 
          top_test('uint8', 1, 1)
