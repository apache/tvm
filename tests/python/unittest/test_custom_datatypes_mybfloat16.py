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
from ctypes import *
import topi
import tvm.ir_pass as ir_pass
import numpy as np

tgt = "llvm"


def setup():
    # You must first load the library containing the datatype implementation.
    # In this case, we have built the test functions used below right into TVM.
    # CDLL("libmybfloat16.so", RTLD_GLOBAL)

    tvm.datatype.register("bfloat", 129)

    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("FloatToBFloat16_wrapper"), "Cast",
        "llvm", "bfloat", "float")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("BFloat16ToFloat_wrapper"), "Cast",
        "llvm", "float", "bfloat")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("BFloat16Add_wrapper"), "Add", "llvm",
        "bfloat")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("FloatToBFloat16_wrapper"), "FloatImm",
        "llvm", "bfloat")

def lower_datatypes_and_build(schedule, args):
    """Create schedule and lower, manually lowering datatypes.

    Once datatype lowering is integrated directly into TVM's lower/build
    process, we won't need to do this manually.
    TODO(gus) integrate datatype lowering into build process; change this test"""
    flist = tvm.lower(schedule, args)
    flist = [flist]
    flist = [ir_pass.LowerCustomDatatypes(func, tgt) for func in flist]
    return tvm.build(flist[0], target=tgt)

def test_bfloat_add_and_cast_1():
    X = tvm.placeholder((3, ), name="X")
    Y = tvm.placeholder((3, ), name="Y")
    Z = topi.cast(
        topi.cast(X, dtype="custom[bfloat]16") +
        topi.cast(Y, dtype="custom[bfloat]16"),
        dtype="float")

    s = tvm.create_schedule([Z.op])
    built_cast = lower_datatypes_and_build(s, [X,Y,Z])

    ctx = tvm.context(tgt, 0)

    # Used float32 calculator at http://www.weitz.de/ieee/. Generated float32s
    # with at most 7-bit mantissas which, when added, produce a result with at
    # most 7-bit mantissas. This is to ensure there are no errors due to
    # float32->bfloat16 conversions.
    x = tvm.nd.array(
        np.array([4.4103796E-32, 14942208.0, 1.78125]).astype("float32"),
        ctx=ctx)
    y = tvm.nd.array(
        np.array([-3.330669E-14, 19660800.0, 2.25]).astype("float32"), ctx=ctx)
    z_expected = np.array([-3.330669E-14, 34603008.0,
                           4.03125]).astype("float32")
    z = tvm.nd.empty(Z.shape, dtype=Z.dtype, ctx=ctx)

    built_cast(x, y, z)

    assert np.array_equal(z_expected, z.asnumpy())


def test_bfloat_add_and_cast_2():
    X = tvm.placeholder((3, ), name="X")
    Y = tvm.placeholder((3, ), name="Y")
    Z = topi.cast(
        topi.cast(X, dtype="custom[bfloat]16") +
        topi.cast(Y, dtype="custom[bfloat]16"),
        dtype="float")

    s = tvm.create_schedule([Z.op])
    built_cast = lower_datatypes_and_build(s, [X,Y,Z])

    ctx = tvm.context(tgt, 0)

    # Used float32 calculator at http://www.weitz.de/ieee/. Generated
    # unconstrained float32s for the operands and copied them in to x and y.
    # Then, to simulate float32->bfloat16 conversion implemented by the mybfloat
    # library, I cut off all but 7 bits of the mantissa. I then added the
    # numbers. To simulate bfloat16 add implemented in mybfloat, I cut off all
    # but 7 bits of the result's mantissa. I then copied that value into
    # z_expected.
    x = tvm.nd.array(
        np.array([1.2348297, -1.0298302E25, 1.2034023E-30]).astype("float32"),
        ctx=ctx)
    y = tvm.nd.array(
        np.array([-2.4992788, -9.888288E19, 9.342338E-29]).astype("float32"),
        ctx=ctx)
    z_expected = np.array([-1.25, -1.027587E25,
                           9.426888E-29]).astype("float32")
    z = tvm.nd.empty(Z.shape, dtype=Z.dtype, ctx=ctx)

    built_cast(x, y, z)

    assert np.array_equal(z_expected, z.asnumpy())


def test_bfloat_add_and_cast_FloatImm():
    X = tvm.placeholder((3, ), name="X")
    Z = topi.cast(
        topi.add(
            topi.cast(X, dtype="custom[bfloat]16"),
            tvm.expr.FloatImm("custom[bfloat]16", 1.5)),
        dtype="float")

    s = tvm.create_schedule([Z.op])
    built_cast = lower_datatypes_and_build(s, [X,Z])

    ctx = tvm.context(tgt, 0)

    x = tvm.nd.array(np.array([0.0, 1.0, 1.5]).astype("float32"), ctx=ctx)
    z_expected = np.array([1.5, 2.5, 3.0]).astype("float32")
    z = tvm.nd.empty(Z.shape, dtype=Z.dtype, ctx=ctx)

    built_cast(x, z)

    assert np.array_equal(z_expected, z.asnumpy())


if __name__ == "__main__":
    setup()
    test_bfloat_add_and_cast_1()
    test_bfloat_add_and_cast_2()
    test_bfloat_add_and_cast_FloatImm()
