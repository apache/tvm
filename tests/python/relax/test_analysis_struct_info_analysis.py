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

"""Tests analysis functions of struct info"""

import pytest

import tvm
import tvm.testing
from tvm import TVMError
from tvm import relax as rx
from tvm import tir, ir


def test_get_static_type_basic():
    # object
    s0 = rx.ObjectStructInfo()
    tvm.ir.assert_structural_equal(rx.analysis.get_static_type(s0), rx.ObjectType())

    # prim
    s1 = rx.PrimStructInfo("float32")
    tvm.ir.assert_structural_equal(rx.analysis.get_static_type(s1), tvm.ir.PrimType("float32"))


def test_get_static_type_shape():
    # shape
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")

    s2 = rx.ShapeStructInfo([1, n + 1, m])
    s3 = rx.ShapeStructInfo(ndim=2)

    tvm.ir.assert_structural_equal(rx.analysis.get_static_type(s2), rx.ShapeType(ndim=3))

    tvm.ir.assert_structural_equal(rx.analysis.get_static_type(s3), rx.ShapeType(ndim=2))


def test_get_static_type_tensor():
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    s4 = rx.TensorStructInfo([1, n + 1, m], "int64")

    tvm.ir.assert_structural_equal(
        rx.analysis.get_static_type(s4), rx.DynTensorType(ndim=3, dtype="int64")
    )


def test_get_static_type_tuple():
    # tuple
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    s0 = rx.ObjectStructInfo()
    s2 = rx.ShapeStructInfo([1, n + 1, m])
    s4 = rx.TensorStructInfo([1, n + 1, m], "int64")
    t0 = rx.TupleStructInfo([s4, s0])
    t1 = rx.TupleStructInfo([t0, s2])

    tvm.ir.assert_structural_equal(
        rx.analysis.get_static_type(t1),
        rx.TupleType(
            [
                rx.TupleType([rx.DynTensorType(ndim=3, dtype="int64"), rx.ObjectType()]),
                rx.ShapeType(ndim=3),
            ]
        ),
    )


def test_get_static_type_func():
    # tuple
    def fn_info(c):
        n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
        x = rx.TensorStructInfo([c, n, m], "float32")
        y = rx.TensorStructInfo([c, n, 1], "float32")
        z = rx.TensorStructInfo([c, n], "float32")
        return rx.FuncStructInfo([x, y], z)

    def fn_type():
        x = rx.DynTensorType(ndim=3, dtype="float32")
        y = rx.DynTensorType(ndim=3, dtype="float32")
        z = rx.DynTensorType(ndim=2, dtype="float32")
        return rx.FuncType([x, y], z)

    f0 = fn_info(1)

    tvm.ir.assert_structural_equal(rx.analysis.get_static_type(fn_info(1)), fn_type())


def test_erase_to_well_defined_basic():
    s0 = rx.ObjectStructInfo()
    tvm.ir.assert_structural_equal(rx.analysis.erase_to_well_defined(s0), s0)

    # prim
    s1 = rx.PrimStructInfo("float32")
    tvm.ir.assert_structural_equal(rx.analysis.erase_to_well_defined(s1), s1)


def test_erase_to_well_defined_shape():
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")

    s2 = rx.ShapeStructInfo([1, n + 1, m])
    s3 = rx.ShapeStructInfo(ndim=2)
    # have undefined
    tvm.ir.assert_structural_equal(
        rx.analysis.erase_to_well_defined(s2), rx.ShapeStructInfo(ndim=3)
    )
    # all defined
    tvm.ir.assert_structural_equal(rx.analysis.erase_to_well_defined(s2, {n: n, m: m}), s2)

    # replacement
    tvm.ir.assert_structural_equal(
        rx.analysis.erase_to_well_defined(s2, {n: 2, m: m + 1}), rx.ShapeStructInfo([1, 3, m + 1])
    )

    # partial defined
    tvm.ir.assert_structural_equal(
        rx.analysis.erase_to_well_defined(s2, {n: n}), rx.ShapeStructInfo(ndim=3)
    )


def test_erase_to_well_defined_tensor():
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    rshape = rx.Var("shape", rx.ShapeStructInfo(ndim=2))
    s0 = rx.TensorStructInfo(rshape, dtype="int32")

    # undefined
    tvm.ir.assert_structural_equal(
        rx.analysis.erase_to_well_defined(s0, None, None),
        rx.TensorStructInfo(ndim=2, dtype="int32"),
    )

    # defined
    tvm.ir.assert_structural_equal(
        rx.analysis.erase_to_well_defined(s0, None, {rshape: rshape}), s0
    )

    tvm.ir.assert_structural_equal(
        rx.analysis.erase_to_well_defined(s0, None, {rshape: rx.ShapeExpr([1, 2])}),
        rx.TensorStructInfo([1, 2], dtype="int32"),
    )

    s1 = rx.TensorStructInfo([m + 1, n], dtype="float32")

    tvm.ir.assert_structural_equal(rx.analysis.erase_to_well_defined(s1, {n: n, m: m}), s1)

    tvm.ir.assert_structural_equal(
        rx.analysis.erase_to_well_defined(s1, {n: 2, m: 3}),
        rx.TensorStructInfo([4, 2], dtype="float32"),
    )

    tvm.ir.assert_structural_equal(
        rx.analysis.erase_to_well_defined(s1, {m: m}, {rshape: rshape}),
        rx.TensorStructInfo(ndim=2, dtype="float32"),
    )

    s2 = rx.TensorStructInfo([1, 2], dtype="float32")

    tvm.ir.assert_structural_equal(rx.analysis.erase_to_well_defined(s2), s2)


def test_erase_to_well_defined_tuple():
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    s0 = rx.ObjectStructInfo()
    s2 = rx.ShapeStructInfo([1, m])
    s4 = rx.TensorStructInfo([1, n + 1, m], "int64")
    t0 = rx.TupleStructInfo([s4, s0])
    t1 = rx.TupleStructInfo([t0, s2])

    tvm.ir.assert_structural_equal(
        rx.analysis.erase_to_well_defined(t1, {m: m + 1}),
        rx.TupleStructInfo(
            [
                rx.TupleStructInfo(
                    [rx.TensorStructInfo(ndim=3, dtype="int64"), rx.ObjectStructInfo()]
                ),
                rx.ShapeStructInfo([1, m + 1]),
            ]
        ),
    )


def test_erase_to_well_defined_func():
    def fn_info(c):
        n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
        x = rx.TensorStructInfo([c, n, m], "float32")
        y = rx.TensorStructInfo([c, n, 1], "float32")
        z = rx.TensorStructInfo([c, n], "float32")
        return rx.FuncStructInfo([x, y], z)

    f0 = fn_info(1)

    tvm.ir.assert_structural_equal(rx.analysis.erase_to_well_defined(f0), f0)


def test_base_check():
    BR = rx.analysis.BaseCheckResult
    bcheck = rx.analysis.struct_info_base_check

    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    obj0 = rx.ObjectStructInfo()
    prim0 = rx.PrimStructInfo("int32")
    prim1 = rx.PrimStructInfo("float32")

    shape0 = rx.ShapeStructInfo(ndim=-1)
    shape1 = rx.ShapeStructInfo(ndim=2)
    shape2 = rx.ShapeStructInfo(ndim=3)
    shape3 = rx.ShapeStructInfo([1, 2, 3])
    shape4 = rx.ShapeStructInfo([1, n, 3])

    vdevice0 = ir.VDevice()
    vdevice1 = ir.VDevice("llvm")
    vdevice2 = ir.VDevice("cuda", 0)
    vdevice3 = ir.VDevice("cuda", 2)
    vdevice4 = ir.VDevice("cuda", 0, "")

    tensor0 = rx.TensorStructInfo(ndim=-1, dtype="int32")
    tensor1 = rx.TensorStructInfo(ndim=-1, dtype="float32")
    tensor2 = rx.TensorStructInfo(ndim=2, dtype="int32")
    tensor3 = rx.TensorStructInfo(ndim=2, dtype="float32")
    tensor4 = rx.TensorStructInfo([n, m], "int32")
    tensor5 = rx.TensorStructInfo([n, m, 1], "int32")
    tensor6 = rx.TensorStructInfo([n, m, 2], "int32")
    tensor7 = rx.TensorStructInfo(ndim=2, dtype="float32", vdevice=vdevice0)
    tensor8 = rx.TensorStructInfo(ndim=2, dtype="float32", vdevice=vdevice1)
    tensor9 = rx.TensorStructInfo(ndim=2, dtype="float32", vdevice=vdevice2)
    tensor10 = rx.TensorStructInfo(ndim=2, dtype="float32", vdevice=vdevice3)
    tensor11 = rx.TensorStructInfo(ndim=2, dtype="float32", vdevice=vdevice4)
    tensor12 = rx.TensorStructInfo([n, m, 2], "int32", vdevice0)
    tensor13 = rx.TensorStructInfo([n, m, 2], "int32", vdevice1)
    tensor14 = rx.TensorStructInfo([n, m, 2], "int32", vdevice2)
    tensor15 = rx.TensorStructInfo([n, m, 2], "int32", vdevice3)
    tensor16 = rx.TensorStructInfo([n, m, 2], "int32", vdevice4)

    # obj
    assert bcheck(obj0, prim0) == BR.PASS
    assert bcheck(obj0, shape1) == BR.PASS
    assert bcheck(obj0, tensor2) == BR.PASS
    assert obj0.is_base_of(tensor2)

    # prim
    assert prim0.is_base_of(prim0)
    assert not prim0.is_base_of(prim1)
    assert bcheck(prim0, obj0) == BR.FAIL_L1
    assert bcheck(prim0, prim0) == BR.PASS
    assert bcheck(prim0, prim1) == BR.FAIL_L0

    # shape
    assert bcheck(shape0, obj0) == BR.FAIL_L1
    assert bcheck(shape0, prim0) == BR.FAIL_L0

    # unknown dim
    assert bcheck(shape0, shape1) == BR.PASS
    assert bcheck(shape1, shape0) == BR.FAIL_L1

    # ndim mismatch
    assert bcheck(shape1, shape2) == BR.FAIL_L0

    # lhs do not have symbolic value but ndim match
    assert bcheck(shape2, shape3) == BR.PASS

    # rhs do not symbolic but lhs do
    assert bcheck(shape3, shape2) == BR.FAIL_L2

    # shape mismatch
    assert bcheck(shape3, shape4) == BR.FAIL_L2
    assert shape4.is_base_of(rx.ShapeStructInfo([1, n, 3]))

    # tensor
    assert bcheck(tensor0, obj0) == BR.FAIL_L1
    assert bcheck(tensor0, prim0) == BR.FAIL_L0
    assert bcheck(tensor0, shape0) == BR.FAIL_L0

    # dtype mismatch
    assert bcheck(tensor0, tensor1) == BR.FAIL_L0
    assert bcheck(tensor0, tensor3) == BR.FAIL_L0
    assert bcheck(tensor3, tensor4) == BR.FAIL_L0
    assert bcheck(tensor1, tensor2) == BR.FAIL_L0

    # vdevice mismatch
    assert bcheck(tensor8, tensor9) == BR.FAIL_L0
    assert bcheck(tensor9, tensor10) == BR.FAIL_L0
    assert bcheck(tensor10, tensor11) == BR.FAIL_L0
    assert bcheck(tensor13, tensor14) == BR.FAIL_L0
    assert bcheck(tensor14, tensor15) == BR.FAIL_L0
    assert bcheck(tensor15, tensor16) == BR.FAIL_L0

    # ndim mismatch
    assert bcheck(tensor2, tensor5) == BR.FAIL_L0

    # static shape mismatch
    assert bcheck(tensor5, tensor6) == BR.FAIL_L0

    # match
    assert tensor0.is_base_of(rx.TensorStructInfo(ndim=-1, dtype="int32"))
    assert tensor0.is_base_of(tensor2)
    assert tensor0.is_base_of(tensor4)
    assert tensor0.is_base_of(tensor5)
    assert tensor0.is_base_of(tensor6)
    assert tensor2.is_base_of(tensor4)
    assert tensor3.is_base_of(tensor7)
    assert tensor3.is_base_of(tensor8)
    assert tensor6.is_base_of(tensor12)
    assert tensor6.is_base_of(tensor13)
    assert tensor4.is_base_of(rx.TensorStructInfo([n, m], dtype="int32"))

    # tuple
    t0 = rx.TupleStructInfo([obj0, tensor0])
    t1 = rx.TupleStructInfo([prim0, tensor4])
    t2 = rx.TupleStructInfo([obj0, tensor0, obj0])
    t3 = rx.TupleStructInfo([tensor0, obj0])

    assert t0.is_base_of(t1)

    assert bcheck(t0, t2) == BR.FAIL_L0
    assert bcheck(t0, t3) == BR.FAIL_L1

    assert rx.TupleStructInfo([t0, t1]).is_base_of(rx.TupleStructInfo([t1, t1]))
    assert bcheck(rx.TupleStructInfo([t0, t1]), rx.TupleStructInfo([t1, t0])) == BR.FAIL_L1

    def fn_info_shape(c):
        n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
        x = rx.TensorStructInfo([c, n, m], "float32")
        y = rx.TensorStructInfo([c, n, 1], "float32")
        z = rx.TensorStructInfo([c, n], "float32")
        return rx.FuncStructInfo([x, y], z)

    def fn_info_erased():
        x = rx.TensorStructInfo(ndim=3, dtype="float32")
        y = rx.TensorStructInfo(ndim=3, dtype="float32")
        z = rx.TensorStructInfo(ndim=2, dtype="float32")
        return rx.FuncStructInfo([x, y], z)

    assert fn_info_shape(1).is_base_of(fn_info_shape(1))
    assert fn_info_erased().is_base_of(fn_info_shape(1))
    assert bcheck(fn_info_shape(1), fn_info_erased()) == BR.FAIL_L2

    fopaque = rx.FuncStructInfo.opaque_func()
    assert fopaque.is_base_of(fn_info_shape(1))


def _check_derive(ctx, finfo, args_sinfo, ret):
    gv = rx.GlobalVar("test")
    rx.expr._update_struct_info(gv, finfo)
    args = []
    for i, sinfo in enumerate(args_sinfo):
        arg = rx.Var("arg%i" % i, sinfo)
        args.append(arg)
    call = rx.Call(gv, args)
    derived_ret = rx.analysis.derive_call_ret_struct_info(finfo, call, ctx)
    tvm.ir.assert_structural_equal(ret, derived_ret)


def test_derive_call_ret_struct_info():
    obj0 = rx.ObjectStructInfo()
    prim0 = rx.PrimStructInfo("float32")

    n, m = tir.Var("n0", "int64"), tir.Var("m0", "int64")
    bb = rx.BlockBuilder()
    # derivation cases
    with bb.testing_scope(def_vars=[n, m]):

        def func0(c):
            n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
            x = rx.TensorStructInfo([n, m], "float32")
            z = rx.TensorStructInfo([m + c, n], "float32")
            return rx.FuncStructInfo([x], z)

        # Tensor => Tensor
        _check_derive(
            bb,
            func0(1),
            [rx.TensorStructInfo([10, 11], "float32")],
            rx.TensorStructInfo([12, 10], "float32"),
        )

        _check_derive(
            bb,
            func0(2),
            [rx.TensorStructInfo([n, m], "float32")],
            rx.TensorStructInfo([m + 2, n], "float32"),
        )

        # passing in information that cannot deduce n, m
        # it is still OK as type still matches, return an
        # eriased output
        _check_derive(
            bb,
            func0(2),
            [rx.TensorStructInfo(ndim=2, dtype="float32")],
            rx.TensorStructInfo(ndim=2, dtype="float32"),
        )

        # Error: wrong number of arguments
        with pytest.raises(TVMError):
            _check_derive(
                bb,
                func0(2),
                [rx.TensorStructInfo(ndim=2, dtype="float32"), obj0],
                rx.TensorStructInfo(ndim=2, dtype="float32"),
            )

        # Error:type mismatch
        with pytest.raises(TVMError):
            _check_derive(bb, func0(2), [obj0], obj0)

        # Tensor with vdevice
        vdev = ir.VDevice("llvm")

        def func1(c):
            n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
            x = rx.TensorStructInfo([n, m], "float32", vdev)
            z = rx.TensorStructInfo([m + c, n], "float32", vdev)
            return rx.FuncStructInfo([x], z)

        _check_derive(
            bb,
            func1(1),
            [rx.TensorStructInfo([10, 11], "float32", vdev)],
            rx.TensorStructInfo([12, 10], "float32", vdev),
        )

        # opaque derivation
        fopaque0 = lambda: rx.FuncStructInfo.opaque_func()
        fopaque1 = lambda: rx.FuncStructInfo.opaque_func(ret=prim0)
        _check_derive(bb, fopaque0(), [obj0, prim0], obj0)
        _check_derive(bb, fopaque1(), [obj0, prim0], prim0)

        # recursive tuple derivation
        def func_tuple0(c):
            n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
            x0 = rx.TensorStructInfo([n, c], "float32")
            x1 = rx.TensorStructInfo([n + c, m], "float32")
            z = rx.TupleStructInfo([rx.TensorStructInfo([m, n], "float32")])
            return rx.FuncStructInfo([rx.TupleStructInfo([x0, x1])], z)

        _check_derive(
            bb,
            func_tuple0(2),
            [
                rx.TupleStructInfo(
                    [
                        rx.TensorStructInfo([n, 2], "float32"),
                        rx.TensorStructInfo([n + 2, 10], "float32"),
                    ]
                )
            ],
            rx.TupleStructInfo([rx.TensorStructInfo([10, n], "float32")]),
        )

        def func_tuple1(c):
            n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
            x0 = rx.TensorStructInfo([n, m], "float32")
            x1 = rx.TensorStructInfo([n + c, c], "float32")
            z = rx.TupleStructInfo([rx.TensorStructInfo([m, n], "float32")])
            return rx.FuncStructInfo([rx.TupleStructInfo([x0, x1])], z)

        # Still OK, to pass erased tensor into n+2, n is captured by other argument.
        _check_derive(
            bb,
            func_tuple1(4),
            [
                rx.TupleStructInfo(
                    [
                        rx.TensorStructInfo([n, 4], "float32"),
                        rx.TensorStructInfo(ndim=2, dtype="float32"),
                    ]
                )
            ],
            rx.TupleStructInfo([rx.TensorStructInfo([4, n], "float32")]),
        )

        # tuple length mismatch is not causes an error
        with pytest.raises(TVMError):
            _check_derive(
                bb,
                func_tuple0(4),
                [rx.TupleStructInfo([rx.TensorStructInfo([n, 4], "float32")])],
                rx.TupleStructInfo([rx.TensorStructInfo([10, n], "float32")]),
            )

        # mixed shape types
        def func_shape_mixed(c):
            n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
            x0 = rx.ShapeStructInfo([n, m])
            f0 = func_tuple0(c)
            z = rx.ShapeStructInfo([m + n, c])
            return rx.FuncStructInfo([x0, f0], z)

        _check_derive(
            bb,
            func_shape_mixed(3),
            [
                rx.ShapeStructInfo([10, 20]),
                # have to specify purity because an impure function cannot be passed
                # where a pure one is expected
                rx.FuncStructInfo.opaque_func(ret=rx.ShapeStructInfo(ndim=2), purity=True),
            ],
            rx.ShapeStructInfo([30, 3]),
        )


def _check_lca(lhs, rhs, target):
    tvm.ir.assert_structural_equal(rx.analysis.struct_info_lca(lhs, rhs), target)
    tvm.ir.assert_structural_equal(rx.analysis.struct_info_lca(rhs, lhs), target)


def test_struct_info_lca():
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    obj0 = rx.ObjectStructInfo()
    prim0 = rx.PrimStructInfo("int32")
    prim1 = rx.PrimStructInfo("float32")

    vdevice0 = ir.VDevice("llvm")
    vdevice1 = ir.VDevice("cuda", 0)

    shape0 = rx.ShapeStructInfo(ndim=-1)
    shape1 = rx.ShapeStructInfo(ndim=2)
    shape2 = rx.ShapeStructInfo(ndim=3)
    shape3 = rx.ShapeStructInfo([1, 2, 3])
    shape4 = rx.ShapeStructInfo([1, n, 3])

    tensor0 = rx.TensorStructInfo(ndim=-1, dtype="int32")
    tensor1 = rx.TensorStructInfo(ndim=-1, dtype="float32")
    tensor2 = rx.TensorStructInfo(ndim=2, dtype="int32")
    tensor3 = rx.TensorStructInfo(ndim=2, dtype="float32")
    tensor4 = rx.TensorStructInfo([n, m], "int32")
    tensor5 = rx.TensorStructInfo([n, m, 1], "int32")
    tensor6 = rx.TensorStructInfo([n, m, 2], "int32")
    tensor7 = rx.TensorStructInfo(ndim=2, dtype="float32", vdevice=vdevice0)
    tensor8 = rx.TensorStructInfo(ndim=2, dtype="float32", vdevice=vdevice1)
    tensor9 = rx.TensorStructInfo([n, m, 2], "int32", vdevice0)
    tensor10 = rx.TensorStructInfo([n, m, 2], "int32", vdevice1)

    # obj
    _check_lca(obj0, prim0, obj0)
    _check_lca(obj0, prim1, obj0)

    # shape
    _check_lca(shape0, tensor0, obj0)
    _check_lca(shape0, shape1, shape0)
    _check_lca(shape1, shape2, shape0)
    _check_lca(shape1, shape3, shape0)

    _check_lca(shape2, shape3, shape2)
    _check_lca(shape3, shape4, shape2)
    _check_lca(shape4, rx.ShapeStructInfo([1, n, 3]), shape4)

    # tensor
    _check_lca(tensor0, prim0, obj0)
    _check_lca(tensor0, tensor1, rx.TensorStructInfo(ndim=-1, dtype=None))
    _check_lca(tensor0, tensor2, tensor0)
    _check_lca(tensor0, tensor4, tensor0)
    _check_lca(tensor0, tensor4, tensor0)
    _check_lca(tensor1, tensor3, tensor1)
    _check_lca(tensor3, tensor7, tensor3)
    _check_lca(tensor3, tensor8, tensor3)
    _check_lca(tensor1, tensor8, tensor1)
    _check_lca(tensor6, tensor9, tensor6)
    _check_lca(tensor6, tensor10, tensor6)

    _check_lca(tensor2, tensor4, tensor2)
    _check_lca(tensor5, tensor6, rx.TensorStructInfo(ndim=3, dtype="int32"))
    _check_lca(tensor4, tensor5, rx.TensorStructInfo(ndim=-1, dtype="int32"))
    _check_lca(tensor4, rx.TensorStructInfo([n, m], dtype="int32"), tensor4)

    # tuple
    t0 = rx.TupleStructInfo([obj0, tensor0])
    t1 = rx.TupleStructInfo([prim0, tensor4])
    t2 = rx.TupleStructInfo([obj0, tensor0, obj0])
    t3 = rx.TupleStructInfo([tensor0, obj0])

    _check_lca(t0, t1, t0)
    _check_lca(t0, t2, obj0)
    _check_lca(t0, t3, rx.TupleStructInfo([obj0, obj0]))

    t5 = rx.TupleStructInfo([t0, t1])
    t6 = rx.TupleStructInfo([t1, t2])

    _check_lca(t5, t6, rx.TupleStructInfo([t0, obj0]))

    t7 = rx.TupleStructInfo([])
    _check_lca(t7, rx.TupleStructInfo([]), t7)

    def fn_info_shape(c):
        n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
        x = rx.TensorStructInfo([c, n, m], "float32")
        y = rx.TensorStructInfo([c, n, 1], "float32")
        z = rx.TensorStructInfo([c, n], "float32")
        return rx.FuncStructInfo([x, y], z)

    def fn_info_erased():
        x = rx.TensorStructInfo(ndim=3, dtype="float32")
        y = rx.TensorStructInfo(ndim=3, dtype="float32")
        z = rx.TensorStructInfo(ndim=2, dtype="float32")
        return rx.FuncStructInfo([x, y], z)

    fopaque0 = lambda: rx.FuncStructInfo.opaque_func()
    fopaque1 = lambda: rx.FuncStructInfo.opaque_func(ret=prim0)
    fopaque2 = lambda: rx.FuncStructInfo.opaque_func(
        ret=rx.TensorStructInfo(ndim=2, dtype="float32")
    )

    _check_lca(fn_info_shape(1), fn_info_shape(2), fn_info_erased())
    _check_lca(fn_info_shape(2), fn_info_shape(2), fn_info_shape(2))

    _check_lca(fopaque0(), fopaque1(), fopaque0())
    _check_lca(fopaque0(), fn_info_shape(1), fopaque0())
    _check_lca(fopaque2(), fn_info_shape(1), fopaque2())


def _generate_tir_var_test_cases():
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    shape0 = rx.ShapeStructInfo([1, n, 3])
    shape1 = rx.ShapeStructInfo([1, 2 * n, n, m])
    shape2 = rx.ShapeStructInfo([1, 2 * n, m])
    tensor0 = rx.TensorStructInfo([1, n, 3], "int32")
    tensor1 = rx.TensorStructInfo([1, 2 * n, n, m], "int32")
    tensor2 = rx.TensorStructInfo([1, 2 * n, m], "int32")
    func = rx.FuncStructInfo(
        [rx.TensorStructInfo([1, 2 * n, n, m], "int32")], rx.TensorStructInfo([1, n, 3], "int32")
    )

    yield shape0, [n], [n]
    yield shape1, [n, m], [n, m]
    yield shape2, [m], [n, m]
    yield tensor0, [n], [n]
    yield tensor1, [n, m], [n, m]
    yield tensor2, [m], [n, m]
    yield func, [n, m], [n, m]


tir_var_test_case = tvm.testing.parameter(*_generate_tir_var_test_cases())


def test_tir_vars_in_struct_info(tir_var_test_case):
    sinfo, _vars_definable, vars_used = tir_var_test_case
    tvm.ir.assert_structural_equal(rx.analysis.tir_vars_in_struct_info(sinfo), vars_used)


def test_definable_tir_vars_in_struct_info(tir_var_test_case):
    sinfo, vars_definable, _vars_used = tir_var_test_case
    tvm.ir.assert_structural_equal(
        rx.analysis.definable_tir_vars_in_struct_info(sinfo), vars_definable
    )


def test_collect_symbolic_var_from_tensor_shape():
    n, m, k, q, p = (
        tir.Var("n", "int64"),
        tir.Var("m", "int64"),
        tir.Var("k", "int64"),
        tir.Var("q", "int64"),
        tir.Var("p", "int64"),
    )
    bb = rx.BlockBuilder()
    x = rx.Var("x", rx.TensorStructInfo([m, m + n], "float32"))
    with bb.function("main", [x]):
        v0 = bb.match_cast(x, rx.TensorStructInfo([m, k], "float32"))
        v1 = bb.emit(rx.call_dps_packed("test", x, rx.TensorStructInfo([p, q], "float32")))
        bb.emit_func_output(rx.const(1))
    func = bb.get()["main"]

    defined_vars = set(rx.analysis.defined_symbolic_vars(func))
    free_vars = set(rx.analysis.free_symbolic_vars(func))
    assert defined_vars == {m, k}
    assert free_vars == {n, p, q}


param_type = tvm.testing.parameter("shape_expr", "prim_value")
param_order = tvm.testing.parameter("definition_first", "usage_first")


def test_collect_symbolic_var_from_non_tensor_params(param_type, param_order):
    tir_n = tir.Var("n", "int64")
    tir_m = tir.Var("m", "int64")

    bb = rx.BlockBuilder()
    arg = rx.Var("arg", rx.TensorStructInfo([tir_n * tir_m]))

    if param_type == "shape_expr":
        extra_params = [
            rx.Var("shape_expr", rx.ShapeStructInfo([tir_n, tir_m])),
        ]
    elif param_type == "prim_value":
        extra_params = [
            rx.Var("n", rx.PrimStructInfo(value=tir_n)),
            rx.Var("m", rx.PrimStructInfo(value=tir_m)),
        ]
    else:
        raise ValueError(f"Unknown param_type: {param_type}")

    if param_order == "definition_first":
        params = [*extra_params, arg]
    elif param_order == "usage_first":
        params = [arg, *extra_params]
    else:
        raise ValueError(f"Unknown param_order: {param_order}")

    with bb.function("main", params=params):
        out = rx.op.reshape(arg, [tir_n, tir_m])
        bb.emit_func_output(out)
    func = bb.get()["main"]

    defined_vars = set(rx.analysis.defined_symbolic_vars(func))
    free_vars = set(rx.analysis.free_symbolic_vars(func))
    assert defined_vars == {tir_n, tir_m}
    assert free_vars == set()


if __name__ == "__main__":
    tvm.testing.main()
