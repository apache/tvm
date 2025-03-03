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
import tvm.testing
from tvm import relax, tir
from tvm import TVMError
from tvm.ir import Op, VDevice
from tvm.script import relax as R


def test_conv1d_op_correctness():
    x = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    w = relax.Var("w", R.Tensor((4, 3, 3), "float32"))
    assert relax.op.nn.conv1d(x, w).op == Op.get("relax.nn.conv1d")
    assert relax.op.nn.conv1d_transpose(x, w).op == Op.get("relax.nn.conv1d_transpose")


def test_conv2d_op_correctness():
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    w = relax.Var("w", R.Tensor((4, 3, 3, 3), "float32"))
    assert relax.op.nn.conv2d(x, w).op == Op.get("relax.nn.conv2d")
    assert relax.op.nn.conv2d_transpose(x, w).op == Op.get("relax.nn.conv2d_transpose")


def test_conv3d_op_correctness():
    x = relax.Var("x", R.Tensor((2, 3, 28, 28, 28), "float32"))
    w = relax.Var("w", R.Tensor((4, 3, 3, 3, 3), "float32"))
    assert relax.op.nn.conv3d(x, w).op == Op.get("relax.nn.conv3d")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_sinfo: relax.StructInfo):
    ret = bb.normalize(call)
    tvm.ir.assert_structural_equal(ret.struct_info, expected_sinfo)


def test_conv1d_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 28, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=3))
    x3 = relax.Var("x", R.Tensor("float32"))
    x4 = relax.Var("x", R.Tensor())
    x5 = relax.Var("x", R.Tensor((2, 4, 28, 16), "float32"))
    x6 = relax.Var("x", R.Tensor((2, 3, 28), "float32", vdev0))
    w0 = relax.Var("w", R.Tensor((4, 3, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((3, 4, 3), "float32"))
    w2 = relax.Var("w", R.Tensor("float32", ndim=3))
    w3 = relax.Var("w", R.Tensor("float32"))
    w4 = relax.Var("w", R.Tensor((48, 4, 3, 16), "float32"))
    w5 = relax.Var("w", R.Tensor((4, 3, 3), "float32", vdev0))

    _check_inference(bb, relax.op.nn.conv1d(x0, w0), relax.TensorStructInfo((2, 4, 26), "float32"))
    _check_inference(
        bb, relax.op.nn.conv1d(x6, w5), relax.TensorStructInfo((2, 4, 26), "float32", vdev0)
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w0, out_dtype="float16"),
        relax.TensorStructInfo((2, 4, 26), "float16"),
    )
    _check_inference(
        bb, relax.op.nn.conv1d(x0, w0, padding=1), relax.TensorStructInfo((2, 4, 28), "float32")
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w0, padding=[1, 3]),
        relax.TensorStructInfo((2, 4, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w0, strides=2),
        relax.TensorStructInfo((2, 4, 13), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w0, strides=(2,)),
        relax.TensorStructInfo((2, 4, 13), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w0, dilation=2),
        relax.TensorStructInfo((2, 4, 24), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w0, dilation=(2,)),
        relax.TensorStructInfo((2, 4, 24), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x1, w0, data_layout="NWC"),
        relax.TensorStructInfo((2, 26, 4), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w0, out_layout="NWC"),
        relax.TensorStructInfo((2, 26, 4), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w1, kernel_layout="IOW"),
        relax.TensorStructInfo((2, 4, 26), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(
            x5, w4, data_layout="NCW16c", kernel_layout="OIW16i", out_layout="NWC16c"
        ),
        relax.TensorStructInfo((2, 26, 3, 16), "float32"),
    )
    _check_inference(
        bb, relax.op.nn.conv1d(x2, w0), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.nn.conv1d(x3, w0), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.nn.conv1d(x0, w2), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.nn.conv1d(x0, w3), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(bb, relax.op.nn.conv1d(x4, w0), relax.TensorStructInfo(dtype="", ndim=3))


def test_conv1d_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    c = tir.Var("c", "int64")
    c16 = tir.Var("c16", "int64")
    iw = tir.Var("iw", "int64")
    ki = tir.Var("ki", "int64")
    ko = tir.Var("ko", "int64")
    kw = tir.Var("kw", "int64")
    x0 = relax.Var("x", R.Tensor((n, c, iw), "float32"))
    x1 = relax.Var("x", R.Tensor((n, c, iw, c16), "float32"))
    w0 = relax.Var("w", R.Tensor((ko, ki, kw), "float32"))
    w1 = relax.Var("w", R.Tensor((ko, c, kw), "float32"))
    w2 = relax.Var("w", R.Tensor((ko, c, kw, c16), "float32"))

    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w0),
        relax.TensorStructInfo((n, ko, iw + 1 - kw), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w1),
        relax.TensorStructInfo((n, ko, iw + 1 - kw), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x1, w2, data_layout="NCW16c", kernel_layout="OIW16i", out_layout="NCW"),
        relax.TensorStructInfo((n, ko, iw + 1 - kw), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w0, strides=2, padding=1, dilation=2),
        relax.TensorStructInfo(
            (n, ko, tvm.tir.floordiv(iw + 3, 2) + 1 - kw),
            "float32",
        ),
    )


def test_conv1d_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=3))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=4))
    s2 = relax.Var("s", relax.ShapeStructInfo(ndim=3))
    s3 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s3, "float32"))
    w = relax.Var("w", relax.TensorStructInfo(s2, "float32"))

    _check_inference(bb, relax.op.nn.conv1d(x0, w), relax.TensorStructInfo(dtype="float32", ndim=3))
    _check_inference(
        bb,
        relax.op.nn.conv1d(x1, w, data_layout="NCW16c"),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w, out_layout="NCW16c"),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x2, w),
        relax.TensorStructInfo(dtype="float32", ndim=3),
    )


def test_conv1d_infer_struct_info_groups():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 128, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 8, 28, 16), "float32"))
    w0 = relax.Var("w", R.Tensor((48, 16, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((48, 2, 3, 8), "float32"))

    _check_inference(
        bb, relax.op.nn.conv1d(x0, w0, groups=8), relax.TensorStructInfo((2, 48, 26), "float32")
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w1, kernel_layout="OIW8i", groups=8),
        relax.TensorStructInfo((2, 48, 26), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x1, w0, data_layout="NCW16c", groups=8),
        relax.TensorStructInfo((2, 3, 26, 16), "float32"),
    )


def test_conv1d_infer_struct_info_symbolic_groups():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    ic = tir.Var("c", "int64")
    oc = tir.Var("oc", "int64")
    x = relax.Var("x", R.Tensor((n, ic * 4, 28), "float32"))
    w0 = relax.Var("w", R.Tensor((oc * 4, ic, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((oc, ic, 3), "float32"))

    _check_inference(
        bb,
        relax.op.nn.conv1d(x, w0, groups=4),
        relax.TensorStructInfo((n, oc * 4, 26), "float32"),
    )
    _check_inference(
        bb, relax.op.nn.conv1d(x, w1, groups=4), relax.TensorStructInfo((n, oc, 26), "float32")
    )


def test_conv1d_infer_struct_info_input_channel_group_incompatible():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    ic = tir.Var("c", "int64")
    oc = tir.Var("oc", "int64")
    x0 = relax.Var("x", R.Tensor((2, 128, 28), "float32"))
    w0 = relax.Var("w", R.Tensor((48, 20, 3), "float32"))
    x1 = relax.Var("x", R.Tensor((n, ic * 6, 28), "float32"))
    w1 = relax.Var("w", R.Tensor((oc, ic - 1, 3), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d(x0, w0, groups=6))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d(x1, w1, groups=6))


def test_conv1d_infer_struct_info_output_channel_group_incompatible():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    ic = tir.Var("c", "int64")
    oc = tir.Var("oc", "int64")
    x0 = relax.Var("x", R.Tensor((2, 120, 28), "float32"))
    w0 = relax.Var("w", R.Tensor((128, 20, 3), "float32"))
    x1 = relax.Var("x", R.Tensor((n, ic * 6, 28), "float32"))
    w1 = relax.Var("w", R.Tensor((oc * 6 + 4, ic * 6, 3), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d(x0, w0, groups=6))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d(x1, w1, groups=6))


def test_conv1d_non_positive_group():
    x = relax.Var("x", R.Tensor((2, 128, 28), "float32"))
    w = relax.Var("w", R.Tensor((48, 16, 3), "float32"))

    with pytest.raises(TVMError):
        relax.op.nn.conv1d(x, w, groups=0)
    with pytest.raises(TVMError):
        relax.op.nn.conv1d(x, w, groups=-2)


def test_conv1d_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28), "float16"))
    w0 = relax.Var("w", R.Tensor((4, 3, 3), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 28), "float64"))
    w1 = relax.Var("w", R.Tensor((4, 3, 3), "float64"))
    x2 = relax.Var("x", R.Tensor((2, 3, 28), "int8"))
    w2 = relax.Var("w", R.Tensor((4, 3, 3), "int8"))
    x3 = relax.Var("x", R.Tensor((2, 3, 28), "int32"))
    w3 = relax.Var("w", R.Tensor((4, 3, 3), "int32"))

    _check_inference(bb, relax.op.nn.conv1d(x0, w0), relax.TensorStructInfo((2, 4, 26), "float16"))
    _check_inference(bb, relax.op.nn.conv1d(x1, w1), relax.TensorStructInfo((2, 4, 26), "float64"))
    _check_inference(bb, relax.op.nn.conv1d(x2, w2), relax.TensorStructInfo((2, 4, 26), "int8"))
    _check_inference(bb, relax.op.nn.conv1d(x3, w3), relax.TensorStructInfo((2, 4, 26), "int32"))


def test_conv1d_infer_struct_info_mixed_precision():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28), "float16"))
    w0 = relax.Var("w", R.Tensor((4, 3, 3), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 28), "int8"))
    w1 = relax.Var("w", R.Tensor((4, 3, 3), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3, 28)))
    w2 = relax.Var("w", R.Tensor((4, 3, 3)))

    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w0, out_dtype="float32"),
        relax.TensorStructInfo((2, 4, 26), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x1, w1, out_dtype="int32"),
        relax.TensorStructInfo((2, 4, 26), "int32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x2, w2, out_dtype="float32"),
        relax.TensorStructInfo((2, 4, 26), "float32"),
    )


def test_conv1d_unequal_input_channel():
    bb = relax.BlockBuilder()
    ic = tir.Var("ic", "int64")
    x0 = relax.Var("x", R.Tensor([2, 3, 28], "float32"))
    w0 = relax.Var("w", R.Tensor([3, 4, 3], "float32"))
    x1 = relax.Var("x", R.Tensor([2, ic, 28], "float32"))
    w1 = relax.Var("w", R.Tensor([4, ic + 2, 3], "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d(x0, w0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d(x1, w1))


def test_conv1d_stride_padding_dilation_int64():
    x = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    w = relax.Var("w", R.Tensor((4, 3, 3), "float32"))
    conv1d = relax.op.nn.conv1d(x, w, strides=(1,), padding=(1, 1), dilation=(1,))

    assert conv1d.attrs.strides[0].dtype == "int64"
    assert conv1d.attrs.padding[0].dtype == "int64"
    assert conv1d.attrs.padding[1].dtype == "int64"
    assert conv1d.attrs.dilation[0].dtype == "int64"


def test_conv1d_wrong_strides_padding_dilation_length():
    x = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    w = relax.Var("w", R.Tensor((4, 3, 3), "float32"))
    with pytest.raises(TVMError):
        relax.op.nn.conv1d(x, w, strides=(1, 2))
    with pytest.raises(TVMError):
        relax.op.nn.conv1d(x, w, padding=(1, 2, 3))
    with pytest.raises(TVMError):
        relax.op.nn.conv1d(x, w, dilation=(1, 2))


def test_conv1d_infer_struct_info_wrong_layout_string():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    w = relax.Var("w", R.Tensor((4, 3, 3), "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d(x, w, data_layout="OIW"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d(x, w, kernel_layout="NWC"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d(x, w, out_layout="OWI"))


def test_conv1d_dtype_mismatch():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    w = relax.Var("w", R.Tensor((4, 3, 3), "int8"))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.nn.conv1d(x, w))


def test_conv1d_wrong_input_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 3, 28, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=2))
    w0 = relax.Var("w", R.Tensor((4, 3, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((4, 3, 6, 3), "float32"))
    w2 = relax.Var("w", R.Tensor("float32", ndim=5))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d(x0, w1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d(x0, w1, data_layout="NCW16c"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d(x0, w2))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d(x1, w0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d(x2, w0))


def test_conv1d_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    x1 = relax.Var("x", relax.ShapeStructInfo((2, 3, 28)))
    w0 = relax.Var("w", R.Tensor((4, 3, 3), "float32"))
    w1 = relax.Var("w", relax.FuncStructInfo([], R.Tensor((4, 3, 3), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d(x0, w1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d(x1, w0))


def test_conv1d_transpose_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 28, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=3))
    x3 = relax.Var("x", R.Tensor("float32"))
    x4 = relax.Var("x", R.Tensor())
    x5 = relax.Var("x", R.Tensor((2, 4, 28, 16), "float32"))
    x6 = relax.Var("x", R.Tensor((2, 3, 28), "float32", vdev0))
    w0 = relax.Var("w", R.Tensor((3, 4, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((4, 3, 3), "float32"))
    w2 = relax.Var("w", R.Tensor("float32", ndim=3))
    w3 = relax.Var("w", R.Tensor("float32"))
    w4 = relax.Var("w", R.Tensor((4, 48, 3, 16), "float32"))
    w5 = relax.Var("w", R.Tensor((3, 4, 3), "float32", vdev0))

    _check_inference(
        bb, relax.op.nn.conv1d_transpose(x0, w0), relax.TensorStructInfo((2, 4, 30), "float32")
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x6, w5),
        relax.TensorStructInfo((2, 4, 30), "float32", vdev0),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w0, out_dtype="float16"),
        relax.TensorStructInfo((2, 4, 30), "float16"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w0, padding=1),
        relax.TensorStructInfo((2, 4, 28), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w0, padding=[1, 3]),
        relax.TensorStructInfo((2, 4, 26), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w0, strides=3, output_padding=1),
        relax.TensorStructInfo((2, 4, 85), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w0, strides=2),
        relax.TensorStructInfo((2, 4, 57), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w0, dilation=2),
        relax.TensorStructInfo((2, 4, 32), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w0, dilation=(2,)),
        relax.TensorStructInfo((2, 4, 32), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x1, w0, data_layout="NWC"),
        relax.TensorStructInfo((2, 30, 4), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w0, out_layout="NWC"),
        relax.TensorStructInfo((2, 30, 4), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w1, kernel_layout="OIW"),
        relax.TensorStructInfo((2, 4, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(
            x5, w4, data_layout="NCW16c", kernel_layout="IOW16i", out_layout="NWC16c"
        ),
        relax.TensorStructInfo((2, 30, 3, 16), "float32"),
    )
    _check_inference(
        bb, relax.op.nn.conv1d_transpose(x2, w0), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.nn.conv1d_transpose(x3, w0), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.nn.conv1d_transpose(x0, w2), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.nn.conv1d_transpose(x0, w3), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.nn.conv1d_transpose(x4, w0), relax.TensorStructInfo(dtype="", ndim=3)
    )


def test_conv1d_transpose_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    c = tir.Var("c", "int64")
    c16 = tir.Var("c16", "int64")
    iw = tir.Var("iw", "int64")
    ki = tir.Var("ki", "int64")
    ko = tir.Var("ko", "int64")
    kw = tir.Var("kw", "int64")
    x0 = relax.Var("x", R.Tensor((n, c, iw), "float32"))
    x1 = relax.Var("x", R.Tensor((n, c, iw, c16), "float32"))
    w0 = relax.Var("w", R.Tensor((ki, ko, kw), "float32"))
    w1 = relax.Var("w", R.Tensor((c, ko, kw), "float32"))
    w2 = relax.Var("w", R.Tensor((c, ko, kw, c16), "float32"))

    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w0),
        relax.TensorStructInfo((n, ko, iw + kw - 1), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w1),
        relax.TensorStructInfo((n, ko, iw + kw - 1), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(
            x1, w2, data_layout="NCW16c", kernel_layout="IOW16i", out_layout="NCW"
        ),
        relax.TensorStructInfo((n, ko, iw + kw - 1), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w0, strides=2, padding=1, dilation=2, output_padding=1),
        relax.TensorStructInfo(
            (n, ko, iw * 2 + kw * 2 - 4),
            "float32",
        ),
    )


def test_conv1d_transpose_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=3))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=4))
    s2 = relax.Var("s", relax.ShapeStructInfo(ndim=3))
    s3 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s3, "float32"))
    w = relax.Var("w", relax.TensorStructInfo(s2, "float32"))

    _check_inference(bb, relax.op.nn.conv1d(x0, w), relax.TensorStructInfo(dtype="float32", ndim=3))
    _check_inference(
        bb,
        relax.op.nn.conv1d(x1, w, data_layout="NCW16c"),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w, out_layout="NCW16c"),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x2, w),
        relax.TensorStructInfo(dtype="float32", ndim=3),
    )


def test_conv1d_transpose_infer_struct_info_groups():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 128, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 8, 28, 16), "float32"))
    w0 = relax.Var("w", R.Tensor((128, 6, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((16, 6, 3, 8), "float32"))

    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w0, groups=8),
        relax.TensorStructInfo((2, 48, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w1, kernel_layout="IOW8i", groups=8),
        relax.TensorStructInfo((2, 48, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x1, w0, data_layout="NCW16c", groups=8),
        relax.TensorStructInfo((2, 3, 30, 16), "float32"),
    )


def test_conv1d_transpose_infer_struct_info_symbolic_groups():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    ic = tir.Var("c", "int64")
    oc = tir.Var("oc", "int64")
    x = relax.Var("x", R.Tensor((n, ic * 4, 28), "float32"))
    w0 = relax.Var("w", R.Tensor((ic, oc, 3), "float32"))

    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x, w0, groups=4),
        relax.TensorStructInfo((n, oc * 4, 30), "float32"),
    )


def test_conv1d_transpose_infer_struct_info_input_channel_group_incompatible():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    ic = tir.Var("c", "int64")
    oc = tir.Var("oc", "int64")
    x0 = relax.Var("x", R.Tensor((2, 128, 28), "float32"))
    w0 = relax.Var("w", R.Tensor((128, 20, 3), "float32"))
    x1 = relax.Var("x", R.Tensor((n, ic, 28), "float32"))
    w1 = relax.Var("w", R.Tensor((ic - 1, oc, 3), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d_transpose(x0, w0, groups=6))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d_transpose(x1, w1, groups=6))


def test_conv1d_transpose_non_positive_group():
    x = relax.Var("x", R.Tensor((2, 128, 28), "float32"))
    w = relax.Var("w", R.Tensor((128, 16, 3), "float32"))

    with pytest.raises(TVMError):
        relax.op.nn.conv1d_transpose(x, w, groups=0)
    with pytest.raises(TVMError):
        relax.op.nn.conv1d_transpose(x, w, groups=-2)


def test_conv1d_transpose_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28), "float16"))
    w0 = relax.Var("w", R.Tensor((3, 4, 3), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 28), "float64"))
    w1 = relax.Var("w", R.Tensor((3, 4, 3), "float64"))
    x2 = relax.Var("x", R.Tensor((2, 3, 28), "int8"))
    w2 = relax.Var("w", R.Tensor((3, 4, 3), "int8"))
    x3 = relax.Var("x", R.Tensor((2, 3, 28), "int32"))
    w3 = relax.Var("w", R.Tensor((3, 4, 3), "int32"))

    _check_inference(
        bb, relax.op.nn.conv1d_transpose(x0, w0), relax.TensorStructInfo((2, 4, 30), "float16")
    )
    _check_inference(
        bb, relax.op.nn.conv1d_transpose(x1, w1), relax.TensorStructInfo((2, 4, 30), "float64")
    )
    _check_inference(
        bb, relax.op.nn.conv1d_transpose(x2, w2), relax.TensorStructInfo((2, 4, 30), "int8")
    )
    _check_inference(
        bb, relax.op.nn.conv1d_transpose(x3, w3), relax.TensorStructInfo((2, 4, 30), "int32")
    )


def test_conv1d_transpose_unequal_input_channel():
    bb = relax.BlockBuilder()
    ic = tir.Var("ic", "int64")
    x0 = relax.Var("x", R.Tensor([2, 3, 28], "float32"))
    w0 = relax.Var("w", R.Tensor([4, 3, 3], "float32"))
    x1 = relax.Var("x", R.Tensor([2, ic, 28], "float32"))
    w1 = relax.Var("w", R.Tensor([ic + 2, 4, 3], "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d_transpose(x0, w0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d_transpose(x1, w1))


def test_conv1d_transpose_wrong_output_padding():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor([2, 3, 28], "float32"))
    w0 = relax.Var("w", R.Tensor([3, 4, 3], "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d_transpose(x0, w0, strides=2, output_padding=2))


def test_conv1d_transpose_stride_padding_dilation_int64():
    x = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    w = relax.Var("w", R.Tensor((3, 4, 3), "float32"))
    conv1d = relax.op.nn.conv1d_transpose(x, w, strides=1, padding=1, dilation=1)

    assert conv1d.attrs.strides[0].dtype == "int64"
    assert conv1d.attrs.padding[0].dtype == "int64"
    assert conv1d.attrs.dilation[0].dtype == "int64"


def test_conv1d_transpose_wrong_strides_padding_dilation_length():
    x = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    w = relax.Var("w", R.Tensor((3, 4, 3), "float32"))
    with pytest.raises(TVMError):
        relax.op.nn.conv1d_transpose(x, w, strides=(1, 2))
    with pytest.raises(TVMError):
        relax.op.nn.conv1d_transpose(x, w, padding=(1, 2, 3))
    with pytest.raises(TVMError):
        relax.op.nn.conv1d_transpose(x, w, dilation=(1, 2))


def test_conv1d_transpose_infer_struct_info_wrong_layout_string():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    w = relax.Var("w", R.Tensor((3, 4, 3), "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d_transpose(x, w, data_layout="IOW"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d_transpose(x, w, kernel_layout="NWC"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d_transpose(x, w, out_layout="OWI"))


def test_conv1d_transpose_dtype_mismatch():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    w = relax.Var("w", R.Tensor((3, 4, 3), "int8"))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.nn.conv1d_transpose(x, w))


def test_conv1d_transpose_wrong_input_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 3, 28, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=2))
    w0 = relax.Var("w", R.Tensor((3, 4, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((3, 4, 6, 3), "float32"))
    w2 = relax.Var("w", R.Tensor("float32", ndim=5))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d_transpose(x0, w1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d_transpose(x0, w1, data_layout="NCW16c"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d_transpose(x0, w2))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d_transpose(x1, w0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d_transpose(x2, w0))


def test_conv1d_transpose_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    x1 = relax.Var("x", relax.ShapeStructInfo((2, 3, 28)))
    w0 = relax.Var("w", R.Tensor((3, 4, 3), "float32"))
    w1 = relax.Var("w", relax.FuncStructInfo([], R.Tensor((3, 4, 3), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d_transpose(x0, w1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv1d_transpose(x1, w0))


def test_conv2d_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 28, 28, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=4))
    x3 = relax.Var("x", R.Tensor("float32"))
    x4 = relax.Var("x", R.Tensor())
    x5 = relax.Var("x", R.Tensor((2, 4, 28, 28, 16), "float32"))
    x6 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32", vdev0))
    w0 = relax.Var("w", R.Tensor((4, 3, 3, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((3, 4, 3, 3), "float32"))
    w2 = relax.Var("w", R.Tensor("float32", ndim=4))
    w3 = relax.Var("w", R.Tensor("float32"))
    w4 = relax.Var("w", R.Tensor((48, 4, 3, 3, 16), "float32"))
    w5 = relax.Var("w", R.Tensor((4, 3, 3, 3), "float32", vdev0))

    _check_inference(
        bb, relax.op.nn.conv2d(x0, w0), relax.TensorStructInfo((2, 4, 26, 26), "float32")
    )
    _check_inference(
        bb, relax.op.nn.conv2d(x6, w5), relax.TensorStructInfo((2, 4, 26, 26), "float32", vdev0)
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w0, out_dtype="float16"),
        relax.TensorStructInfo((2, 4, 26, 26), "float16"),
    )
    _check_inference(
        bb, relax.op.nn.conv2d(x0, w0, padding=1), relax.TensorStructInfo((2, 4, 28, 28), "float32")
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w0, padding=[1, 2]),
        relax.TensorStructInfo((2, 4, 28, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w0, padding=[1, 2, 3, 4]),
        relax.TensorStructInfo((2, 4, 30, 32), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w0, strides=2),
        relax.TensorStructInfo((2, 4, 13, 13), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w0, strides=(2, 3)),
        relax.TensorStructInfo((2, 4, 13, 9), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w0, dilation=2),
        relax.TensorStructInfo((2, 4, 24, 24), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w0, dilation=(2, 1)),
        relax.TensorStructInfo((2, 4, 24, 26), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x1, w0, data_layout="NHWC"),
        relax.TensorStructInfo((2, 26, 26, 4), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w0, out_layout="NHWC"),
        relax.TensorStructInfo((2, 26, 26, 4), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w1, kernel_layout="IOHW"),
        relax.TensorStructInfo((2, 4, 26, 26), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(
            x5, w4, data_layout="NCHW16c", kernel_layout="OIHW16i", out_layout="NHWC16c"
        ),
        relax.TensorStructInfo((2, 26, 26, 3, 16), "float32"),
    )
    _check_inference(
        bb, relax.op.nn.conv2d(x2, w0), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.nn.conv2d(x3, w0), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.nn.conv2d(x0, w2), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.nn.conv2d(x0, w3), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(bb, relax.op.nn.conv2d(x4, w0), relax.TensorStructInfo(dtype="", ndim=4))


def test_conv2d_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    c = tir.Var("c", "int64")
    c16 = tir.Var("c16", "int64")
    ih = tir.Var("ih", "int64")
    iw = tir.Var("iw", "int64")
    ki = tir.Var("ki", "int64")
    ko = tir.Var("ko", "int64")
    kh = tir.Var("kh", "int64")
    kw = tir.Var("kw", "int64")
    x0 = relax.Var("x", R.Tensor((n, c, ih, iw), "float32"))
    x1 = relax.Var("x", R.Tensor((n, c, ih, iw, c16), "float32"))
    w0 = relax.Var("w", R.Tensor((ko, ki, kh, kw), "float32"))
    w1 = relax.Var("w", R.Tensor((ko, c, kh, kw), "float32"))
    w2 = relax.Var("w", R.Tensor((ko, c, kh, kw, c16), "float32"))

    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w0),
        relax.TensorStructInfo((n, ko, ih + 1 - kh, iw + 1 - kw), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w1),
        relax.TensorStructInfo((n, ko, ih + 1 - kh, iw + 1 - kw), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(
            x1, w2, data_layout="NCHW16c", kernel_layout="OIHW16i", out_layout="NCHW"
        ),
        relax.TensorStructInfo((n, ko, ih + 1 - kh, iw + 1 - kw), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w0, strides=(2, 2), padding=(1, 1), dilation=(2, 2)),
        relax.TensorStructInfo(
            (n, ko, tvm.tir.floordiv(ih + 3, 2) + 1 - kh, tvm.tir.floordiv(iw + 3, 2) + 1 - kw),
            "float32",
        ),
    )


def test_conv2d_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=4))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=5))
    s2 = relax.Var("s", relax.ShapeStructInfo(ndim=4))
    s3 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s3, "float32"))
    w = relax.Var("w", relax.TensorStructInfo(s2, "float32"))

    _check_inference(bb, relax.op.nn.conv2d(x0, w), relax.TensorStructInfo(dtype="float32", ndim=4))
    _check_inference(
        bb,
        relax.op.nn.conv2d(x1, w, data_layout="NCHW16c"),
        relax.TensorStructInfo(dtype="float32", ndim=5),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w, out_layout="NCHW16c"),
        relax.TensorStructInfo(dtype="float32", ndim=5),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x2, w),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )


def test_conv2d_infer_struct_info_groups():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 128, 28, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 8, 28, 28, 16), "float32"))
    w0 = relax.Var("w", R.Tensor((48, 16, 3, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((48, 2, 3, 3, 8), "float32"))

    _check_inference(
        bb, relax.op.nn.conv2d(x0, w0, groups=8), relax.TensorStructInfo((2, 48, 26, 26), "float32")
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w1, kernel_layout="OIHW8i", groups=8),
        relax.TensorStructInfo((2, 48, 26, 26), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x1, w0, data_layout="NCHW16c", groups=8),
        relax.TensorStructInfo((2, 3, 26, 26, 16), "float32"),
    )


def test_conv2d_infer_struct_info_symbolic_groups():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    ic = tir.Var("c", "int64")
    oc = tir.Var("oc", "int64")
    x = relax.Var("x", R.Tensor((n, ic * 4, 28, 28), "float32"))
    w0 = relax.Var("w", R.Tensor((oc * 4, ic, 3, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((oc, ic, 3, 3), "float32"))

    _check_inference(
        bb,
        relax.op.nn.conv2d(x, w0, groups=4),
        relax.TensorStructInfo((n, oc * 4, 26, 26), "float32"),
    )
    _check_inference(
        bb, relax.op.nn.conv2d(x, w1, groups=4), relax.TensorStructInfo((n, oc, 26, 26), "float32")
    )


def test_conv2d_infer_struct_info_input_channel_group_incompatible():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    ic = tir.Var("c", "int64")
    oc = tir.Var("oc", "int64")
    x0 = relax.Var("x", R.Tensor((2, 128, 28, 28), "float32"))
    w0 = relax.Var("w", R.Tensor((48, 20, 3, 3), "float32"))
    x1 = relax.Var("x", R.Tensor((n, ic * 6, 28, 28), "float32"))
    w1 = relax.Var("w", R.Tensor((oc, ic - 1, 3, 3), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d(x0, w0, groups=6))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d(x1, w1, groups=6))


def test_conv2d_infer_struct_info_output_channel_group_incompatible():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    ic = tir.Var("c", "int64")
    oc = tir.Var("oc", "int64")
    x0 = relax.Var("x", R.Tensor((2, 120, 28, 28), "float32"))
    w0 = relax.Var("w", R.Tensor((128, 20, 3, 3), "float32"))
    x1 = relax.Var("x", R.Tensor((n, ic * 6, 28, 28), "float32"))
    w1 = relax.Var("w", R.Tensor((oc * 6 + 4, ic * 6, 3, 3), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d(x0, w0, groups=6))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d(x1, w1, groups=6))


def test_conv2d_non_positive_group():
    x = relax.Var("x", R.Tensor((2, 128, 28, 28), "float32"))
    w = relax.Var("w", R.Tensor((48, 16, 3, 3), "float32"))

    with pytest.raises(TVMError):
        relax.op.nn.conv2d(x, w, groups=0)
    with pytest.raises(TVMError):
        relax.op.nn.conv2d(x, w, groups=-2)


def test_conv2d_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float16"))
    w0 = relax.Var("w", R.Tensor((4, 3, 3, 3), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float64"))
    w1 = relax.Var("w", R.Tensor((4, 3, 3, 3), "float64"))
    x2 = relax.Var("x", R.Tensor((2, 3, 28, 28), "int8"))
    w2 = relax.Var("w", R.Tensor((4, 3, 3, 3), "int8"))
    x3 = relax.Var("x", R.Tensor((2, 3, 28, 28), "int32"))
    w3 = relax.Var("w", R.Tensor((4, 3, 3, 3), "int32"))

    _check_inference(
        bb, relax.op.nn.conv2d(x0, w0), relax.TensorStructInfo((2, 4, 26, 26), "float16")
    )
    _check_inference(
        bb, relax.op.nn.conv2d(x1, w1), relax.TensorStructInfo((2, 4, 26, 26), "float64")
    )
    _check_inference(bb, relax.op.nn.conv2d(x2, w2), relax.TensorStructInfo((2, 4, 26, 26), "int8"))
    _check_inference(
        bb, relax.op.nn.conv2d(x3, w3), relax.TensorStructInfo((2, 4, 26, 26), "int32")
    )


def test_conv2d_infer_struct_info_mixed_precision():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float16"))
    w0 = relax.Var("w", R.Tensor((4, 3, 3, 3), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 28, 28), "int8"))
    w1 = relax.Var("w", R.Tensor((4, 3, 3, 3), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3, 28, 28)))
    w2 = relax.Var("w", R.Tensor((4, 3, 3, 3)))

    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w0, out_dtype="float32"),
        relax.TensorStructInfo((2, 4, 26, 26), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x1, w1, out_dtype="int32"),
        relax.TensorStructInfo((2, 4, 26, 26), "int32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x2, w2, out_dtype="float32"),
        relax.TensorStructInfo((2, 4, 26, 26), "float32"),
    )


def test_conv2d_unequal_input_channel():
    bb = relax.BlockBuilder()
    ic = tir.Var("ic", "int64")
    x0 = relax.Var("x", R.Tensor([2, 3, 28, 28], "float32"))
    w0 = relax.Var("w", R.Tensor([3, 4, 3, 3], "float32"))
    x1 = relax.Var("x", R.Tensor([2, ic, 28, 28], "float32"))
    w1 = relax.Var("w", R.Tensor([4, ic + 2, 3, 3], "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d(x0, w0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d(x1, w1))


def test_conv2d_stride_padding_dilation_int64():
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    w = relax.Var("w", R.Tensor((4, 3, 3, 3), "float32"))
    conv2d = relax.op.nn.conv2d(x, w, strides=(1, 1), padding=(1, 1), dilation=(1, 1))

    assert conv2d.attrs.strides[0].dtype == "int64"
    assert conv2d.attrs.strides[1].dtype == "int64"
    assert conv2d.attrs.padding[0].dtype == "int64"
    assert conv2d.attrs.padding[1].dtype == "int64"
    assert conv2d.attrs.padding[2].dtype == "int64"
    assert conv2d.attrs.padding[3].dtype == "int64"
    assert conv2d.attrs.dilation[0].dtype == "int64"
    assert conv2d.attrs.dilation[1].dtype == "int64"


def test_conv2d_wrong_strides_padding_dilation_length():
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    w = relax.Var("w", R.Tensor((4, 3, 3, 3), "float32"))
    with pytest.raises(TVMError):
        relax.op.nn.conv2d(x, w, strides=(1, 2, 3))
    with pytest.raises(TVMError):
        relax.op.nn.conv2d(x, w, padding=(1, 2, 3))
    with pytest.raises(TVMError):
        relax.op.nn.conv2d(x, w, dilation=(1, 2, 3))


def test_conv2d_infer_struct_info_wrong_layout_string():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    w = relax.Var("w", R.Tensor((4, 3, 3, 3), "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d(x, w, data_layout="OIHW"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d(x, w, kernel_layout="NHWC"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d(x, w, out_layout="OHWI"))


def test_conv2d_dtype_mismatch():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    w = relax.Var("w", R.Tensor((4, 3, 3, 3), "int8"))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.nn.conv2d(x, w))


def test_conv2d_wrong_input_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 3, 28, 28, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=3))
    w0 = relax.Var("w", R.Tensor((4, 3, 3, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((4, 3, 6, 3, 3), "float32"))
    w2 = relax.Var("w", R.Tensor("float32", ndim=6))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d(x0, w1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d(x0, w1, data_layout="NCHW16c"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d(x0, w2))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d(x1, w0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d(x2, w0))


def test_conv2d_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    x1 = relax.Var("x", relax.ShapeStructInfo((2, 3, 28, 28)))
    w0 = relax.Var("w", R.Tensor((4, 3, 3, 3), "float32"))
    w1 = relax.Var("w", relax.FuncStructInfo([], R.Tensor((4, 3, 3, 3), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d(x0, w1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d(x1, w0))


def test_conv2d_transpose_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 28, 28, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=4))
    x3 = relax.Var("x", R.Tensor("float32"))
    x4 = relax.Var("x", R.Tensor())
    x5 = relax.Var("x", R.Tensor((2, 4, 28, 28, 16), "float32"))
    x6 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32", vdev0))
    w0 = relax.Var("w", R.Tensor((3, 4, 3, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((4, 3, 3, 3), "float32"))
    w2 = relax.Var("w", R.Tensor("float32", ndim=4))
    w3 = relax.Var("w", R.Tensor("float32"))
    w4 = relax.Var("w", R.Tensor((4, 48, 3, 3, 16), "float32"))
    w5 = relax.Var("w", R.Tensor((3, 4, 3, 3), "float32", vdev0))

    _check_inference(
        bb, relax.op.nn.conv2d_transpose(x0, w0), relax.TensorStructInfo((2, 4, 30, 30), "float32")
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x6, w5),
        relax.TensorStructInfo((2, 4, 30, 30), "float32", vdev0),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0, out_dtype="float16"),
        relax.TensorStructInfo((2, 4, 30, 30), "float16"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0, padding=1),
        relax.TensorStructInfo((2, 4, 28, 28), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0, padding=[1, 2]),
        relax.TensorStructInfo((2, 4, 28, 26), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0, padding=[1, 2, 3, 4]),
        relax.TensorStructInfo((2, 4, 26, 24), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0, strides=3, output_padding=1),
        relax.TensorStructInfo((2, 4, 85, 85), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0, strides=3, output_padding=[2, 1]),
        relax.TensorStructInfo((2, 4, 86, 85), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0, strides=2),
        relax.TensorStructInfo((2, 4, 57, 57), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0, strides=(2, 3)),
        relax.TensorStructInfo((2, 4, 57, 84), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0, dilation=2),
        relax.TensorStructInfo((2, 4, 32, 32), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0, dilation=(2, 1)),
        relax.TensorStructInfo((2, 4, 32, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x1, w0, data_layout="NHWC"),
        relax.TensorStructInfo((2, 30, 30, 4), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0, out_layout="NHWC"),
        relax.TensorStructInfo((2, 30, 30, 4), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w1, kernel_layout="OIHW"),
        relax.TensorStructInfo((2, 4, 30, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(
            x5, w4, data_layout="NCHW16c", kernel_layout="IOHW16i", out_layout="NHWC16c"
        ),
        relax.TensorStructInfo((2, 30, 30, 3, 16), "float32"),
    )
    _check_inference(
        bb, relax.op.nn.conv2d_transpose(x2, w0), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.nn.conv2d_transpose(x3, w0), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.nn.conv2d_transpose(x0, w2), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.nn.conv2d_transpose(x0, w3), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.nn.conv2d_transpose(x4, w0), relax.TensorStructInfo(dtype="", ndim=4)
    )


def test_conv2d_transpose_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    c = tir.Var("c", "int64")
    c16 = tir.Var("c16", "int64")
    ih = tir.Var("ih", "int64")
    iw = tir.Var("iw", "int64")
    ki = tir.Var("ki", "int64")
    ko = tir.Var("ko", "int64")
    kh = tir.Var("kh", "int64")
    kw = tir.Var("kw", "int64")
    x0 = relax.Var("x", R.Tensor((n, c, ih, iw), "float32"))
    x1 = relax.Var("x", R.Tensor((n, c, ih, iw, c16), "float32"))
    w0 = relax.Var("w", R.Tensor((ki, ko, kh, kw), "float32"))
    w1 = relax.Var("w", R.Tensor((c, ko, kh, kw), "float32"))
    w2 = relax.Var("w", R.Tensor((c, ko, kh, kw, c16), "float32"))

    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0),
        relax.TensorStructInfo((n, ko, ih + kh - 1, iw + kw - 1), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w1),
        relax.TensorStructInfo((n, ko, ih + kh - 1, iw + kw - 1), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(
            x1, w2, data_layout="NCHW16c", kernel_layout="IOHW16i", out_layout="NCHW"
        ),
        relax.TensorStructInfo((n, ko, ih + kh - 1, iw + kw - 1), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(
            x0, w0, strides=(2, 2), padding=(1, 1), output_padding=(1, 0), dilation=(2, 2)
        ),
        relax.TensorStructInfo(
            (n, ko, ih * 2 + kh * 2 - 4, iw * 2 + kw * 2 - 5),
            "float32",
        ),
    )


def test_conv2d_transpose_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=4))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=5))
    s2 = relax.Var("s", relax.ShapeStructInfo(ndim=4))
    s3 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s3, "float32"))
    w = relax.Var("w", relax.TensorStructInfo(s2, "float32"))

    _check_inference(
        bb, relax.op.nn.conv2d_transpose(x0, w), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x1, w, data_layout="NCHW16c"),
        relax.TensorStructInfo(dtype="float32", ndim=5),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w, out_layout="NCHW16c"),
        relax.TensorStructInfo(dtype="float32", ndim=5),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x2, w),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )


def test_conv2d_transpose_infer_struct_info_groups():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 128, 28, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 8, 28, 28, 16), "float32"))
    w0 = relax.Var("w", R.Tensor((128, 6, 3, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((16, 6, 3, 3, 8), "float32"))

    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0, groups=8),
        relax.TensorStructInfo((2, 48, 30, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w1, kernel_layout="IOHW8i", groups=8),
        relax.TensorStructInfo((2, 48, 30, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x1, w0, data_layout="NCHW16c", groups=8),
        relax.TensorStructInfo((2, 3, 30, 30, 16), "float32"),
    )


def test_conv2d_transpose_infer_struct_info_symbolic_groups():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    ic = tir.Var("c", "int64")
    oc = tir.Var("oc", "int64")
    x = relax.Var("x", R.Tensor((n, ic * 4, 28, 28), "float32"))
    w0 = relax.Var("w", R.Tensor((ic, oc, 3, 3), "float32"))

    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x, w0, groups=4),
        relax.TensorStructInfo((n, oc * 4, 30, 30), "float32"),
    )


def test_conv2d_transpose_infer_struct_info_input_channel_group_incompatible():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    ic = tir.Var("c", "int64")
    oc = tir.Var("oc", "int64")
    x0 = relax.Var("x", R.Tensor((2, 128, 28, 28), "float32"))
    w0 = relax.Var("w", R.Tensor((128, 20, 3, 3), "float32"))
    x1 = relax.Var("x", R.Tensor((n, ic, 28, 28), "float32"))
    w1 = relax.Var("w", R.Tensor((ic - 1, oc, 3, 3), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d_transpose(x0, w0, groups=6))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d_transpose(x1, w1, groups=6))


def test_conv2d_transpose_non_positive_group():
    x = relax.Var("x", R.Tensor((2, 128, 28, 28), "float32"))
    w = relax.Var("w", R.Tensor((128, 16, 3, 3), "float32"))

    with pytest.raises(TVMError):
        relax.op.nn.conv2d_transpose(x, w, groups=0)
    with pytest.raises(TVMError):
        relax.op.nn.conv2d_transpose(x, w, groups=-2)


def test_conv2d_transpose_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float16"))
    w0 = relax.Var("w", R.Tensor((3, 4, 3, 3), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float64"))
    w1 = relax.Var("w", R.Tensor((3, 4, 3, 3), "float64"))
    x2 = relax.Var("x", R.Tensor((2, 3, 28, 28), "int8"))
    w2 = relax.Var("w", R.Tensor((3, 4, 3, 3), "int8"))
    x3 = relax.Var("x", R.Tensor((2, 3, 28, 28), "int32"))
    w3 = relax.Var("w", R.Tensor((3, 4, 3, 3), "int32"))

    _check_inference(
        bb, relax.op.nn.conv2d_transpose(x0, w0), relax.TensorStructInfo((2, 4, 30, 30), "float16")
    )
    _check_inference(
        bb, relax.op.nn.conv2d_transpose(x1, w1), relax.TensorStructInfo((2, 4, 30, 30), "float64")
    )
    _check_inference(
        bb, relax.op.nn.conv2d_transpose(x2, w2), relax.TensorStructInfo((2, 4, 30, 30), "int8")
    )
    _check_inference(
        bb, relax.op.nn.conv2d_transpose(x3, w3), relax.TensorStructInfo((2, 4, 30, 30), "int32")
    )


def test_conv2d_transpose_unequal_input_channel():
    bb = relax.BlockBuilder()
    ic = tir.Var("ic", "int64")
    x0 = relax.Var("x", R.Tensor([2, 3, 28, 28], "float32"))
    w0 = relax.Var("w", R.Tensor([4, 3, 3, 3], "float32"))
    x1 = relax.Var("x", R.Tensor([2, ic, 28, 28], "float32"))
    w1 = relax.Var("w", R.Tensor([ic + 2, 4, 3, 3], "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d_transpose(x0, w0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d_transpose(x1, w1))


def test_conv2d_transpose_wrong_output_padding():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor([2, 3, 28, 28], "float32"))
    w0 = relax.Var("w", R.Tensor([3, 4, 3, 3], "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d_transpose(x0, w0, strides=2, output_padding=2))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d_transpose(x0, w0, strides=(2, 2), output_padding=(2, 2)))


def test_conv2d_transpose_stride_padding_dilation_int64():
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    w = relax.Var("w", R.Tensor((3, 4, 3, 3), "float32"))
    conv2d_transpose = relax.op.nn.conv2d_transpose(
        x, w, strides=(1, 1), padding=(1, 1), output_padding=(1, 2), dilation=(1, 1)
    )

    assert conv2d_transpose.attrs.strides[0].dtype == "int64"
    assert conv2d_transpose.attrs.strides[1].dtype == "int64"
    assert conv2d_transpose.attrs.padding[0].dtype == "int64"
    assert conv2d_transpose.attrs.padding[1].dtype == "int64"
    assert conv2d_transpose.attrs.padding[2].dtype == "int64"
    assert conv2d_transpose.attrs.padding[3].dtype == "int64"
    assert conv2d_transpose.attrs.output_padding[0].dtype == "int64"
    assert conv2d_transpose.attrs.output_padding[1].dtype == "int64"
    assert conv2d_transpose.attrs.dilation[0].dtype == "int64"
    assert conv2d_transpose.attrs.dilation[1].dtype == "int64"


def test_conv2d_transpose_wrong_strides_padding_dilation_length():
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    w = relax.Var("w", R.Tensor((3, 4, 3, 3), "float32"))
    with pytest.raises(TVMError):
        relax.op.nn.conv2d_transpose(x, w, strides=(1, 2, 3))
    with pytest.raises(TVMError):
        relax.op.nn.conv2d_transpose(x, w, padding=(1, 2, 3))
    with pytest.raises(TVMError):
        relax.op.nn.conv2d_transpose(x, w, output_padding=(1, 2, 3))
    with pytest.raises(TVMError):
        relax.op.nn.conv2d_transpose(x, w, dilation=(1, 2, 3))


def test_conv2d_transpose_infer_struct_info_wrong_layout_string():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    w = relax.Var("w", R.Tensor((3, 4, 3, 3), "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d_transpose(x, w, data_layout="IOHW"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d_transpose(x, w, kernel_layout="NHWC"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d_transpose(x, w, out_layout="OHWI"))


def test_conv2d_transpose_dtype_mismatch():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    w = relax.Var("w", R.Tensor((3, 4, 3, 3), "int8"))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.nn.conv2d_transpose(x, w))


def test_conv2d_transpose_wrong_input_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 3, 28, 28, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=3))
    w0 = relax.Var("w", R.Tensor((3, 4, 3, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((3, 4, 6, 3, 3), "float32"))
    w2 = relax.Var("w", R.Tensor("float32", ndim=6))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d_transpose(x0, w1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d_transpose(x0, w1, data_layout="NCHW16c"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d_transpose(x0, w2))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d_transpose(x1, w0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d_transpose(x2, w0))


def test_conv2d_transpose_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    x1 = relax.Var("x", relax.ShapeStructInfo((2, 3, 28, 28)))
    w0 = relax.Var("w", R.Tensor((3, 4, 3, 3), "float32"))
    w1 = relax.Var("w", relax.FuncStructInfo([], R.Tensor((3, 4, 3, 3), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d_transpose(x0, w1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.conv2d_transpose(x1, w0))


def test_conv3d_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 28, 28, 28, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=5))
    x3 = relax.Var("x", R.Tensor("float32"))
    x4 = relax.Var("x", R.Tensor())
    x5 = relax.Var("x", R.Tensor((2, 4, 28, 28, 28, 16), "float32"))
    x6 = relax.Var("x", R.Tensor((2, 3, 28, 28, 28), "float32", vdev0))
    w0 = relax.Var("w", R.Tensor((4, 3, 3, 3, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((3, 4, 3, 3, 3), "float32"))
    w2 = relax.Var("w", R.Tensor("float32", ndim=5))
    w3 = relax.Var("w", R.Tensor("float32"))
    w4 = relax.Var("w", R.Tensor((48, 4, 3, 3, 3, 16), "float32"))
    w5 = relax.Var("w", R.Tensor((4, 3, 3, 3, 3), "float32", vdev0))

    _check_inference(
        bb, relax.op.nn.conv3d(x0, w0), relax.TensorStructInfo((2, 4, 26, 26, 26), "float32")
    )
    _check_inference(
        bb, relax.op.nn.conv3d(x6, w5), relax.TensorStructInfo((2, 4, 26, 26, 26), "float32", vdev0)
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w0, out_dtype="float16"),
        relax.TensorStructInfo((2, 4, 26, 26, 26), "float16"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w0, padding=1),
        relax.TensorStructInfo((2, 4, 28, 28, 28), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w0, padding=[1, 2, 3]),
        relax.TensorStructInfo((2, 4, 28, 30, 32), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w0, padding=[1, 2, 3, 4, 5, 6]),
        relax.TensorStructInfo((2, 4, 31, 33, 35), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w0, strides=2),
        relax.TensorStructInfo((2, 4, 13, 13, 13), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w0, strides=(2, 3, 4)),
        relax.TensorStructInfo((2, 4, 13, 9, 7), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w0, dilation=2),
        relax.TensorStructInfo((2, 4, 24, 24, 24), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w0, dilation=(3, 2, 1)),
        relax.TensorStructInfo((2, 4, 22, 24, 26), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x1, w0, data_layout="NDHWC"),
        relax.TensorStructInfo((2, 26, 26, 26, 4), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w0, out_layout="NDHWC"),
        relax.TensorStructInfo((2, 26, 26, 26, 4), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w1, kernel_layout="IODHW"),
        relax.TensorStructInfo((2, 4, 26, 26, 26), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(
            x5, w4, data_layout="NCDHW16c", kernel_layout="OIDHW16i", out_layout="NDHWC16c"
        ),
        relax.TensorStructInfo((2, 26, 26, 26, 3, 16), "float32"),
    )
    _check_inference(
        bb, relax.op.nn.conv3d(x2, w0), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(
        bb, relax.op.nn.conv3d(x3, w0), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(
        bb, relax.op.nn.conv3d(x0, w2), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(
        bb, relax.op.nn.conv3d(x0, w3), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(bb, relax.op.nn.conv3d(x4, w0), relax.TensorStructInfo(dtype="", ndim=5))


def test_conv3d_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    c = tir.Var("c", "int64")
    c16 = tir.Var("c16", "int64")
    id = tir.Var("id", "int64")
    ih = tir.Var("ih", "int64")
    iw = tir.Var("iw", "int64")
    ki = tir.Var("ki", "int64")
    ko = tir.Var("ko", "int64")
    kd = tir.Var("kd", "int64")
    kh = tir.Var("kh", "int64")
    kw = tir.Var("kw", "int64")
    x0 = relax.Var("x", R.Tensor((n, c, id, ih, iw), "float32"))
    x1 = relax.Var("x", R.Tensor((n, c, id, ih, iw, c16), "float32"))
    w0 = relax.Var("w", R.Tensor((ko, ki, kd, kh, kw), "float32"))
    w1 = relax.Var("w", R.Tensor((ko, c, kd, kh, kw), "float32"))
    w2 = relax.Var("w", R.Tensor((ko, c, kd, kh, kw, c16), "float32"))

    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w0),
        relax.TensorStructInfo((n, ko, id + 1 - kd, ih + 1 - kh, iw + 1 - kw), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w1),
        relax.TensorStructInfo((n, ko, id + 1 - kd, ih + 1 - kh, iw + 1 - kw), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(
            x1, w2, data_layout="NCDHW16c", kernel_layout="OIDHW16i", out_layout="NCDHW"
        ),
        relax.TensorStructInfo((n, ko, id + 1 - kd, ih + 1 - kh, iw + 1 - kw), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w0, strides=(2, 2, 2), padding=(1, 1, 1), dilation=(2, 2, 2)),
        relax.TensorStructInfo(
            (
                n,
                ko,
                tvm.tir.floordiv(id + 3, 2) + 1 - kd,
                tvm.tir.floordiv(ih + 3, 2) + 1 - kh,
                tvm.tir.floordiv(iw + 3, 2) + 1 - kw,
            ),
            "float32",
        ),
    )


def test_conv3d_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=5))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=6))
    s2 = relax.Var("s", relax.ShapeStructInfo(ndim=5))
    s3 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s3, "float32"))
    w = relax.Var("w", relax.TensorStructInfo(s2, "float32"))

    _check_inference(bb, relax.op.nn.conv3d(x0, w), relax.TensorStructInfo(dtype="float32", ndim=5))
    _check_inference(
        bb,
        relax.op.nn.conv3d(x1, w, data_layout="NCDHW16c"),
        relax.TensorStructInfo(dtype="float32", ndim=6),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w, out_layout="NCDHW16c"),
        relax.TensorStructInfo(dtype="float32", ndim=6),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x2, w),
        relax.TensorStructInfo(dtype="float32", ndim=5),
    )


if __name__ == "__main__":
    tvm.testing.main()
