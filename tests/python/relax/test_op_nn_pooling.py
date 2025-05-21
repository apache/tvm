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


def test_op_correctness():
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    x1 = relax.Var("x1", R.Tensor((2, 3, 64), "float32"))
    x2 = relax.Var("x2", R.Tensor((2, 3, 8, 28, 28), "float32"))
    assert relax.op.nn.max_pool1d(x1).op == Op.get("relax.nn.max_pool1d")
    assert relax.op.nn.max_pool2d(x).op == Op.get("relax.nn.max_pool2d")
    assert relax.op.nn.max_pool3d(x2).op == Op.get("relax.nn.max_pool3d")
    assert relax.op.nn.avg_pool1d(x).op == Op.get("relax.nn.avg_pool1d")
    assert relax.op.nn.avg_pool2d(x).op == Op.get("relax.nn.avg_pool2d")
    assert relax.op.nn.avg_pool3d(x).op == Op.get("relax.nn.avg_pool3d")
    assert relax.op.nn.adaptive_avg_pool1d(x).op == Op.get("relax.nn.adaptive_avg_pool1d")
    assert relax.op.nn.adaptive_avg_pool2d(x).op == Op.get("relax.nn.adaptive_avg_pool2d")
    assert relax.op.nn.adaptive_avg_pool3d(x).op == Op.get("relax.nn.adaptive_avg_pool3d")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_sinfo: relax.StructInfo):
    ret = bb.normalize(call)
    tvm.ir.assert_structural_equal(ret.struct_info, expected_sinfo)


def test_max_pool1d_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 32), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor(ndim=3))
    x3 = relax.Var("x", R.Tensor("float32"))
    x4 = relax.Var("x", R.Tensor())
    x5 = relax.Var("x", R.Tensor((2, 3, 32), "float32", vdev0))

    _check_inference(bb, relax.op.nn.max_pool1d(x0), relax.TensorStructInfo((2, 3, 32), "float32"))
    _check_inference(
        bb, relax.op.nn.max_pool1d(x5), relax.TensorStructInfo((2, 3, 32), "float32", vdev0)
    )
    _check_inference(
        bb, relax.op.nn.max_pool1d(x0, pool_size=3), relax.TensorStructInfo((2, 3, 30), "float32")
    )
    _check_inference(
        bb, relax.op.nn.max_pool1d(x0, strides=2), relax.TensorStructInfo((2, 3, 16), "float32")
    )
    _check_inference(
        bb, relax.op.nn.max_pool1d(x0, padding=1), relax.TensorStructInfo((2, 3, 34), "float32")
    )
    _check_inference(
        bb, relax.op.nn.max_pool1d(x0, dilation=2), relax.TensorStructInfo((2, 3, 32), "float32")
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool1d(x0, layout="NCW", out_layout="NWC"),
        relax.TensorStructInfo((2, 32, 3), "float32"),
    )
    _check_inference(
        bb, relax.op.nn.max_pool1d(x1), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(bb, relax.op.nn.max_pool1d(x2), relax.TensorStructInfo(dtype="", ndim=3))
    _check_inference(
        bb, relax.op.nn.max_pool1d(x3), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(bb, relax.op.nn.max_pool1d(x4), relax.TensorStructInfo(dtype="", ndim=3))


def test_max_pool1d_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    c = tir.Var("c", "int64")
    w = tir.Var("w", "int64")
    c16 = tir.Var("c16", "int64")

    x0 = relax.Var("x", R.Tensor((n, c, w), "float32"))
    x1 = relax.Var("x", R.Tensor((n, c, w, c16), "float32"))

    _check_inference(
        bb,
        relax.op.nn.max_pool1d(x0, pool_size=3, strides=3, padding=2, dilation=2),
        relax.TensorStructInfo(
            (
                n,
                c,
                tvm.tir.floordiv(w - 1, 3) + 1,
            ),
            "float32",
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool1d(x1, layout="NCW16c", out_layout="NWC"),
        relax.TensorStructInfo((n, w, c * 16), "float32"),
    )


def test_max_pool1d_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=3))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=4))
    s2 = relax.Var("s", relax.ShapeStructInfo())

    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))

    _check_inference(
        bb, relax.op.nn.max_pool1d(x0), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool1d(x1, layout="NCW16c"),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool1d(x2),
        relax.TensorStructInfo(dtype="float32", ndim=3),
    )


def test_max_pool1d_infer_struct_info_ceil_mode():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 32), "float32"))

    _check_inference(
        bb,
        relax.op.nn.max_pool1d(x, pool_size=3, strides=2, ceil_mode=True),
        relax.TensorStructInfo((2, 3, 16), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool1d(x, pool_size=5, strides=2, ceil_mode=True),
        relax.TensorStructInfo((2, 3, 15), "float32"),
    )


def test_max_pool1d_infer_struct_info_ceil_mode_symbolic():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    c = tir.Var("c", "int64")
    w = tir.Var("w", "int64")
    x = relax.Var("x", R.Tensor((n, c, w), "float32"))

    _check_inference(
        bb,
        relax.op.nn.max_pool1d(x, pool_size=3, strides=2, padding=1, dilation=2, ceil_mode=True),
        relax.TensorStructInfo((n, c, tvm.tir.floordiv(w, 2)), "float32"),
    )


def test_max_pool1d_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 32), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 32), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3, 32), "int64"))

    _check_inference(bb, relax.op.nn.max_pool1d(x0), relax.TensorStructInfo((2, 3, 32), "float16"))
    _check_inference(bb, relax.op.nn.max_pool1d(x1), relax.TensorStructInfo((2, 3, 32), "int8"))
    _check_inference(bb, relax.op.nn.max_pool1d(x2), relax.TensorStructInfo((2, 3, 32), "int64"))


def test_max_pool1d_stride_padding_dilation_int64():
    x = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    max_pool1d = relax.op.nn.max_pool1d(x, pool_size=3, strides=1, padding=1, dilation=1)

    assert max_pool1d.attrs.strides[0].dtype == "int64"
    assert max_pool1d.attrs.padding[0].dtype == "int64"
    assert max_pool1d.attrs.padding[1].dtype == "int64"
    assert max_pool1d.attrs.dilation[0].dtype == "int64"


def test_max_pool1d_wrong_pool_size_strides_padding_dilation_length():
    x = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    with pytest.raises(TVMError):
        relax.op.nn.max_pool1d(x, pool_size=(1, 2))
    with pytest.raises(TVMError):
        relax.op.nn.max_pool1d(x, strides=(1, 2))
    with pytest.raises(TVMError):
        relax.op.nn.max_pool1d(x, padding=(1, 2, 3))
    with pytest.raises(TVMError):
        relax.op.nn.max_pool1d(x, dilation=(1, 2))


def test_max_pool1d_infer_struct_info_wrong_layout_string():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.max_pool1d(x, layout="OIW"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.max_pool1d(x, out_layout="OWI"))


def test_max_pool1d_wrong_input_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=5))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.max_pool1d(x0))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.max_pool1d(x1))


def test_max_pool1d_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 28)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 28), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.max_pool1d(x0))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.max_pool1d(x1))


def test_max_pool2d_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 32, 32, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=4))
    x3 = relax.Var("x", R.Tensor("float32"))
    x4 = relax.Var("x", R.Tensor(ndim=4))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((2, 4, 32, 32, 16), "float32"))
    x7 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32", vdev0))

    _check_inference(
        bb, relax.op.nn.max_pool2d(x0), relax.TensorStructInfo((2, 3, 32, 32), "float32")
    )
    _check_inference(
        bb, relax.op.nn.max_pool2d(x7), relax.TensorStructInfo((2, 3, 32, 32), "float32", vdev0)
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool2d(x0, pool_size=3),
        relax.TensorStructInfo((2, 3, 30, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool2d(x0, pool_size=(5, 3)),
        relax.TensorStructInfo((2, 3, 28, 30), "float32"),
    )
    _check_inference(
        bb, relax.op.nn.max_pool2d(x0, padding=1), relax.TensorStructInfo((2, 3, 34, 34), "float32")
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool2d(x0, padding=[1, 2]),
        relax.TensorStructInfo((2, 3, 34, 36), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool2d(x0, strides=2),
        relax.TensorStructInfo((2, 3, 16, 16), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool2d(x0, dilation=2),
        relax.TensorStructInfo((2, 3, 32, 32), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool2d(x1, layout="NHWC"),
        relax.TensorStructInfo((2, 32, 32, 3), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool2d(x0, out_layout="NHWC"),
        relax.TensorStructInfo((2, 32, 32, 3), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool2d(x6, layout="NCHW16c", out_layout="NHWC16c"),
        relax.TensorStructInfo((2, 32, 32, 4, 16), "float32"),
    )
    _check_inference(
        bb, relax.op.nn.max_pool2d(x2), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.nn.max_pool2d(x3), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(bb, relax.op.nn.max_pool2d(x4), relax.TensorStructInfo(dtype="", ndim=4))
    _check_inference(bb, relax.op.nn.max_pool2d(x5), relax.TensorStructInfo(dtype="", ndim=4))


def test_max_pool2d_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    c = tir.Var("c", "int64")
    c16 = tir.Var("c16", "int64")
    ih = tir.Var("ih", "int64")
    iw = tir.Var("iw", "int64")
    x0 = relax.Var("x", R.Tensor((n, c, ih, iw), "float32"))
    x1 = relax.Var("x", R.Tensor((n, c, ih, iw, c16), "float32"))

    _check_inference(
        bb,
        relax.op.nn.max_pool2d(
            x0, pool_size=(3, 3), strides=(3, 3), padding=(2, 2), dilation=(2, 2)
        ),
        relax.TensorStructInfo(
            (
                n,
                c,
                tvm.tir.floordiv(ih - 1, 3) + 1,
                tvm.tir.floordiv(iw - 1, 3) + 1,
            ),
            "float32",
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool2d(x1, layout="NCHW16c", out_layout="NHWC"),
        relax.TensorStructInfo((n, ih, iw, c * 16), "float32"),
    )


def test_max_pool2d_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=4))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=5))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))

    _check_inference(
        bb, relax.op.nn.max_pool2d(x0), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool2d(x1, layout="NCHW16c"),
        relax.TensorStructInfo(dtype="float32", ndim=5),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool2d(x2),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )


def test_max_pool2d_infer_struct_info_ceil_mode():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32"))

    _check_inference(
        bb,
        relax.op.nn.max_pool2d(x, pool_size=3, strides=2, ceil_mode=True),
        relax.TensorStructInfo((2, 3, 16, 16), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool2d(x, pool_size=(5, 3), strides=2, ceil_mode=True),
        relax.TensorStructInfo((2, 3, 15, 16), "float32"),
    )


def test_max_pool2d_infer_struct_info_ceil_mode_symbolic():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    c = tir.Var("c", "int64")
    ih = tir.Var("ih", "int64")
    iw = tir.Var("iw", "int64")
    x = relax.Var("x", R.Tensor((n, c, ih, iw), "float32"))

    _check_inference(
        bb,
        relax.op.nn.max_pool2d(
            x, pool_size=(3, 3), strides=(2, 2), padding=(1, 1), dilation=(2, 2), ceil_mode=True
        ),
        relax.TensorStructInfo((n, c, tvm.tir.floordiv(ih, 2), tvm.tir.floordiv(iw, 2)), "float32"),
    )


def test_max_pool2d_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 32, 32), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3, 32, 32), "int64"))
    _check_inference(
        bb, relax.op.nn.max_pool2d(x0), relax.TensorStructInfo((2, 3, 32, 32), "float16")
    )
    _check_inference(bb, relax.op.nn.max_pool2d(x1), relax.TensorStructInfo((2, 3, 32, 32), "int8"))
    _check_inference(
        bb, relax.op.nn.max_pool2d(x2), relax.TensorStructInfo((2, 3, 32, 32), "int64")
    )


def test_max_pool2d_stride_padding_dilation_int64():
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    max_pool2d = relax.op.nn.max_pool2d(x, (3, 3), strides=(1, 1), padding=(1, 1), dilation=(1, 1))

    assert max_pool2d.attrs.strides[0].dtype == "int64"
    assert max_pool2d.attrs.strides[1].dtype == "int64"
    assert max_pool2d.attrs.padding[0].dtype == "int64"
    assert max_pool2d.attrs.padding[1].dtype == "int64"
    assert max_pool2d.attrs.padding[2].dtype == "int64"
    assert max_pool2d.attrs.padding[3].dtype == "int64"
    assert max_pool2d.attrs.dilation[0].dtype == "int64"
    assert max_pool2d.attrs.dilation[1].dtype == "int64"


def test_max_pool2d_wrong_pool_size_strides_padding_dilation_length():
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    with pytest.raises(TVMError):
        relax.op.nn.max_pool2d(x, pool_size=(1, 2, 3))
    with pytest.raises(TVMError):
        relax.op.nn.max_pool2d(x, strides=(1, 2, 3))
    with pytest.raises(TVMError):
        relax.op.nn.max_pool2d(x, padding=(1, 2, 3))
    with pytest.raises(TVMError):
        relax.op.nn.max_pool2d(x, dilation=(1, 2, 3))


def test_max_pool2d_infer_struct_info_wrong_layout_string():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.max_pool2d(x, layout="OIHW"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.max_pool2d(x, out_layout="OHWI"))


def test_max_pool2d_wrong_input_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.max_pool2d(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.max_pool2d(x1))


def test_max_pool2d_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 28, 28)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 28, 28), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.max_pool2d(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.max_pool2d(x1))


def test_max_pool3d_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 16, 32, 32), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 16, 32, 32, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=5))
    x3 = relax.Var("x", R.Tensor("float32"))
    x4 = relax.Var("x", R.Tensor(ndim=5))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((2, 4, 16, 32, 32, 16), "float32"))
    x7 = relax.Var("x", R.Tensor((2, 3, 16, 32, 32), "float32", vdev0))

    _check_inference(
        bb, relax.op.nn.max_pool3d(x0), relax.TensorStructInfo((2, 3, 16, 32, 32), "float32")
    )
    _check_inference(
        bb, relax.op.nn.max_pool3d(x7), relax.TensorStructInfo((2, 3, 16, 32, 32), "float32", vdev0)
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool3d(x0, pool_size=3),
        relax.TensorStructInfo((2, 3, 14, 30, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool3d(x0, pool_size=(3, 5, 3)),
        relax.TensorStructInfo((2, 3, 14, 28, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool3d(x0, padding=1),
        relax.TensorStructInfo((2, 3, 18, 34, 34), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool3d(x0, padding=[1, 2, 3]),
        relax.TensorStructInfo((2, 3, 18, 36, 38), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool3d(x0, strides=2),
        relax.TensorStructInfo((2, 3, 8, 16, 16), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool3d(x0, dilation=2),
        relax.TensorStructInfo((2, 3, 16, 32, 32), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool3d(x1, layout="NDHWC"),
        relax.TensorStructInfo((2, 16, 32, 32, 3), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool3d(x0, out_layout="NDHWC"),
        relax.TensorStructInfo((2, 16, 32, 32, 3), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool3d(x6, layout="NCDHW16c", out_layout="NDHWC16c"),
        relax.TensorStructInfo((2, 16, 32, 32, 4, 16), "float32"),
    )
    _check_inference(
        bb, relax.op.nn.max_pool3d(x2), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(
        bb, relax.op.nn.max_pool3d(x3), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(bb, relax.op.nn.max_pool3d(x4), relax.TensorStructInfo(dtype="", ndim=5))
    _check_inference(bb, relax.op.nn.max_pool3d(x5), relax.TensorStructInfo(dtype="", ndim=5))


def test_max_pool3d_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    c = tir.Var("c", "int64")
    c16 = tir.Var("c16", "int64")
    id = tir.Var("id", "int64")
    ih = tir.Var("ih", "int64")
    iw = tir.Var("iw", "int64")
    x0 = relax.Var("x", R.Tensor((n, c, id, ih, iw), "float32"))
    x1 = relax.Var("x", R.Tensor((n, c, id, ih, iw, c16), "float32"))

    _check_inference(
        bb,
        relax.op.nn.max_pool3d(
            x0, pool_size=(3, 3, 3), strides=(3, 3, 3), padding=(2, 2, 2), dilation=(2, 2, 2)
        ),
        relax.TensorStructInfo(
            (
                n,
                c,
                tvm.tir.floordiv(id - 1, 3) + 1,
                tvm.tir.floordiv(ih - 1, 3) + 1,
                tvm.tir.floordiv(iw - 1, 3) + 1,
            ),
            "float32",
        ),
    )

    _check_inference(
        bb,
        relax.op.nn.max_pool3d(x1, layout="NCDHW16c", out_layout="NDHWC"),
        relax.TensorStructInfo((n, id, ih, iw, c * 16), "float32"),
    )


def test_max_pool3d_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=5))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=6))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))

    _check_inference(
        bb, relax.op.nn.max_pool3d(x0), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool3d(x1, layout="NCDHW16c"),
        relax.TensorStructInfo(dtype="float32", ndim=6),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool3d(x2),
        relax.TensorStructInfo(dtype="float32", ndim=5),
    )


def test_max_pool3d_infer_struct_info_ceil_mode():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 32, 32, 32), "float32"))

    _check_inference(
        bb,
        relax.op.nn.max_pool3d(x, pool_size=3, strides=2, ceil_mode=True),
        relax.TensorStructInfo((2, 3, 16, 16, 16), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.max_pool3d(x, pool_size=(5, 3, 3), strides=2, ceil_mode=True),
        relax.TensorStructInfo((2, 3, 15, 16, 16), "float32"),
    )


def test_max_pool3d_infer_struct_info_ceil_mode_symbolic():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    c = tir.Var("c", "int64")
    id_ = tir.Var("id", "int64")
    ih = tir.Var("ih", "int64")
    iw = tir.Var("iw", "int64")
    x = relax.Var("x", R.Tensor((n, c, id_, ih, iw), "float32"))

    _check_inference(
        bb,
        relax.op.nn.max_pool3d(
            x,
            pool_size=(3, 3, 3),
            strides=(2, 2, 2),
            padding=(1, 1, 1),
            dilation=(2, 2, 2),
            ceil_mode=True,
        ),
        relax.TensorStructInfo(
            (n, c, tvm.tir.floordiv(id_, 2), tvm.tir.floordiv(ih, 2), tvm.tir.floordiv(iw, 2)),
            "float32",
        ),
    )


def test_max_pool3d_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32, 32), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 32, 32, 32), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3, 32, 32, 32), "int64"))
    _check_inference(
        bb, relax.op.nn.max_pool3d(x0), relax.TensorStructInfo((2, 3, 32, 32, 32), "float16")
    )
    _check_inference(
        bb, relax.op.nn.max_pool3d(x1), relax.TensorStructInfo((2, 3, 32, 32, 32), "int8")
    )
    _check_inference(
        bb, relax.op.nn.max_pool3d(x2), relax.TensorStructInfo((2, 3, 32, 32, 32), "int64")
    )


def test_max_pool3d_stride_padding_dilation_int64():
    x = relax.Var("x", R.Tensor((2, 3, 28, 28, 28), "float32"))
    max_pool3d = relax.op.nn.max_pool3d(
        x, (3, 3, 3), strides=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1)
    )

    assert max_pool3d.attrs.strides[0].dtype == "int64"
    assert max_pool3d.attrs.strides[1].dtype == "int64"
    assert max_pool3d.attrs.strides[2].dtype == "int64"
    assert max_pool3d.attrs.padding[0].dtype == "int64"
    assert max_pool3d.attrs.padding[1].dtype == "int64"
    assert max_pool3d.attrs.padding[2].dtype == "int64"
    assert max_pool3d.attrs.padding[3].dtype == "int64"
    assert max_pool3d.attrs.padding[4].dtype == "int64"
    assert max_pool3d.attrs.dilation[0].dtype == "int64"
    assert max_pool3d.attrs.dilation[1].dtype == "int64"
    assert max_pool3d.attrs.dilation[2].dtype == "int64"


def test_max_pool3d_wrong_pool_size_strides_padding_dilation_length():
    x = relax.Var("x", R.Tensor((2, 3, 28, 28, 28), "float32"))
    with pytest.raises(TVMError):
        relax.op.nn.max_pool3d(x, pool_size=(1, 2, 3, 4))
    with pytest.raises(TVMError):
        relax.op.nn.max_pool3d(x, strides=(1, 2, 3, 4))
    with pytest.raises(TVMError):
        relax.op.nn.max_pool3d(x, padding=(1, 2, 3, 4))
    with pytest.raises(TVMError):
        relax.op.nn.max_pool3d(x, dilation=(1, 2, 3, 4))


def test_max_pool3d_infer_struct_info_wrong_layout_string():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28, 28, 28), "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.max_pool3d(x, layout="OIHW"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.max_pool3d(x, out_layout="OHWI"))


def test_max_pool3d_wrong_input_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28, 28, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.max_pool3d(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.max_pool3d(x1))


def test_max_pool3d_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 28, 28, 28)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 28, 28, 28), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.max_pool3d(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.max_pool3d(x1))


def test_avg_pool1d_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 32), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 32, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=3))
    x3 = relax.Var("x", R.Tensor("float32"))
    x4 = relax.Var("x", R.Tensor(ndim=3))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((2, 4, 32, 16), "float32"))
    x7 = relax.Var("x", R.Tensor((2, 3, 32), "float32", vdev0))

    _check_inference(bb, relax.op.nn.avg_pool1d(x0), relax.TensorStructInfo((2, 3, 32), "float32"))
    _check_inference(
        bb, relax.op.nn.avg_pool1d(x7), relax.TensorStructInfo((2, 3, 32), "float32", vdev0)
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool1d(x0, pool_size=3),
        relax.TensorStructInfo((2, 3, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool1d(x0, padding=1),
        relax.TensorStructInfo((2, 3, 34), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool1d(x0, padding=[1, 2]),
        relax.TensorStructInfo((2, 3, 35), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool1d(x0, strides=2),
        relax.TensorStructInfo((2, 3, 16), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool1d(x0, dilation=2),
        relax.TensorStructInfo((2, 3, 32), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool1d(x1, layout="NWC"),
        relax.TensorStructInfo((2, 32, 3), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool1d(x0, out_layout="NWC"),
        relax.TensorStructInfo((2, 32, 3), "float32"),
    )
    _check_inference(
        bb, relax.op.nn.avg_pool1d(x2), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.nn.avg_pool1d(x3), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(bb, relax.op.nn.avg_pool1d(x4), relax.TensorStructInfo(dtype="", ndim=3))
    _check_inference(bb, relax.op.nn.avg_pool1d(x5), relax.TensorStructInfo(dtype="", ndim=3))


def test_avg_pool1d_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    c = tir.Var("c", "int64")
    c16 = tir.Var("c16", "int64")
    iw = tir.Var("iw", "int64")
    x0 = relax.Var("x", R.Tensor((n, c, iw), "float32"))
    x1 = relax.Var("x", R.Tensor((n, c, iw, c16), "float32"))

    _check_inference(
        bb,
        relax.op.nn.avg_pool1d(x0, pool_size=3, strides=3, padding=2, dilation=2),
        relax.TensorStructInfo(
            (
                n,
                c,
                tvm.tir.floordiv(iw - 1, 3) + 1,
            ),
            "float32",
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool1d(x1, layout="NCW16c", out_layout="NWC"),
        relax.TensorStructInfo((n, iw, c * 16), "float32"),
    )


def test_avg_pool1d_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=3))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=4))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))

    _check_inference(
        bb, relax.op.nn.avg_pool1d(x0), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool1d(x1, layout="NCW16c"),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool1d(x2),
        relax.TensorStructInfo(dtype="float32", ndim=3),
    )


def test_avg_pool1d_infer_struct_info_ceil_mode():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 32), "float32"))

    _check_inference(
        bb,
        relax.op.nn.avg_pool1d(x, pool_size=3, strides=2, ceil_mode=True),
        relax.TensorStructInfo((2, 3, 16), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool1d(x, pool_size=5, strides=2, ceil_mode=True),
        relax.TensorStructInfo((2, 3, 15), "float32"),
    )


def test_avg_pool1d_infer_struct_info_ceil_mode_symbolic():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    c = tir.Var("c", "int64")
    iw = tir.Var("iw", "int64")
    x = relax.Var("x", R.Tensor((n, c, iw), "float32"))

    _check_inference(
        bb,
        relax.op.nn.avg_pool1d(x, pool_size=3, strides=2, padding=1, dilation=2, ceil_mode=True),
        relax.TensorStructInfo(
            (n, c, tvm.tir.floordiv(iw, 2)),
            "float32",
        ),
    )


def test_avg_pool1d_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 32), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 32), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3, 32), "int64"))
    _check_inference(bb, relax.op.nn.avg_pool1d(x0), relax.TensorStructInfo((2, 3, 32), "float16"))
    _check_inference(bb, relax.op.nn.avg_pool1d(x1), relax.TensorStructInfo((2, 3, 32), "int8"))
    _check_inference(bb, relax.op.nn.avg_pool1d(x2), relax.TensorStructInfo((2, 3, 32), "int64"))


def test_avg_pool1d_stride_padding_dilation_int64():
    x = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    avg_pool1d = relax.op.nn.avg_pool1d(x, 3, strides=1, padding=1, dilation=1)

    assert avg_pool1d.attrs.strides[0].dtype == "int64"
    assert avg_pool1d.attrs.padding[0].dtype == "int64"
    assert avg_pool1d.attrs.padding[1].dtype == "int64"
    assert avg_pool1d.attrs.dilation[0].dtype == "int64"


def test_avg_pool1d_wrong_pool_size_strides_padding_dilation_length():
    x = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    with pytest.raises(TVMError):
        relax.op.nn.avg_pool1d(x, pool_size=(1, 2))
    with pytest.raises(TVMError):
        relax.op.nn.avg_pool1d(x, strides=(1, 2))
    with pytest.raises(TVMError):
        relax.op.nn.avg_pool1d(x, padding=(1, 2, 3))
    with pytest.raises(TVMError):
        relax.op.nn.avg_pool1d(x, dilation=(1, 2))


def test_avg_pool1d_infer_struct_info_wrong_layout_string():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.avg_pool1d(x, layout="OIW"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.avg_pool1d(x, out_layout="OWI"))


def test_avg_pool1d_wrong_input_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=2))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.avg_pool1d(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.avg_pool1d(x1))


def test_avg_pool1d_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 28)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 28), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.avg_pool1d(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.avg_pool1d(x1))


def test_avg_pool2d_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 32, 32, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=4))
    x3 = relax.Var("x", R.Tensor("float32"))
    x4 = relax.Var("x", R.Tensor(ndim=4))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((2, 4, 32, 32, 16), "float32"))
    x7 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32", vdev0))

    _check_inference(
        bb, relax.op.nn.avg_pool2d(x0), relax.TensorStructInfo((2, 3, 32, 32), "float32")
    )
    _check_inference(
        bb, relax.op.nn.avg_pool2d(x7), relax.TensorStructInfo((2, 3, 32, 32), "float32", vdev0)
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool2d(x0, pool_size=3),
        relax.TensorStructInfo((2, 3, 30, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool2d(x0, pool_size=(5, 3)),
        relax.TensorStructInfo((2, 3, 28, 30), "float32"),
    )
    _check_inference(
        bb, relax.op.nn.avg_pool2d(x0, padding=1), relax.TensorStructInfo((2, 3, 34, 34), "float32")
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool2d(x0, padding=[1, 2]),
        relax.TensorStructInfo((2, 3, 34, 36), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool2d(x0, strides=2),
        relax.TensorStructInfo((2, 3, 16, 16), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool2d(x0, dilation=2),
        relax.TensorStructInfo((2, 3, 32, 32), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool2d(x1, layout="NHWC"),
        relax.TensorStructInfo((2, 32, 32, 3), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool2d(x0, out_layout="NHWC"),
        relax.TensorStructInfo((2, 32, 32, 3), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool2d(x6, layout="NCHW16c", out_layout="NHWC16c"),
        relax.TensorStructInfo((2, 32, 32, 4, 16), "float32"),
    )
    _check_inference(
        bb, relax.op.nn.avg_pool2d(x2), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.nn.avg_pool2d(x3), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(bb, relax.op.nn.avg_pool2d(x4), relax.TensorStructInfo(dtype="", ndim=4))
    _check_inference(bb, relax.op.nn.avg_pool2d(x5), relax.TensorStructInfo(dtype="", ndim=4))


def test_avg_pool2d_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    c = tir.Var("c", "int64")
    c16 = tir.Var("c16", "int64")
    ih = tir.Var("ih", "int64")
    iw = tir.Var("iw", "int64")
    x0 = relax.Var("x", R.Tensor((n, c, ih, iw), "float32"))
    x1 = relax.Var("x", R.Tensor((n, c, ih, iw, c16), "float32"))

    _check_inference(
        bb,
        relax.op.nn.avg_pool2d(
            x0, pool_size=(3, 3), strides=(3, 3), padding=(2, 2), dilation=(2, 2)
        ),
        relax.TensorStructInfo(
            (
                n,
                c,
                tvm.tir.floordiv(ih - 1, 3) + 1,
                tvm.tir.floordiv(iw - 1, 3) + 1,
            ),
            "float32",
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool2d(x1, layout="NCHW16c", out_layout="NHWC"),
        relax.TensorStructInfo((n, ih, iw, c * 16), "float32"),
    )


def test_avg_pool2d_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=4))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=5))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))

    _check_inference(
        bb, relax.op.nn.avg_pool2d(x0), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool2d(x1, layout="NCHW16c"),
        relax.TensorStructInfo(dtype="float32", ndim=5),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool2d(x2),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )


def test_avg_pool2d_infer_struct_info_ceil_mode():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32"))

    _check_inference(
        bb,
        relax.op.nn.avg_pool2d(x, pool_size=3, strides=2, ceil_mode=True),
        relax.TensorStructInfo((2, 3, 16, 16), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool2d(x, pool_size=(5, 3), strides=2, ceil_mode=True),
        relax.TensorStructInfo((2, 3, 15, 16), "float32"),
    )


def test_avg_pool2d_infer_struct_info_ceil_mode_symbolic():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    c = tir.Var("c", "int64")
    ih = tir.Var("ih", "int64")
    iw = tir.Var("iw", "int64")
    x = relax.Var("x", R.Tensor((n, c, ih, iw), "float32"))

    _check_inference(
        bb,
        relax.op.nn.avg_pool2d(
            x, pool_size=(3, 3), strides=(2, 2), padding=(1, 1), dilation=(2, 2), ceil_mode=True
        ),
        relax.TensorStructInfo((n, c, tvm.tir.floordiv(ih, 2), tvm.tir.floordiv(iw, 2)), "float32"),
    )


def test_avg_pool2d_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 32, 32), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3, 32, 32), "int64"))
    _check_inference(
        bb, relax.op.nn.avg_pool2d(x0), relax.TensorStructInfo((2, 3, 32, 32), "float16")
    )
    _check_inference(bb, relax.op.nn.avg_pool2d(x1), relax.TensorStructInfo((2, 3, 32, 32), "int8"))
    _check_inference(
        bb, relax.op.nn.avg_pool2d(x2), relax.TensorStructInfo((2, 3, 32, 32), "int64")
    )


def test_avg_pool2d_stride_padding_dilation_int64():
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    avg_pool2d = relax.op.nn.avg_pool2d(x, (3, 3), strides=(1, 1), padding=(1, 1), dilation=(1, 1))

    assert avg_pool2d.attrs.strides[0].dtype == "int64"
    assert avg_pool2d.attrs.strides[1].dtype == "int64"
    assert avg_pool2d.attrs.padding[0].dtype == "int64"
    assert avg_pool2d.attrs.padding[1].dtype == "int64"
    assert avg_pool2d.attrs.padding[2].dtype == "int64"
    assert avg_pool2d.attrs.padding[3].dtype == "int64"
    assert avg_pool2d.attrs.dilation[0].dtype == "int64"
    assert avg_pool2d.attrs.dilation[1].dtype == "int64"


def test_avg_pool2d_wrong_pool_size_strides_padding_dilation_length():
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    with pytest.raises(TVMError):
        relax.op.nn.avg_pool2d(x, pool_size=(1, 2, 3))
    with pytest.raises(TVMError):
        relax.op.nn.avg_pool2d(x, strides=(1, 2, 3))
    with pytest.raises(TVMError):
        relax.op.nn.avg_pool2d(x, padding=(1, 2, 3))
    with pytest.raises(TVMError):
        relax.op.nn.avg_pool2d(x, dilation=(1, 2, 3))


def test_avg_pool2d_infer_struct_info_wrong_layout_string():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.avg_pool2d(x, layout="OIHW"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.avg_pool2d(x, out_layout="OHWI"))


def test_avg_pool2d_wrong_input_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.avg_pool2d(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.avg_pool2d(x1))


def test_avg_pool2d_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 28, 28)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 28, 28), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.avg_pool2d(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.avg_pool2d(x1))


def test_avg_pool3d_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")

    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32, 32), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 32, 32, 32, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=5))
    x3 = relax.Var("x", R.Tensor("float32"))
    x4 = relax.Var("x", R.Tensor(ndim=5))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((2, 4, 32, 32, 32, 16), "float32"))
    x7 = relax.Var("x", R.Tensor((2, 3, 32, 32, 32), "float32", vdev0))

    _check_inference(
        bb, relax.op.nn.avg_pool3d(x0), relax.TensorStructInfo((2, 3, 32, 32, 32), "float32")
    )
    _check_inference(
        bb, relax.op.nn.avg_pool3d(x7), relax.TensorStructInfo((2, 3, 32, 32, 32), "float32", vdev0)
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool3d(x0, pool_size=3),
        relax.TensorStructInfo((2, 3, 30, 30, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool3d(x0, pool_size=(5, 3, 3)),
        relax.TensorStructInfo((2, 3, 28, 30, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool3d(x0, padding=1),
        relax.TensorStructInfo((2, 3, 34, 34, 34), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool3d(x0, padding=[1, 2, 3]),
        relax.TensorStructInfo((2, 3, 34, 36, 38), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool3d(x0, strides=2),
        relax.TensorStructInfo((2, 3, 16, 16, 16), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool3d(x0, dilation=2),
        relax.TensorStructInfo((2, 3, 32, 32, 32), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool3d(x1, layout="NCDHW"),
        relax.TensorStructInfo((2, 32, 32, 32, 3), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool3d(x0, out_layout="NCDHW"),
        relax.TensorStructInfo((2, 3, 32, 32, 32), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool3d(x6, layout="NCDHW16c", out_layout="NDHWC16c"),
        relax.TensorStructInfo((2, 32, 32, 32, 4, 16), "float32"),
    )
    _check_inference(
        bb, relax.op.nn.avg_pool3d(x2), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(
        bb, relax.op.nn.avg_pool3d(x3), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(bb, relax.op.nn.avg_pool3d(x4), relax.TensorStructInfo(dtype="", ndim=5))
    _check_inference(bb, relax.op.nn.avg_pool3d(x5), relax.TensorStructInfo(dtype="", ndim=5))


def test_avg_pool3d_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    c = tir.Var("c", "int64")
    c16 = tir.Var("c16", "int64")
    id_ = tir.Var("id", "int64")
    ih = tir.Var("ih", "int64")
    iw = tir.Var("iw", "int64")
    x0 = relax.Var("x", R.Tensor((n, c, id_, ih, iw), "float32"))
    x1 = relax.Var("x", R.Tensor((n, c, id_, ih, iw, c16), "float32"))

    _check_inference(
        bb,
        relax.op.nn.avg_pool3d(
            x0, pool_size=(3, 3, 3), strides=(3, 3, 3), padding=(2, 2, 2), dilation=(2, 2, 2)
        ),
        relax.TensorStructInfo(
            (
                n,
                c,
                tvm.tir.floordiv(id_ - 1, 3) + 1,
                tvm.tir.floordiv(ih - 1, 3) + 1,
                tvm.tir.floordiv(iw - 1, 3) + 1,
            ),
            "float32",
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool3d(x1, layout="NCDHW16c", out_layout="NDHWC"),
        relax.TensorStructInfo((n, id_, ih, iw, c * 16), "float32"),
    )


def test_avg_pool3d_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=5))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=6))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))

    _check_inference(
        bb, relax.op.nn.avg_pool3d(x0), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool3d(x1, layout="NCDHW16c"),
        relax.TensorStructInfo(dtype="float32", ndim=6),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool3d(x2),
        relax.TensorStructInfo(dtype="float32", ndim=5),
    )


def test_avg_pool3d_infer_struct_info_ceil_mode():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 32, 32, 32), "float32"))

    _check_inference(
        bb,
        relax.op.nn.avg_pool3d(x, pool_size=3, strides=2, ceil_mode=True),
        relax.TensorStructInfo((2, 3, 16, 16, 16), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.avg_pool3d(x, pool_size=(5, 3, 3), strides=2, ceil_mode=True),
        relax.TensorStructInfo((2, 3, 15, 16, 16), "float32"),
    )


def test_avg_pool3d_infer_struct_info_ceil_mode_symbolic():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    c = tir.Var("c", "int64")
    id_ = tir.Var("id", "int64")
    ih = tir.Var("ih", "int64")
    iw = tir.Var("iw", "int64")
    x = relax.Var("x", R.Tensor((n, c, id_, ih, iw), "float32"))

    _check_inference(
        bb,
        relax.op.nn.avg_pool3d(
            x,
            pool_size=(3, 3, 3),
            strides=(2, 2, 2),
            padding=(1, 1, 1),
            dilation=(2, 2, 2),
            ceil_mode=True,
        ),
        relax.TensorStructInfo(
            (
                n,
                c,
                tvm.tir.floordiv(id_, 2),
                tvm.tir.floordiv(ih, 2),
                tvm.tir.floordiv(iw, 2),
            ),
            "float32",
        ),
    )


def test_avg_pool3d_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32, 32), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 32, 32, 32), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3, 32, 32, 32), "int64"))

    _check_inference(
        bb, relax.op.nn.avg_pool3d(x0), relax.TensorStructInfo((2, 3, 32, 32, 32), "float16")
    )
    _check_inference(
        bb, relax.op.nn.avg_pool3d(x1), relax.TensorStructInfo((2, 3, 32, 32, 32), "int8")
    )
    _check_inference(
        bb, relax.op.nn.avg_pool3d(x2), relax.TensorStructInfo((2, 3, 32, 32, 32), "int64")
    )


def test_avg_pool3d_stride_padding_dilation_int64():
    x = relax.Var("x", R.Tensor((2, 3, 28, 28, 28), "float32"))
    avg_pool3d = relax.op.nn.avg_pool3d(
        x, (3, 3, 3), strides=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1)
    )

    assert avg_pool3d.attrs.strides[0].dtype == "int64"
    assert avg_pool3d.attrs.strides[1].dtype == "int64"
    assert avg_pool3d.attrs.strides[2].dtype == "int64"
    assert avg_pool3d.attrs.padding[0].dtype == "int64"
    assert avg_pool3d.attrs.padding[1].dtype == "int64"
    assert avg_pool3d.attrs.padding[2].dtype == "int64"
    assert avg_pool3d.attrs.dilation[0].dtype == "int64"
    assert avg_pool3d.attrs.dilation[1].dtype == "int64"
    assert avg_pool3d.attrs.dilation[2].dtype == "int64"


def test_avg_pool3d_wrong_pool_size_strides_padding_dilation_length():
    x = relax.Var("x", R.Tensor((2, 3, 28, 28, 28), "float32"))
    with pytest.raises(TVMError):
        relax.op.nn.avg_pool3d(x, pool_size=(1, 2, 3, 4))
    with pytest.raises(TVMError):
        relax.op.nn.avg_pool3d(x, strides=(1, 2, 3, 4))
    with pytest.raises(TVMError):
        relax.op.nn.avg_pool3d(x, padding=(1, 2, 3, 4))
    with pytest.raises(TVMError):
        relax.op.nn.avg_pool3d(x, dilation=(1, 2, 3, 4))


def test_avg_pool3d_infer_struct_info_wrong_layout_string():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28, 28, 28), "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.avg_pool3d(x, layout="OIHW"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.avg_pool3d(x, out_layout="OHWI"))


def test_avg_pool3d_wrong_input_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28, 28, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.avg_pool3d(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.avg_pool3d(x1))


def test_avg_pool3d_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 28, 28, 28)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 28, 28, 28), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.avg_pool3d(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.avg_pool3d(x1))


def test_adaptive_avg_pool1d_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")

    x0 = relax.Var("x", R.Tensor((2, 3, 32), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor(ndim=3))
    x4 = relax.Var("x", R.Tensor())

    x5 = relax.Var("x", R.Tensor((2, 3, 32), "float32", vdev0))

    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool1d(x0),
        relax.TensorStructInfo((2, 3, 32), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool1d(x5),
        relax.TensorStructInfo((2, 3, 32), "float32", vdev0),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool1d(x0, output_size=16),
        relax.TensorStructInfo((2, 3, 16), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool1d(x1),
        relax.TensorStructInfo(dtype="float32", ndim=3),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool1d(x2),
        relax.TensorStructInfo(dtype="float32", ndim=3),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool1d(x3),
        relax.TensorStructInfo(dtype="", ndim=3),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool1d(x4),
        relax.TensorStructInfo(dtype="", ndim=3),
    )


def test_adaptive_avg_pool1d_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    c = tir.Var("c", "int64")
    l = tir.Var("l", "int64")

    x0 = relax.Var("x", R.Tensor((n, c, l), "float32"))

    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool1d(x0),
        relax.TensorStructInfo((n, c, l), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool1d(x0, output_size=64),
        relax.TensorStructInfo((n, c, 64), "float32"),
    )


def test_adaptive_avg_pool1d_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=3))
    s1 = relax.Var("s", relax.ShapeStructInfo())

    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))

    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool1d(x0),
        relax.TensorStructInfo(s0, "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool1d(x0, output_size=20),
        relax.TensorStructInfo(dtype="float32", ndim=3),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool1d(x1),
        relax.TensorStructInfo(s1, dtype="float32"),
    )


def test_adaptive_avg_pool1d_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 64), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 64), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3, 64), "int64"))

    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool1d(x0), relax.TensorStructInfo((2, 3, 64), "float16")
    )
    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool1d(x1), relax.TensorStructInfo((2, 3, 64), "int8")
    )
    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool1d(x2), relax.TensorStructInfo((2, 3, 64), "int64")
    )


def test_adaptive_avg_pool1d_wrong_output_size_ndim():
    x = relax.Var("x", R.Tensor((2, 3, 64), "float32"))
    with pytest.raises(TVMError):
        relax.op.nn.adaptive_avg_pool1d(x, output_size=(32, 32))


def test_adaptive_avg_pool1d_infer_struct_info_wrong_layout_string():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 64), "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.adaptive_avg_pool1d(x, layout="OIW"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.adaptive_avg_pool1d(x, out_layout="OWI"))


def test_adaptive_avg_pool1d_wrong_input_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=2))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.adaptive_avg_pool1d(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.adaptive_avg_pool1d(x1))


def test_adaptive_avg_pool1d_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 64)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 64), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.adaptive_avg_pool1d(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.adaptive_avg_pool1d(x1))


def test_adaptive_avg_pool2d_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 32, 32, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=4))
    x3 = relax.Var("x", R.Tensor("float32"))
    x4 = relax.Var("x", R.Tensor(ndim=4))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((2, 4, 32, 32, 16), "float32"))
    x7 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32", vdev0))

    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool2d(x0), relax.TensorStructInfo((2, 3, 32, 32), "float32")
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool2d(x7),
        relax.TensorStructInfo((2, 3, 32, 32), "float32", vdev0),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool2d(x0, output_size=30),
        relax.TensorStructInfo((2, 3, 30, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool2d(x0, output_size=(28, 30)),
        relax.TensorStructInfo((2, 3, 28, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool2d(x1, layout="NHWC"),
        relax.TensorStructInfo((2, 32, 32, 3), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool2d(x0, out_layout="NHWC"),
        relax.TensorStructInfo((2, 32, 32, 3), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool2d(x6, layout="NCHW16c", out_layout="NHWC16c"),
        relax.TensorStructInfo((2, 32, 32, 4, 16), "float32"),
    )
    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool2d(x2), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool2d(x3), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool2d(x4), relax.TensorStructInfo(dtype="", ndim=4)
    )
    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool2d(x5), relax.TensorStructInfo(dtype="", ndim=4)
    )


def test_adaptive_avg_pool2d_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    c = tir.Var("c", "int64")
    c16 = tir.Var("c16", "int64")
    ih = tir.Var("ih", "int64")
    iw = tir.Var("iw", "int64")
    x0 = relax.Var("x", R.Tensor((n, c, ih, iw), "float32"))
    x1 = relax.Var("x", R.Tensor((n, c, ih, iw, c16), "float32"))

    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool2d(x0), relax.TensorStructInfo((n, c, ih, iw), "float32")
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool2d(x0, output_size=256),
        relax.TensorStructInfo((n, c, 256, 256), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool2d(x0, output_size=(256, 128)),
        relax.TensorStructInfo((n, c, 256, 128), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool2d(x1, layout="NCHW16c", out_layout="NHWC"),
        relax.TensorStructInfo((n, ih, iw, c * 16), "float32"),
    )


def test_adaptive_avg_pool2d_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=4))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=5))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))

    _check_inference(bb, relax.op.nn.adaptive_avg_pool2d(x0), relax.TensorStructInfo(s0, "float32"))
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool2d(x0, output_size=32),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool2d(x1, layout="NCHW16c"),
        relax.TensorStructInfo(s1, "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool2d(x0, out_layout="NCHW16c"),
        relax.TensorStructInfo(dtype="float32", ndim=5),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool2d(x2, out_layout="NCHW16c"),
        relax.TensorStructInfo(dtype="float32", ndim=5),
    )


def test_adaptive_avg_pool2d_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 32, 32), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3, 32, 32), "int64"))
    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool2d(x0), relax.TensorStructInfo((2, 3, 32, 32), "float16")
    )
    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool2d(x1), relax.TensorStructInfo((2, 3, 32, 32), "int8")
    )
    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool2d(x2), relax.TensorStructInfo((2, 3, 32, 32), "int64")
    )


def test_adaptive_avg_pool2d_wrong_output_size_ndim():
    x = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32"))
    with pytest.raises(TVMError):
        relax.op.nn.adaptive_avg_pool2d(x, (32, 32, 32))


def test_adaptive_avg_pool2d_infer_struct_info_wrong_layout_string():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.adaptive_avg_pool2d(x, layout="OIHW"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.adaptive_avg_pool2d(x, out_layout="OHWI"))


def test_adaptive_avg_pool2d_wrong_input_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.adaptive_avg_pool2d(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.adaptive_avg_pool2d(x1))


def test_adaptive_avg_pool2d_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 28, 28)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 28, 28), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.adaptive_avg_pool2d(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.adaptive_avg_pool2d(x1))


def test_adaptive_avg_pool3d_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")

    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32, 32), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 32, 32, 32, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=5))
    x3 = relax.Var("x", R.Tensor("float32"))
    x4 = relax.Var("x", R.Tensor(ndim=5))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((2, 4, 32, 32, 32, 16), "float32"))
    x7 = relax.Var("x", R.Tensor((2, 3, 32, 32, 32), "float32", vdev0))

    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool3d(x0),
        relax.TensorStructInfo((2, 3, 32, 32, 32), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool3d(x7),
        relax.TensorStructInfo((2, 3, 32, 32, 32), "float32", vdev0),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool3d(x0, output_size=30),
        relax.TensorStructInfo((2, 3, 30, 30, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool3d(x0, output_size=(28, 30, 32)),
        relax.TensorStructInfo((2, 3, 28, 30, 32), "float32"),
    )

    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool3d(x1, layout="NCDHW"),
        relax.TensorStructInfo((2, 32, 32, 32, 3), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool3d(x0, out_layout="NCDHW"),
        relax.TensorStructInfo((2, 3, 32, 32, 32), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool3d(x6, layout="NCDHW16c", out_layout="NDHWC16c"),
        relax.TensorStructInfo((2, 32, 32, 32, 4, 16), "float32"),
    )

    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool3d(x2), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool3d(x3), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool3d(x4), relax.TensorStructInfo(dtype="", ndim=5)
    )
    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool3d(x5), relax.TensorStructInfo(dtype="", ndim=5)
    )


def test_adaptive_avg_pool3d_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()

    n = tir.Var("n", "int64")
    c = tir.Var("c", "int64")
    c16 = tir.Var("c16", "int64")
    d = tir.Var("d", "int64")
    ih = tir.Var("ih", "int64")
    iw = tir.Var("iw", "int64")

    x0 = relax.Var("x", R.Tensor((n, c, d, ih, iw), "float32"))
    x1 = relax.Var("x", R.Tensor((n, c, d, ih, iw, c16), "float32"))

    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool3d(x0),
        relax.TensorStructInfo((n, c, d, ih, iw), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool3d(x0, output_size=256),
        relax.TensorStructInfo((n, c, 256, 256, 256), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool3d(x0, output_size=(256, 128, 64)),
        relax.TensorStructInfo((n, c, 256, 128, 64), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool3d(x1, layout="NCDHW16c", out_layout="NDHWC"),
        relax.TensorStructInfo((n, d, ih, iw, c * 16), "float32"),
    )


def test_adaptive_avg_pool3d_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()

    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=5))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=6))
    s2 = relax.Var("s", relax.ShapeStructInfo())

    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))

    _check_inference(bb, relax.op.nn.adaptive_avg_pool3d(x0), relax.TensorStructInfo(s0, "float32"))
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool3d(x0, output_size=32),
        relax.TensorStructInfo(dtype="float32", ndim=5),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool3d(x1, layout="NCDHW16c"),
        relax.TensorStructInfo(s1, "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool3d(x0, out_layout="NCDHW16c"),
        relax.TensorStructInfo(dtype="float32", ndim=6),
    )
    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool3d(x2, out_layout="NCDHW16c"),
        relax.TensorStructInfo(dtype="float32", ndim=6),
    )


def test_adaptive_avg_pool3d_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32, 32), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 32, 32, 32), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3, 32, 32, 32), "int64"))

    _check_inference(
        bb,
        relax.op.nn.adaptive_avg_pool3d(x0),
        relax.TensorStructInfo((2, 3, 32, 32, 32), "float16"),
    )
    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool3d(x1), relax.TensorStructInfo((2, 3, 32, 32, 32), "int8")
    )
    _check_inference(
        bb, relax.op.nn.adaptive_avg_pool3d(x2), relax.TensorStructInfo((2, 3, 32, 32, 32), "int64")
    )


def test_adaptive_avg_pool3d_wrong_output_size_ndim():
    x = relax.Var("x", R.Tensor((2, 3, 32, 32, 32), "float32"))

    with pytest.raises(TVMError):
        relax.op.nn.adaptive_avg_pool3d(x, (32, 32, 32, 32))


def test_adaptive_avg_pool3d_infer_struct_info_wrong_layout_string():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28, 28, 28), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.adaptive_avg_pool3d(x, layout="OIDHW"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.adaptive_avg_pool3d(x, out_layout="OHIDW"))


def test_adaptive_avg_pool3d_wrong_input_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28, 28, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.adaptive_avg_pool3d(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.adaptive_avg_pool3d(x1))


def test_adaptive_avg_pool3d_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 28, 28, 28)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 28, 28, 28), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.adaptive_avg_pool3d(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.adaptive_avg_pool3d(x1))


if __name__ == "__main__":
    tvm.testing.main()
