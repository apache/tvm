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


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_sinfo: relax.StructInfo):
    ret = bb.normalize(call)
    tvm.ir.assert_structural_equal(ret.struct_info, expected_sinfo)


def test_all_class_non_max_suppression_infer_struct_info():
    bb = relax.BlockBuilder()
    batch_size, num_classes, num_boxes = 10, 8, 5
    boxes = relax.Var("boxes", R.Tensor((batch_size, num_boxes, 4), "float32"))
    scores = relax.Var("scores", R.Tensor((batch_size, num_classes, num_boxes), "float32"))
    max_output_boxes_per_class = relax.const(10, "int64")
    iou_threshold = relax.const(0.5, "float32")
    score_threshold = relax.const(0.1, "float32")

    _check_inference(
        bb,
        relax.op.vision.all_class_non_max_suppression(
            boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, "onnx"
        ),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((batch_size * num_classes * num_boxes, 3), "int64"),
                relax.TensorStructInfo((1,), "int64"),
            ]
        ),
    )


def test_all_class_non_max_suppression_wrong_input_number():
    bb = relax.BlockBuilder()
    boxes = relax.Var("boxes", R.Tensor((1, 5, 4), "float32"))
    scores = relax.Var("scores", R.Tensor((1, 3, 5), "float32"))

    with pytest.raises(TVMError):
        relax.op.vision.all_class_non_max_suppression(boxes, scores)


def test_all_class_non_max_suppression_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    batch_size = tir.Var("batch_size", "int64")
    num_classes = tir.Var("num_classes", "int64")
    num_boxes = tir.Var("num_boxes", "int64")
    boxes = relax.Var("boxes", R.Tensor((batch_size, num_boxes, 4), "float32"))
    scores = relax.Var("scores", R.Tensor((batch_size, num_classes, num_boxes), "float32"))
    max_output_boxes_per_class = relax.const(10, "int64")
    iou_threshold = relax.const(0.5, "float32")
    score_threshold = relax.const(0.1, "float32")

    _check_inference(
        bb,
        relax.op.vision.all_class_non_max_suppression(
            boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, "onnx"
        ),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((batch_size * num_classes * num_boxes, 3), "int64"),
                relax.TensorStructInfo((1,), "int64"),
            ]
        ),
    )


if __name__ == "__main__":
    tvm.testing.main()
