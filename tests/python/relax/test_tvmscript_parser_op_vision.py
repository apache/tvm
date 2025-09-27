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

from typing import Optional, Union

import tvm
import tvm.script
import tvm.testing
from tvm import IRModule, relax
from tvm.script import relax as R


def _check(
    parsed: Union[relax.Function, IRModule],
    expect: Optional[Union[relax.Function, IRModule]],
):
    test = parsed.script(show_meta=True)
    roundtrip_mod = tvm.script.from_source(test)
    tvm.ir.assert_structural_equal(parsed, roundtrip_mod)
    if expect:
        tvm.ir.assert_structural_equal(parsed, expect)


def test_all_class_non_max_suppression():
    @R.function
    def foo(
        boxes: R.Tensor((10, 5, 4), "float32"),
        scores: R.Tensor((10, 8, 5), "float32"),
        max_output_boxes_per_class: R.Tensor((), "int64"),
        iou_threshold: R.Tensor((), "float32"),
        score_threshold: R.Tensor((), "float32"),
    ) -> R.Tuple(R.Tensor((400, 3), "int64"), R.Tensor((1,), "int64")):
        gv: R.Tuple(
            R.Tensor((400, 3), "int64"), R.Tensor((1,), "int64")
        ) = R.vision.all_class_non_max_suppression(
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            "onnx",
        )
        return gv

    boxes = relax.Var("boxes", R.Tensor((10, 5, 4), "float32"))
    scores = relax.Var("scores", R.Tensor((10, 8, 5), "float32"))
    max_output_boxes_per_class = relax.Var("max_output_boxes_per_class", R.Tensor((), "int64"))
    iou_threshold = relax.Var("iou_threshold", R.Tensor((), "float32"))
    score_threshold = relax.Var("score_threshold", R.Tensor((), "float32"))

    bb = relax.BlockBuilder()
    with bb.function(
        "foo", [boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold]
    ):
        gv = bb.emit(
            relax.op.vision.all_class_non_max_suppression(
                boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, "onnx"
            )
        )
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


if __name__ == "__main__":
    tvm.testing.main()
