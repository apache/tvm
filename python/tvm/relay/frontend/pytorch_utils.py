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
# pylint: disable=import-outside-toplevel, unused-argument, invalid-name
""" Common utilities used by PyTorch frontend """
from .. import op
from ..dataflow_pattern import (
    is_constant,
    is_op,
    rewrite,
    is_tuple,
    wildcard,
    DFPatternCallback,
)


def is_version_greater_than(ver):
    import torch
    import re

    return "".join(re.findall(r"(\d+\.)(\d+\.)(\d)", torch.__version__)[0]) > "".join(
        re.findall(r"(\d+\.)(\d+\.)(\d)", ver)[0]
    )


def batched_nms_pattern(boxes, scores, idxs, iou_threshold, num_boxes, indices):
    """A pattern to detect batched_nms function in torchvision

    The inputs to this function, boxes, scores, idxs, iou_threshold are wildcard
    patterns which can be used later in the rewriting to extract matched Relay fragments.

    We want to detect the following PyTorch code snippet:

    def batched_nms(boxes, scores, idxs, iou_threshold):
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]
        keep = nms(boxes_for_nms, scores, iou_threshold)
        return keep

    Here is how PyTorch frontend lowers above PyTorch code. For simplicity, Relay ops for
    dealing with dynamic strided_slice are omitted. %num_boxes, %indices are complex
    expressions, but since we can use the wildcard part for them, we do not need to construct
    their patterns.

    %2 = expand_dims(%scores, axis=-1);
    %3 = cast(%idxs, dtype="float32");
    %4 = max(%boxes);
    %5 = add(%4, 1f);
    %6 = multiply(%3, %5);
    %7 = strided_slice(%6, begin=[0], end=[4507], strides=[1]);
    %8 = expand_dims(%7, axis=1);
    %9 = add(%boxes, %8);
    %10 = (%2, %9);
    %11 = concatenate(%10, axis=-1);
    %12 = expand_dims(%11, axis=0);
    ...
    ...
    %17 = vision.non_max_suppression(%12, %num_boxes, %indices, -1, 0.7f, ...);

    """
    one = is_constant()
    zero = is_constant()

    # Equivelent PyTorch code from above snippet
    # offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    cast = is_op("cast")(idxs)
    mx = is_op("max")(boxes)
    add = is_op("add")(mx, one)
    mul = is_op("multiply")(cast, add)

    # The following doesn't appear in the above Relay snippet. It is required for dynamic
    # stride_slice handling
    cast_like = is_op("cast_like")(zero, is_constant())
    less = is_op("less")(is_constant(), cast_like)
    shape_of = is_op("shape_of")(mul)
    cast_like = is_op("cast_like")(shape_of, is_constant())
    add = is_op("add")(is_constant(), cast_like)
    where = is_op("where")(less, add, is_constant())
    shape_of = is_op("shape_of")(mul)
    cast = is_op("cast")(shape_of)

    # This corresponds to offsets[:, None], where offsets is the result of multiplication
    dyn_strided_slice = is_op("dyn.strided_slice")(mul, where, cast, is_constant())

    # Add offsets to the boxes
    expand_dims = is_op("expand_dims")(dyn_strided_slice)
    add = is_op("add")(boxes, expand_dims)

    # The rest of patterns correspond to the PyTorch frontend conversion
    # function for torchvision::nms
    score_expand_dims = is_op("expand_dims")(scores)
    tup = is_tuple([score_expand_dims, add])
    concat = is_op("concatenate")(tup)
    data = is_op("expand_dims")(concat)

    return is_op("vision.non_max_suppression")(
        data, num_boxes, indices, is_constant(), iou_threshold
    )


class NMSRewrite(DFPatternCallback):
    """A callback to rewrite nms and restore batched nms"""

    def __init__(self):
        super().__init__()
        # exprs to extract
        self.boxes = wildcard()
        self.scores = wildcard()
        self.idxs = wildcard()
        self.iou_threshold = wildcard()
        self.num_boxes = wildcard()
        self.indices = wildcard()

        self.pattern = batched_nms_pattern(
            self.boxes,
            self.scores,
            self.idxs,
            self.iou_threshold,
            self.num_boxes,
            self.indices,
        )

    def convert_batched_nms(self, boxes, scores, idxs, iou_thres, num_boxes, indices):
        """Restore class-aware NMS using extracted class indices"""
        scores = op.expand_dims(scores, axis=-1, num_newaxis=1)
        idxs = op.expand_dims(idxs, axis=-1, num_newaxis=1)
        idxs = op.cast(idxs, "float32")
        data = op.concatenate([idxs, scores, boxes], -1)
        data = op.expand_dims(data, 0, 1)

        top_k = max_out_size = -1
        out = op.vision.non_max_suppression(
            data=data,
            valid_count=num_boxes,
            indices=indices,
            max_output_size=max_out_size,
            iou_threshold=iou_thres,
            force_suppress=False,
            top_k=top_k,
            coord_start=2,
            score_index=1,
            id_index=0,
            return_indices=True,
            invalid_to_bottom=False,
        )
        return out.tuple_value

    def callback(self, pre, post, node_map):
        boxes = node_map[self.boxes][0]
        scores = node_map[self.scores][0]
        idxs = node_map[self.idxs][0]
        iou_thres = node_map[self.iou_threshold][0]
        num_boxes = node_map[self.num_boxes][0]
        indices = node_map[self.indices][0]
        return self.convert_batched_nms(boxes, scores, idxs, iou_thres, num_boxes, indices)


def rewrite_nms_to_batched_nms(mod):
    """Rewrite the input graph to replace non maximum surpression
    in torchvision that does not take class id into account with the one
    that avoids IOU tests between different classes.
    """
    mod["main"] = rewrite(NMSRewrite(), mod["main"])
    return mod
