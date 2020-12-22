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
    is_tuple_get_item,
    wildcard,
    DFPatternCallback,
)


def is_version_greater_than(ver):
    import torch
    import re

    return "".join(re.findall(r"(\d+\.)(\d+\.)(\d)", torch.__version__)[0]) > "".join(
        re.findall(r"(\d+\.)(\d+\.)(\d)", ver)[0]
    )


def batched_nms_pattern(boxes, scores, idxs, iou_threshold):
    """A pattern to detect batched_nms function in torchvision"""
    one = is_constant()
    zero = is_constant()

    score_expand_dims = is_op("expand_dims")(scores)

    cast = is_op("cast")(idxs)
    mx = is_op("max")(boxes)
    add = is_op("add")(mx, one)
    mul = is_op("multiply")(cast, add)

    cast_like = is_op("cast_like")(zero, is_constant())
    less = is_op("less")(is_constant(), cast_like)
    shape_of = is_op("shape_of")(mul)
    cast_like = is_op("cast_like")(shape_of, is_constant())
    add = is_op("add")(is_constant(), cast_like)
    where = is_op("where")(less, add, is_constant())
    shape_of = is_op("shape_of")(mul)
    cast = is_op("cast")(shape_of)

    dyn_strided_slice = is_op("dyn.strided_slice")(mul, where, cast, is_constant())

    expand_dims = is_op("expand_dims")(dyn_strided_slice)
    add = is_op("add")(boxes, expand_dims)
    tup = is_tuple([score_expand_dims, add])
    concat = is_op("concatenate")(tup)
    expand_dims = is_op("expand_dims")(concat)

    get_valid_counts_out = is_op("vision.get_valid_counts")(expand_dims, is_constant())
    data = is_tuple_get_item(get_valid_counts_out, 1)
    valid_counts = is_tuple_get_item(get_valid_counts_out, 0)
    indices = is_tuple_get_item(get_valid_counts_out, 2)

    return is_op("vision.non_max_suppression")(
        data, valid_counts, indices, is_constant(), iou_threshold
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
        self.pattern = batched_nms_pattern(self.boxes, self.scores, self.idxs, self.iou_threshold)

    def convert_batched_nms(self, boxes, scores, idxs, iou_thres):
        """Restore class-aware NMS using extracted class indices"""
        scores = op.expand_dims(scores, axis=-1, num_newaxis=1)
        idxs = op.expand_dims(idxs, axis=-1, num_newaxis=1)
        idxs = op.cast(idxs, "float32")
        data = op.concatenate([idxs, scores, boxes], -1)
        data = op.expand_dims(data, 0, 1)
        ct, data, indices = op.vision.get_valid_counts(
            data, score_threshold=-1.0, id_index=0, score_index=1
        )
        top_k = max_out_size = -1
        out = op.vision.non_max_suppression(
            data=data,
            valid_count=ct,
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
        return self.convert_batched_nms(boxes, scores, idxs, iou_thres)


def rewrite_nms_to_batched_nms(mod):
    """Rewrite the input graph to replace non maximum surpression
    in torchvision that does not take class id into account with the one
    that avoids IOU tests between different classes.
    """
    mod["main"] = rewrite(NMSRewrite(), mod["main"])
    return mod
