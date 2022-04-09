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
from .. import expr
from .. import op
from ..dataflow_pattern import (
    wildcard,
    is_constant,
    is_op,
    rewrite,
    is_tuple,
    is_tuple_get_item,
    is_if,
    DFPatternCallback,
)


def is_version_greater_than(ver):
    import torch
    from distutils.version import LooseVersion

    return LooseVersion(torch.__version__) > ver


def getattr_attr_name(node):
    attribute_names = node.attributeNames()
    assert len(attribute_names) == 1
    return node.s(attribute_names[0])


def dyn_strided_slice_pattern(inp, end):
    """A pattern to detect dynamic strided slice op."""
    zero = is_constant()
    cast_like = is_op("cast_like")(zero, is_constant())
    less = is_op("less")(is_constant(), cast_like)
    shape_of = is_op("shape_of")(inp)
    cast_like = is_op("cast_like")(shape_of, is_constant())
    add = is_op("add")(is_constant(), cast_like)
    where = is_op("where")(less, add, is_constant())

    return is_op("dyn.strided_slice")(inp, where, end, is_constant())


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

    # Equivalent PyTorch code from above snippet
    # offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    cast = is_op("cast")(idxs)
    mx = is_op("max")(boxes)
    add = is_op("add")(mx, one)
    mul = is_op("multiply")(cast, add)

    shape_of = is_op("shape_of")(mul)
    cast = is_op("cast")(shape_of)

    # Add offsets to the boxes
    expand_dims = is_op("expand_dims")(mul)
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


def topk_after_batch_nms_pattern(cond, true_branch, data, valid_count, indices, iou_threshold):
    """
    Detect the following pattern used in torchvision detection models.

    def batched_nms(...):
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        else:
            ...
            return nms(boxes_for_nms, scores, iou_threshold)

    keep = batched_nms(boxes, scores, lvl, self.nms_thresh)
    keep = keep[:post_nms_top_k] # keep only topk scoring predictions

    An equivalent Relay subgraph:

    %1184 = if (%1117) {
      ...
    } else {
      ...
      %1172 = vision.non_max_suppression(%1167, %1168, %1171, -1, 0.7f, ...);
      ...
      %1183 = dyn.strided_slice(%1174, %1180, %1182, ...);
      cast(%1183, dtype="int64")
    };
    %1185 = strided_slice(%1184, begin=[0], end=[1000], strides=[1]);

    """
    nms = is_op("vision.non_max_suppression")(
        data, valid_count, indices, is_constant(), iou_threshold
    )
    indices = is_op("squeeze")(is_tuple_get_item(nms, 0))
    size = is_op("squeeze")(is_tuple_get_item(nms, 1))
    dyn_strided_slice = dyn_strided_slice_pattern(indices, size)
    cast_i64 = is_op("cast")(dyn_strided_slice)

    batched_nms_result = is_if(cond, true_branch, cast_i64)

    return is_op("strided_slice")(batched_nms_result)


class MulticlassNMSRewrite(DFPatternCallback):
    """A callback to rewrite nms and restore batched nms."""

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


class PostNMSTopKRewrite(DFPatternCallback):
    """A callback to rewrite nms to exploit max_out_size parameter."""

    def __init__(self):
        super().__init__()
        self.cond = wildcard()
        self.true_branch = wildcard()
        self.data = wildcard()
        self.valid_count = wildcard()
        self.indices = wildcard()
        self.iou_threshold = wildcard()

        self.pattern = topk_after_batch_nms_pattern(
            self.cond,
            self.true_branch,
            self.data,
            self.valid_count,
            self.indices,
            self.iou_threshold,
        )

    def rewrite_batch_nms_with_max_out_size(
        self, cond, true_branch, data, valid_count, indices, iou_threshold, post_nms_topk
    ):
        """Use the detected post NMS topk parameter in NMS op."""
        nms_ret = op.vision.non_max_suppression(
            data=data,
            valid_count=valid_count,
            indices=indices,
            max_output_size=post_nms_topk,
            iou_threshold=iou_threshold,
            force_suppress=False,
            top_k=-1,
            coord_start=2,
            score_index=1,
            id_index=0,
            return_indices=True,
            invalid_to_bottom=False,
        )

        size = op.squeeze(nms_ret[1], axis=[1])
        data_slice = op.squeeze(nms_ret[0], axis=[0])

        ret = op.strided_slice(data_slice, begin=expr.const([0]), end=size, slice_mode="size")

        nms_result = op.cast(ret, "int64")

        return expr.If(cond, true_branch, nms_result)

    def callback(self, pre, post, node_map):
        post_nms_topk = post.attrs.end[0].value
        return self.rewrite_batch_nms_with_max_out_size(
            node_map[self.cond][0],
            node_map[self.true_branch][0],
            node_map[self.data][0],
            node_map[self.valid_count][0],
            node_map[self.indices][0],
            node_map[self.iou_threshold][0],
            post_nms_topk,
        )


def scatter_roi_align_result_pattern(levels, roi_align_results, num_scales):
    """Detect the Relay subgraph corresponding to the following PyTorch code

    first_result = roi_align_results[0]
    dtype, device = first_result.dtype, first_result.device
    res = torch.zeros((levels.size(0), first_result.size(1),
                       first_result.size(2), first_result.size(3)),
                      dtype=dtype, device=device)
    for level in range(len(roi_align_results)):
        index = torch.where(levels == level)[0].view(-1, 1, 1, 1)
        index = index.expand(index.size(0),
                             roi_align_results[level].size(1),
                             roi_align_results[level].size(2),
                             roi_align_results[level].size(3))
        res = res.scatter(0, index, roi_align_results[level])
    return res
    """

    def do_where(levels, _):
        idx_in_level = is_op("argwhere")(is_op("equal")(levels, is_constant()))
        idx_in_level = is_op("split")(idx_in_level)
        idx_in_level = is_tuple_get_item(idx_in_level, 0)
        idx_in_level = is_op("squeeze")(idx_in_level)
        idx_in_level = is_tuple_get_item(is_tuple([idx_in_level]), 0)
        return idx_in_level

    scatter_res = wildcard()

    for i in range(num_scales):
        # index = torch.where(levels == level)[0].view(-1, 1, 1, 1)
        scatter_indices = do_where(levels, i)
        scatter_indices = is_op("reshape")(scatter_indices)

        # index = index.expand(index.size(0),
        #                      unmerged_results[level].size(1),
        #                      unmerged_results[level].size(2),
        #                      unmerged_results[level].size(3))
        scatter_indices = is_op("repeat")(scatter_indices)
        scatter_indices = is_op("repeat")(scatter_indices)
        scatter_indices = is_op("repeat")(scatter_indices)

        scatter_res = is_op("scatter")(scatter_res, scatter_indices, roi_align_results[i])

    return is_op("reshape")(scatter_res)


class ScatterRewrite(DFPatternCallback):
    """A callback to rewrite repeated scatters with a batched gather."""

    def __init__(self, num_scales):
        super().__init__()
        self.num_scales = num_scales
        self.levels = wildcard()
        self.roi_align_results = []
        for _ in range(num_scales):
            self.roi_align_results.append(wildcard())

        self.pattern = scatter_roi_align_result_pattern(
            self.levels, self.roi_align_results, num_scales
        )

    def convert_scatter_to_gather(self, levels, roi_align_results):
        """Replace the detected scatter loop with the following PyTorch code

        indices_per_level = []
        for level in range(num_scales):
            idx_in_level = torch.where(levels == level)[0]
            indices_per_leve.append(idx_in_level)

        stacked_features = torch.cat(roi_align_results, dim=0)
        stacked_indices = torch.cat(indices_per_level, dim=0)
        argsort_indices = torch.argort(stacked_indices)
        return stacked_features[argsort_indices, :]
        """

        # Collect inidices and concat them
        indices_per_level = []
        for i in range(self.num_scales):
            equal = op.equal(levels, expr.const(i, dtype="int64"))
            argwhere = op.argwhere(equal)
            split = op.split(argwhere, indices_or_sections=1, axis=1)
            squeeze = op.squeeze(split[0], axis=[1])
            indices = op.cast(squeeze, dtype="int64")
            indices_per_level.append(indices)

        indices_concat = op.concatenate(indices_per_level, 0)

        # Concat roi align results per level, and argsort indices
        # To prepare for a batched gather
        roi_align_results_concat = op.concatenate(roi_align_results, 0)
        argsort_indices = op.cast(op.argsort(indices_concat), dtype="int64")

        # Permute rows by argsorted indices
        permuted = op.take(roi_align_results_concat, argsort_indices, axis=0)

        return op.reshape(permuted, [0, -1, 1, 1])

    def callback(self, pre, post, node_map):
        levels = node_map[self.levels][0]
        roi_align_results = [node_map[feat][0] for feat in self.roi_align_results]
        return self.convert_scatter_to_gather(levels, roi_align_results)


def rewrite_nms_to_batched_nms(mod):
    """Rewrite the input graph to replace non maximum surpression
    in torchvision that does not take class id into account with the one
    that avoids IOU tests between different classes.
    """
    mod["main"] = rewrite(MulticlassNMSRewrite(), mod["main"])
    return mod


def rewrite_batched_nms_with_max_out_size(mod):
    """Rewrite the input graph to detect slicing after batched nms and
    use the slicing size as the parameter max_out_size in NMS.
    """
    mod["main"] = rewrite(PostNMSTopKRewrite(), mod["main"])
    return mod


def rewrite_scatter_to_gather(mod, num_scales):
    """Rewrite the input graph to replace a repeated scatter loop with
    a batched gather. The scatter loop is used in torchvision MultiScaleRoIAlign
    to merge roi_align results for all scales. The scatter is used to emulate
    inplace updates.
    """
    mod["main"] = rewrite(ScatterRewrite(num_scales), mod["main"])
    return mod
