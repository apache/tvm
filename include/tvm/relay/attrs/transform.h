/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/relay/attrs/transform.h
 * \brief Transform operators.
 */
#ifndef TVM_RELAY_ATTRS_TRANSFORM_H_
#define TVM_RELAY_ATTRS_TRANSFORM_H_

#include <tvm/ir/attrs.h>
#include <tvm/relay/base.h>
#include <tvm/relay/expr.h>
#include <tvm/tir/index_map.h>

#include <string>

namespace tvm {
namespace relay {

/*! \brief Attributes used for the sliding_window operator */
struct SlidingWindowAttrs : public tvm::AttrsNode<SlidingWindowAttrs> {
  int axis;
  Array<Integer> window_shape;
  Array<Integer> strides;
  TVM_DECLARE_ATTRS(SlidingWindowAttrs, "relay.attrs.SlidingWindowAttrs") {
    TVM_ATTR_FIELD(axis).describe(
        "What axis the sliding window begin forming over."
        "Window will be slid over this axis and all following axes."
        "The axis value determines the window shape (and thus, the"
        "number of strides):"
        "window shape and strides must both be of length"
        "`data.ndim-axis`.");
    TVM_ATTR_FIELD(window_shape)
        .describe(
            "The window shape to form over the input."
            "Window shape must be of length `data.ndim-axis`.");
    TVM_ATTR_FIELD(strides).describe(
        "How to stride the window along each dimension."
        "Strides must be of length `data.ndim-axis`.");
  }
};  // struct SlidingWindowAttrs

/*! \brief data type cast */
struct CastAttrs : public tvm::AttrsNode<CastAttrs> {
  DataType dtype;

  TVM_DECLARE_ATTRS(CastAttrs, "relay.attrs.CastAttrs") {
    TVM_ATTR_FIELD(dtype).describe("Target data type");
  }
};  // struct CastAttrs.

/*! \brief Attributes used in expand_dims operators */
struct ExpandDimsAttrs : public tvm::AttrsNode<ExpandDimsAttrs> {
  int axis;
  int num_newaxis;

  TVM_DECLARE_ATTRS(ExpandDimsAttrs, "relay.attrs.ExpandDimsAttrs") {
    TVM_ATTR_FIELD(axis).describe(
        "The axis at which the input array is expanded."
        "Should lie in range `[-data.ndim - 1, data.ndim]`."
        "If `axis < 0`, it is the first axis inserted;"
        "If `axis >= 0`, it is the last axis inserted in Python's negative indexing.");
    TVM_ATTR_FIELD(num_newaxis)
        .describe("Number of axes to be inserted. Should be >= 0.")
        .set_lower_bound(0)
        .set_default(1);
  }
};  // struct ExpandDimsAttrs

/*! \brief Attributes used in dynamic expand_dims operators */
struct DynExpandDimsAttrs : public tvm::AttrsNode<DynExpandDimsAttrs> {
  int num_newaxis;

  TVM_DECLARE_ATTRS(DynExpandDimsAttrs, "relay.attrs.DynExpandDimsAttrs") {
    TVM_ATTR_FIELD(num_newaxis)
        .describe("Number of axes to be inserted. Should be >= 0.")
        .set_lower_bound(0)
        .set_default(1);
  }
};  // struct ExpandDimsAttrs

/*! \brief Attributes used in concatenate operators */
struct ConcatenateAttrs : public tvm::AttrsNode<ConcatenateAttrs> {
  int axis;
  TVM_DECLARE_ATTRS(ConcatenateAttrs, "relay.attrs.ConcatenateAttrs") {
    TVM_ATTR_FIELD(axis)
        .describe(
            "The axis at which the input arrays are concatenated."
            "Should lie in range `[-ndim, ndim)`.")
        .set_default(0);
  }
};  // struct ConcatenateAttrs

/*! \brief Attributes used in transpose operators */
struct TransposeAttrs : public tvm::AttrsNode<TransposeAttrs> {
  Array<Integer> axes;
  TVM_DECLARE_ATTRS(TransposeAttrs, "relay.attrs.TransposeAttrs") {
    TVM_ATTR_FIELD(axes).describe("The target axes order, reverse order if not specified.");
  }
};  // struct TransposeAttrs

/*! \brief Attributes used in reshape operators */
struct ReshapeAttrs : public tvm::AttrsNode<ReshapeAttrs> {
  Array<Integer> newshape;
  bool allowzero;
  TVM_DECLARE_ATTRS(ReshapeAttrs, "relay.attrs.ReshapeAttrs") {
    TVM_ATTR_FIELD(newshape).describe(
        "The new shape. Should be compatible with the original shape.");
    TVM_ATTR_FIELD(allowzero).set_default(0).describe(
        "Whether to honor the value of zero in newshape.");
  }
};  // struct ReshapeAttrs

/*! \brief Attributes used in MXNet-style reshape_like operators */
struct ReshapeLikeAttrs : public tvm::AttrsNode<ReshapeLikeAttrs> {
  int lhs_begin;
  Integer lhs_end;  // can be None
  int rhs_begin;
  Integer rhs_end;  // can be None
  TVM_DECLARE_ATTRS(ReshapeLikeAttrs, "relay.attrs.ReshapeLikeAttrs") {
    TVM_ATTR_FIELD(lhs_begin).set_default(0).describe(
        "The axis of the input where reshaping should begin.");
    TVM_ATTR_FIELD(lhs_end)
        .set_default(NullValue<Integer>())
        .describe("The axis of the input where reshaping should end, exclusive.");
    TVM_ATTR_FIELD(rhs_begin).set_default(0).describe(
        "The axis of the shape_like tensor to begin taking dimensions from.");
    TVM_ATTR_FIELD(rhs_end)
        .set_default(NullValue<Integer>())
        .describe("The axis of the shape_like tensor to end taking dimensions from, exclusive.");
  }
};  // struct ReshapeLikeAttrs

struct ScatterElementsAttrs : public tvm::AttrsNode<ScatterElementsAttrs> {
  Integer axis;
  String reduction;

  TVM_DECLARE_ATTRS(ScatterElementsAttrs, "relay.attrs.ScatterElementsAttrs") {
    TVM_ATTR_FIELD(axis).set_default(0).describe("The axis over which to select values.");
    TVM_ATTR_FIELD(reduction).set_default("update").describe(
        "Reduction mode of the scatter elements, "
        "either \"update\", \"add\", \"mul\", \"mean\", \"min\" or \"max\".");
  }
};

struct ScatterNDAttrs : public tvm::AttrsNode<ScatterNDAttrs> {
  String mode;

  TVM_DECLARE_ATTRS(ScatterNDAttrs, "relay.attrs.ScatterNDAttrs") {
    TVM_ATTR_FIELD(mode).set_default("update").describe(
        "Accumulation mode of the ScatterND, "
        "either \"update\", \"add\", \"mul\", \"min\" or \"max\".");
  }
};

struct GatherAttrs : public tvm::AttrsNode<GatherAttrs> {
  Integer axis;

  TVM_DECLARE_ATTRS(GatherAttrs, "relay.attrs.GatherAttrs") {
    TVM_ATTR_FIELD(axis)
        .set_default(NullValue<Integer>())
        .describe("The axis over which to select values.");
  }
};

struct GatherNDAttrs : public tvm::AttrsNode<GatherNDAttrs> {
  Integer batch_dims;
  Optional<Integer> index_rank;

  TVM_DECLARE_ATTRS(GatherNDAttrs, "relay.attrs.GatherNDAttrs") {
    TVM_ATTR_FIELD(batch_dims).set_default(Integer(0)).describe("The number of batch dimensions.");
    TVM_ATTR_FIELD(index_rank)
        .set_default(NullValue<Integer>())
        .describe(
            "The size of an indexing tuple, which is a fixed value. Only needed when the number of "
            "indexting tuples is dynamic.");
  }
};

struct TakeAttrs : public tvm::AttrsNode<TakeAttrs> {
  Integer batch_dims;
  Integer axis;
  tvm::String mode;

  TVM_DECLARE_ATTRS(TakeAttrs, "relay.attrs.TakeAttrs") {
    TVM_ATTR_FIELD(batch_dims)
        .set_default(0)
        .describe("The batch_dims over which to select values.");
    TVM_ATTR_FIELD(axis)
        .set_default(NullValue<Integer>())
        .describe("The axis over which to select values.");
    TVM_ATTR_FIELD(mode).set_default("clip").describe(
        "Specify how out-of-bound indices will behave."
        "clip - clip to the range (default)"
        "wrap - wrap around the indices"
        "fast - no clip or wrap around (user must make sure indices are in-bound)");
  }
};

/*! \brief Attributes that specify a tensor */
struct InitOpAttrs : public tvm::AttrsNode<InitOpAttrs> {
  Optional<Array<Integer>> shape;
  DataType dtype;

  TVM_DECLARE_ATTRS(InitOpAttrs, "relay.attrs.InitOpAttrs") {
    TVM_ATTR_FIELD(shape).describe("Target shape.");
    TVM_ATTR_FIELD(dtype).describe("Target data type.").set_default(NullValue<DataType>());
  }
};  // struct InitOpAttrs

/*! \brief Attributes used in arange operators */
struct ArangeAttrs : public tvm::AttrsNode<ArangeAttrs> {
  Expr start;
  Expr stop;
  Expr step;
  DataType dtype;

  TVM_DECLARE_ATTRS(ArangeAttrs, "relay.attrs.ArangeAttrs") {
    TVM_ATTR_FIELD(start).describe("Start of interval. The interval includes this value.");
    TVM_ATTR_FIELD(stop).describe("Stop of interval. The interval does not include this value.");
    TVM_ATTR_FIELD(step).describe("Spacing between values.");
    TVM_ATTR_FIELD(dtype).describe("Target data type.");
  }
};  // struct ArangeAttrs

/*! \brief Attributes used in meshgrid operators */
struct MeshgridAttrs : public tvm::AttrsNode<MeshgridAttrs> {
  std::string indexing;

  TVM_DECLARE_ATTRS(MeshgridAttrs, "relay.attrs.MeshgridAttrs") {
    TVM_ATTR_FIELD(indexing)
        .describe(
            "Indexing mode, either \"ij\" for matrix or \"xy\" for cartesian in which first two"
            "dimensions are swapped.")
        .set_default("ij");
  }
};  // struct MeshgridAttrs

/*! \brief Attributes used in stack operators */
struct StackAttrs : public tvm::AttrsNode<StackAttrs> {
  Integer axis;
  TVM_DECLARE_ATTRS(StackAttrs, "relay.attrs.StackAttrs") {
    TVM_ATTR_FIELD(axis).set_default(0).describe(
        "The axis in the result array along which the input arrays are stacked.");
  }
};  // struct StackAttrs

/*! \brief Attributes used in repeat operators */
struct RepeatAttrs : public tvm::AttrsNode<RepeatAttrs> {
  Integer repeats;
  Integer axis;
  TVM_DECLARE_ATTRS(RepeatAttrs, "relay.attrs.RepeatAttrs") {
    TVM_ATTR_FIELD(repeats).describe("The number of repetitions for each element.");
    TVM_ATTR_FIELD(axis)
        .set_default(NullValue<Integer>())
        .describe(" The axis along which to repeat values.");
  }
};  // struct RepeatAttrs

/*! \brief Attributes used in tile operators */
struct TileAttrs : public tvm::AttrsNode<TileAttrs> {
  Array<Integer> reps;
  TVM_DECLARE_ATTRS(TileAttrs, "relay.attrs.TileAttrs") {
    TVM_ATTR_FIELD(reps).describe(
        "The number of times for repeating the tensor a."
        "Each dim sizeof reps must be a positive integer.");
  }
};  // struct TileAttrs

/*! \brief Attributes used in reverse operators */
struct ReverseAttrs : public tvm::AttrsNode<ReverseAttrs> {
  Integer axis;
  TVM_DECLARE_ATTRS(ReverseAttrs, "relay.attrs.ReverseAttrs") {
    TVM_ATTR_FIELD(axis)
        .set_default(NullValue<Integer>())
        .describe("The axis along which to reverse elements.");
  }
};  // struct ReverseAttrs

/*! \brief Attributes used in reverse_sequence operators */
struct ReverseSequenceAttrs : public tvm::AttrsNode<ReverseSequenceAttrs> {
  Integer seq_axis;
  Integer batch_axis;

  TVM_DECLARE_ATTRS(ReverseSequenceAttrs, "relay.attrs.ReverseSequenceAttrs") {
    TVM_ATTR_FIELD(seq_axis).set_default(1).describe(
        "The seq axis along which to reverse elements.");
    TVM_ATTR_FIELD(batch_axis)
        .set_default(0)
        .describe("The batch axis along which to slice the tensor.");
  }
};  // struct ReverseSequenceAttrs

/*! \brief Attributes used in squeeze operators */
struct SqueezeAttrs : public tvm::AttrsNode<SqueezeAttrs> {
  // use axis to make the name numpy compatible.
  Array<Integer> axis;

  TVM_DECLARE_ATTRS(SqueezeAttrs, "relay.attrs.SqueezeAttrs") {
    TVM_ATTR_FIELD(axis)
        .describe(
            "The axis to squeeze in the input tensor."
            "If `axis = None`, all axis of dimension 1 get squeezed;"
            "Else, the dimension in axes get squeezed."
            "It is an error if an axis does not has dimension 1.")
        .set_default(NullValue<Array<Integer>>());
  }
};  // struct SqueezeAttrs

struct SplitAttrs : public tvm::AttrsNode<SplitAttrs> {
  Variant<runtime::Int, Array<runtime::Int>> indices_or_sections;
  int axis;

  TVM_DECLARE_ATTRS(SplitAttrs, "relay.attrs.SplitAttrs") {
    TVM_ATTR_FIELD(indices_or_sections)
        .describe(
            "Indices or sections to split into. Accepts an int or a tuple"
            "If indices_or_sections is an integer, the input will be divided equally"
            "along given axis. If such a split is not possible, an error is raised."
            "If indices_or_sections is a tuple of sorted integers,"
            "the entries indicate where along axis the array is split.");
    TVM_ATTR_FIELD(axis).set_default(0).describe("the axis to be splitted.");
  }
};

/*! \brief Attributes for StridedSlice operator */
struct StridedSliceAttrs : public tvm::AttrsNode<StridedSliceAttrs> {
  Optional<Array<Integer>> begin;
  Optional<Array<Integer>> end;
  Optional<Array<Integer>> strides;
  tvm::String slice_mode;
  Optional<Array<Integer>> axes;

  TVM_DECLARE_ATTRS(StridedSliceAttrs, "relay.attrs.StridedSliceAttrs") {
    TVM_ATTR_FIELD(begin).describe("Indices for begin of slice, begin index is also inclusive");
    TVM_ATTR_FIELD(end).describe("Indices for end of slice, end index is exclusive");
    TVM_ATTR_FIELD(strides).describe(
        "Stride values of the slice, a stride can be negative, which causes a reverse slice.");
    TVM_ATTR_FIELD(slice_mode)
        .set_default("end")
        .describe(
            "The slice mode [end, size]."
            "end - The default slice mode, ending indices for the slice."
            "size - The input strides will be ignored, input end in this mode indicates the size"
            "of a slice starting at the location specified by begin. If end[i] is -1,"
            "all remaining elements in that dimension are included in the slice");
    TVM_ATTR_FIELD(axes).describe(
        "Axes along which slicing is applied. When it is specified, the length of begin, end, "
        "strides, and axes must be equal.");
  }
};

struct SliceLikeAttrs : public tvm::AttrsNode<SliceLikeAttrs> {
  Array<Integer> axes;

  TVM_DECLARE_ATTRS(SliceLikeAttrs, "relay.attrs.SliceLikeAttrs") {
    TVM_ATTR_FIELD(axes).describe(
        "List of axes on which input data will be sliced according to the "
        "corresponding size of the second input. By default will slice "
        "on all axes. Negative axes mean counting in reverse.");
  }
};

/*! \brief Attributes for Clip operator */
struct ClipAttrs : public tvm::AttrsNode<ClipAttrs> {
  double a_min;
  double a_max;

  TVM_DECLARE_ATTRS(ClipAttrs, "relay.attrs.ClipAttrs") {
    TVM_ATTR_FIELD(a_min).describe("The minimum clip value.");
    TVM_ATTR_FIELD(a_max).describe("The maximum clip value.");
  }
};

/*! \brief Attributes for FixedPointMultiply operator */
struct FixedPointMultiplyAttrs : public tvm::AttrsNode<FixedPointMultiplyAttrs> {
  int32_t multiplier;
  int32_t shift;

  TVM_DECLARE_ATTRS(FixedPointMultiplyAttrs, "relay.attrs.FixedPointMultiplyAttrs") {
    TVM_ATTR_FIELD(multiplier)
        .describe("Multiplier of a fixed floating point number described as multiplier*2^(shift)");
    TVM_ATTR_FIELD(shift).describe(
        "Shift of a fixed floating point number described as multiplier*2^(shift)");
  }
};

/*! \brief Attributes for per channel/per axes FixedPointMultiply operator */
struct FixedPointMultiplyPerAxisAttrs : public tvm::AttrsNode<FixedPointMultiplyPerAxisAttrs> {
  bool is_lshift_required;
  bool is_rshift_required;
  Array<Integer> axes;

  TVM_DECLARE_ATTRS(FixedPointMultiplyPerAxisAttrs, "relay.attrs.FixedPointMultiplyPerAxisAttrs") {
    TVM_ATTR_FIELD(is_lshift_required)
        .describe("Whether left shift is required in fixed point multiplication.")
        .set_default(false);
    TVM_ATTR_FIELD(is_rshift_required)
        .describe("Whether right shift is required in fixed point multiplication.")
        .set_default(false);
    TVM_ATTR_FIELD(axes).describe("List of axes on which input data was quantized.");
  }
};

/*! \brief Attributes for LayoutTransform operator */
struct LayoutTransformAttrs : public tvm::AttrsNode<LayoutTransformAttrs> {
  std::string src_layout;
  std::string dst_layout;

  TVM_DECLARE_ATTRS(LayoutTransformAttrs, "relay.attrs.LayoutTransformAttrs") {
    TVM_ATTR_FIELD(src_layout).describe("The source layout of the tensor. (e.g. NCHW)");
    TVM_ATTR_FIELD(dst_layout).describe("The destination layout of the tensor. (e.g. NCHW16c)");
  }
};

/*! \brief Attributes for AutoSchedulerLayoutTransform operator */
struct AutoSchedulerLayoutTransformAttrs
    : public tvm::AttrsNode<AutoSchedulerLayoutTransformAttrs> {
  std::string src_layout;
  std::string dst_layout;

  TVM_DECLARE_ATTRS(AutoSchedulerLayoutTransformAttrs,
                    "relay.attrs.AutoSchedulerLayoutTransformAttrs") {
    TVM_ATTR_FIELD(src_layout).describe("The source layout of the tensor. (e.g. 1N32C112H112W)");
    TVM_ATTR_FIELD(dst_layout)
        .describe("The destination layout of the tensor. (e.g. 1N2C112H112W16c)");
  }
};

/*! \brief Attributes for MetaScheduleLayoutTransform operator */
struct MetaScheduleLayoutTransformAttrs : public tvm::AttrsNode<MetaScheduleLayoutTransformAttrs> {
  tir::IndexMap index_map;

  TVM_DECLARE_ATTRS(MetaScheduleLayoutTransformAttrs,
                    "relay.attrs.MetaScheduleLayoutTransformAttrs") {
    TVM_ATTR_FIELD(index_map).describe(
        "The order of the extents, for example, "
        "let extents = [2, 3, 4], reorder = [0, 2, 1], and the shape of buffer A is (4, 6)"
        "then A[i, j] will be first rewritten to "
        "A[(6 * i + j) / 12, (6 * i + j) / 4 % 3 , (6 * i + j) % 4] according to the `extents`,"
        "and then reordered to A[(6 * i + j) / 12, (6 * i + j) % 4 , (6 * i + j) / 4 % 3]"
        "according to `reorder`");
  }
};

/*! \brief Attributes for ShapeOf operator */
struct ShapeOfAttrs : public tvm::AttrsNode<ShapeOfAttrs> {
  DataType dtype;

  TVM_DECLARE_ATTRS(ShapeOfAttrs, "relay.attrs.ShapeOfAttrs") {
    TVM_ATTR_FIELD(dtype).describe("Target data type").set_default(NullValue<DataType>());
  }
};

struct SequenceMaskAttrs : public tvm::AttrsNode<SequenceMaskAttrs> {
  double mask_value;
  int axis;

  TVM_DECLARE_ATTRS(SequenceMaskAttrs, "relay.attrs.SequenceMaskAttrs") {
    TVM_ATTR_FIELD(mask_value).set_default(0).describe("The masking value.");
    TVM_ATTR_FIELD(axis).set_default(0).describe(
        "The axis of the length dimension. Can only be 0 or 1.");
  }
};  // struct SequenceMaskAttrs.

/*! \brief Attributes used in sparse_to_dense operator */
struct SparseToDenseAttrs : public tvm::AttrsNode<SparseToDenseAttrs> {
  Array<Integer> output_shape;

  TVM_DECLARE_ATTRS(SparseToDenseAttrs, "relay.attrs.SparseToDenseAttrs") {
    TVM_ATTR_FIELD(output_shape).describe("Shape of the dense output tensor");
  }
};  // struct SparseToDenseAttrs

/*! \brief Attributes for ndarray_size operator */
struct NdarraySizeAttrs : public tvm::AttrsNode<NdarraySizeAttrs> {
  DataType dtype;

  TVM_DECLARE_ATTRS(NdarraySizeAttrs, "relay.attrs.NdarraySizeAttrs") {
    TVM_ATTR_FIELD(dtype).describe("Target data type").set_default(NullValue<DataType>());
  }
};

/*! \brief Attributes used in one-hot operator */
struct OneHotAttrs : public tvm::AttrsNode<OneHotAttrs> {
  int depth;
  int axis;
  DataType dtype;

  TVM_DECLARE_ATTRS(OneHotAttrs, "relay.attrs.OneHotAttrs") {
    TVM_ATTR_FIELD(depth).set_default(1).describe("Depth of the one hot dimension.");
    TVM_ATTR_FIELD(axis).set_default(-1).describe("Axis to fill.");
    TVM_ATTR_FIELD(dtype).set_default(NullValue<DataType>()).describe("Output data type.");
  }
};  // struct OneHotAttrs

/*! \brief Attributes used in matrix_set_diag operator */
struct MatrixSetDiagAttrs : public tvm::AttrsNode<MatrixSetDiagAttrs> {
  int k1;
  int k2;
  bool super_diag_right_align;
  bool sub_diag_right_align;

  TVM_DECLARE_ATTRS(MatrixSetDiagAttrs, "relay.attrs.MatrixSetDiagAttrs") {
    TVM_ATTR_FIELD(k1).set_default(0).describe("Lower limit (included) of the range of diagonals.");
    TVM_ATTR_FIELD(k2).set_default(0).describe("Upper limit (included) of the range of diagonals.");
    TVM_ATTR_FIELD(super_diag_right_align)
        .set_default(true)
        .describe("Bool, true iff super-diagonal is right aligned (left-padded).");
    TVM_ATTR_FIELD(sub_diag_right_align)
        .set_default(false)
        .describe("Bool, true iff sub-diagonal is right aligned (left-padded).");
  }
};  // struct MatrixSetDiagAttrs

/*! \brief Attributes used in cumsum and cumprod operator */
struct ScanopAttrs : public tvm::AttrsNode<ScanopAttrs> {
  Integer axis;
  DataType dtype;
  Bool exclusive = Bool(false);
  TVM_DECLARE_ATTRS(ScanopAttrs, "relay.attrs.ScanopAttrs") {
    TVM_ATTR_FIELD(axis).describe("The axis to operate over").set_default(NullValue<Integer>());
    TVM_ATTR_FIELD(dtype).describe("Output data type").set_default(NullValue<DataType>());

    // Default is 0 which is "false"
    TVM_ATTR_FIELD(exclusive)
        .describe("The first element is not included")
        .set_default(Bool(false));
  }
};  // struct ScanopAttrs

/*! \brief Attributes used in unique operator */
struct UniqueAttrs : public tvm::AttrsNode<UniqueAttrs> {
  bool sorted;
  bool return_counts;
  TVM_DECLARE_ATTRS(UniqueAttrs, "relay.attrs.UniqueAttrs") {
    TVM_ATTR_FIELD(sorted).describe("Whether the unique elements are sorted").set_default(true);
    TVM_ATTR_FIELD(return_counts)
        .describe("Whether to return an additional tensor with counts of each unique elements")
        .set_default(false);
  }
};  // struct UniqueAttrs

/*! \brief Attributes used in einsum operator */
struct EinsumAttrs : public tvm::AttrsNode<EinsumAttrs> {
  String equation;

  TVM_DECLARE_ATTRS(EinsumAttrs, "relay.attrs.EinsumAttrs") {
    TVM_ATTR_FIELD(equation).describe("The einsum expression string");
  }
};  // struct EinsumAttrs

/*! \brief Attributes used in stft operator */
struct StftAttrs : public tvm::AttrsNode<StftAttrs> {
  int n_fft;
  int hop_length;
  int win_length;
  bool normalized;
  bool onesided;

  TVM_DECLARE_ATTRS(StftAttrs, "relay.attrs.StftAttrs") {
    TVM_ATTR_FIELD(n_fft).set_default(-1).describe("The size of Fourier transform");
    TVM_ATTR_FIELD(hop_length)
        .set_default(-1)
        .describe("The distance between neighboring sliding window frames");
    TVM_ATTR_FIELD(win_length).set_default(-1).describe("The size of window frame and STFT filter");
    TVM_ATTR_FIELD(normalized)
        .set_default(false)
        .describe("Whether to return the normalized STFT results");
    TVM_ATTR_FIELD(onesided).set_default(true).describe(
        "Whether to return onesided result or fill with conjugate symmetry");
  }
};  // struct StftAttrs

/*! \brief Attributes used in DFT operator */
struct DFTAttrs : public tvm::AttrsNode<DFTAttrs> {
  Bool inverse = Bool(false);

  TVM_DECLARE_ATTRS(DFTAttrs, "relay.attrs.DFTAttrs") {
    TVM_ATTR_FIELD(inverse)
        .describe("Whether to perform the inverse discrete Fourier transform")
        .set_default(Bool(false));
  }
};  // struct DFTAttrs

struct TriluAttrs : public tvm::AttrsNode<TriluAttrs> {
  bool upper;

  TVM_DECLARE_ATTRS(TriluAttrs, "relay.attrs.TriluAttrs") {
    TVM_ATTR_FIELD(upper).set_default(true).describe(
        "Whether to keep the upper or lower half of the diagonal.");
  }
};  // struct TriluAttrs

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_TRANSFORM_H_
