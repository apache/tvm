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

#include <tvm/attrs.h>
#include <tvm/relay/base.h>
#include <string>

namespace tvm {
namespace relay {

/*! \brief data type cast */
struct CastAttrs : public tvm::AttrsNode<CastAttrs> {
  DataType dtype;

  TVM_DECLARE_ATTRS(CastAttrs, "relay.attrs.CastAttrs") {
    TVM_ATTR_FIELD(dtype)
        .describe("Target data type");
  }
};  // struct CastAttrs.

/*! \brief Attributes used in expand_dims operators */
struct ExpandDimsAttrs : public tvm::AttrsNode<ExpandDimsAttrs> {
  int axis;
  int num_newaxis;

  TVM_DECLARE_ATTRS(ExpandDimsAttrs, "relay.attrs.ExpandDimsAttrs") {
    TVM_ATTR_FIELD(axis)
        .describe("The axis at which the input array is expanded."
                  "Should lie in range `[-data.ndim - 1, data.ndim]`."
                  "If `axis < 0`, it is the first axis inserted;"
                  "If `axis >= 0`, it is the last axis inserted in Python's negative indexing.");
    TVM_ATTR_FIELD(num_newaxis)
        .describe("Number of axises to be inserted. Should be >= 0.")
        .set_lower_bound(0)
        .set_default(1);
  }
};  // struct ExpandDimsAttrs

/*! \brief Attributes used in concatenate operators */
struct ConcatenateAttrs : public tvm::AttrsNode<ConcatenateAttrs> {
  int axis;
  TVM_DECLARE_ATTRS(ConcatenateAttrs, "relay.attrs.ConcatenateAttrs") {
    TVM_ATTR_FIELD(axis)
        .describe("The axis at which the input arrays are concatenated."
                  "Should lie in range `[-ndim, ndim)`.")
        .set_default(0);
  }
};  // struct ConcatenateAttrs

/*! \brief Attributes used in transpose operators */
struct TransposeAttrs : public tvm::AttrsNode<TransposeAttrs> {
  Array<Integer> axes;
  TVM_DECLARE_ATTRS(TransposeAttrs, "relay.attrs.TransposeAttrs") {
    TVM_ATTR_FIELD(axes)
        .describe("The target axes order, reverse order if not specified.");
  }
};  // struct TransposeAttrs

/*! \brief Attributes used in reshape operators */
struct ReshapeAttrs : public tvm::AttrsNode<ReshapeAttrs> {
  Array<Integer> newshape;
  bool reverse;
  TVM_DECLARE_ATTRS(ReshapeAttrs, "relay.attrs.ReshapeAttrs") {
    TVM_ATTR_FIELD(newshape)
        .describe("The new shape. Should be compatible with the original shape.");
    TVM_ATTR_FIELD(reverse)
        .describe("Infer the special values from right to left if true")
        .set_default(false);
  }
};  // struct ReshapeAttrs

struct TakeAttrs : public tvm::AttrsNode<TakeAttrs> {
  Integer axis;
  std::string mode;

  TVM_DECLARE_ATTRS(TakeAttrs, "relay.attrs.TakeAttrs") {
    TVM_ATTR_FIELD(axis).set_default(NullValue<Integer>())
        .describe("The axis over which to select values.");
    TVM_ATTR_FIELD(mode).set_default("clip")
        .describe("Specify how out-of-bound indices will behave."
                  "clip - clip to the range (default)"
                  "wrap - wrap around the indices"
                  "fast - no clip or wrap around (user must make sure indices are in-bound)");
  }
};

/*! \brief Attributes that specify a tensor */
struct InitOpAttrs : public tvm::AttrsNode<InitOpAttrs> {
  Array<IndexExpr> shape;
  DataType dtype;

  TVM_DECLARE_ATTRS(InitOpAttrs, "relay.attrs.InitOpAttrs") {
    TVM_ATTR_FIELD(shape)
      .describe("Target shape.");
    TVM_ATTR_FIELD(dtype)
      .describe("Target data type.")
      .set_default(NullValue<DataType>());
  }
};  // struct InitOpAttrs

/*! \brief Attributes used in arange operators */
struct ArangeAttrs : public tvm::AttrsNode<ArangeAttrs> {
  Expr start;
  Expr stop;
  Expr step;
  DataType dtype;

  TVM_DECLARE_ATTRS(ArangeAttrs, "relay.attrs.ArangeAttrs") {
    TVM_ATTR_FIELD(start)
        .describe("Start of interval. The interval includes this value.");
    TVM_ATTR_FIELD(stop)
        .describe("Stop of interval. The interval does not include this value.");
    TVM_ATTR_FIELD(step)
        .describe("Spacing between values.");
    TVM_ATTR_FIELD(dtype)
        .describe("Target data type.");
  }
};  // struct ArangeAttrs

/*! \brief Attributes used in stack operators */
struct StackAttrs : public tvm::AttrsNode<StackAttrs> {
  Integer axis;
  TVM_DECLARE_ATTRS(StackAttrs, "relay.attrs.StackAttrs") {
    TVM_ATTR_FIELD(axis).set_default(0)
        .describe("The axis in the result array along which the input arrays are stacked.");
  }
};  // struct StackAttrs

/*! \brief Attributes used in repeat operators */
struct RepeatAttrs : public tvm::AttrsNode<RepeatAttrs> {
  Integer repeats;
  Integer axis;
  TVM_DECLARE_ATTRS(RepeatAttrs, "relay.attrs.RepeatAttrs") {
    TVM_ATTR_FIELD(repeats)
        .describe("The number of repetitions for each element.");
    TVM_ATTR_FIELD(axis).set_default(NullValue<Integer>())
        .describe(" The axis along which to repeat values.");
  }
};  // struct RepeatAttrs

/*! \brief Attributes used in tile operators */
struct TileAttrs : public tvm::AttrsNode<TileAttrs> {
  Array<Integer> reps;
  TVM_DECLARE_ATTRS(TileAttrs, "relay.attrs.TileAttrs") {
    TVM_ATTR_FIELD(reps)
        .describe("The number of times for repeating the tensor a."
                  "Each dim sizeof reps must be a positive integer.");
  }
};  // struct TileAttrs

/*! \brief Attributes used in reverse operators */
struct ReverseAttrs : public tvm::AttrsNode<ReverseAttrs> {
  Integer axis;
  TVM_DECLARE_ATTRS(ReverseAttrs, "relay.attrs.ReverseAttrs") {
    TVM_ATTR_FIELD(axis).set_default(NullValue<Integer>())
        .describe("The axis along which to reverse elements.");
  }
};  // struct ReverseAttrs

/*! \brief Attributes used in squeeze operators */
struct SqueezeAttrs : public tvm::AttrsNode<SqueezeAttrs> {
  // use axis to make the name numpy compatible.
  Array<Integer> axis;

  TVM_DECLARE_ATTRS(SqueezeAttrs, "relay.attrs.SqueezeAttrs") {
    TVM_ATTR_FIELD(axis)
        .describe("The axis to squeeze in the input tensor."
                  "If `axis = None`, all axis of dimension 1 get squeezed;"
                  "Else, the dimension in axes get squeezed."
                  "It is an error if an axis does not has dimension 1.")
        .set_default(NullValue<Array<Integer> >());
  }
};  // struct SqueezeAttrs

struct SplitAttrs : public tvm::AttrsNode<SplitAttrs> {
  NodeRef indices_or_sections;
  int axis;

  TVM_DECLARE_ATTRS(SplitAttrs, "relay.attrs.SplitAttrs") {
    TVM_ATTR_FIELD(indices_or_sections)
        .describe("Indices or sections to split into. Accepts an int or a tuple"
                  "If indices_or_sections is an integer, the input will be divided equally"
                  "along given axis. If such a split is not possible, an error is raised."
                  "If indices_or_sections is a tuple of sorted integers,"
                  "the entries indicate where along axis the array is split.");
    TVM_ATTR_FIELD(axis).set_default(0)
        .describe("the axis to be splitted.");
  }
};

/*! \brief Attributes for StridedSlice operator */
struct StridedSliceAttrs : public tvm::AttrsNode<StridedSliceAttrs> {
  Array<Integer> begin;
  Array<Integer> end;
  Array<Integer> strides;

  TVM_DECLARE_ATTRS(StridedSliceAttrs, "relay.attrs.StridedSliceAttrs") {
    TVM_ATTR_FIELD(begin)
        .describe("Indices for begin of slice, begin index is also inclusive");
    TVM_ATTR_FIELD(end)
        .describe("Indices for end of slice, end index is exclusive");
    TVM_ATTR_FIELD(strides).set_default(Array<Integer>({}))
        .describe("Stride values of the slice");
  }
};

struct SliceLikeAttrs : public tvm::AttrsNode<SliceLikeAttrs> {
  Array<Integer> axes;

  TVM_DECLARE_ATTRS(SliceLikeAttrs, "relay.attrs.SliceLikeAttrs") {
    TVM_ATTR_FIELD(axes)
        .describe("List of axes on which input data will be sliced according to the "
                  "corresponding size of the second input. By default will slice "
                  "on all axes. Negative axes mean counting in reverse.");
  }
};

/*! \brief Attributes for Clip operator */
struct ClipAttrs : public tvm::AttrsNode<ClipAttrs> {
  double a_min;
  double a_max;

  TVM_DECLARE_ATTRS(ClipAttrs, "relay.attrs.ClipAttrs") {
    TVM_ATTR_FIELD(a_min)
      .describe("The minimum clip value.");
    TVM_ATTR_FIELD(a_max)
      .describe("The maximum clip value.");
  }
};

/*! \brief Attributes for LayoutTransform operator */
struct LayoutTransformAttrs : public tvm::AttrsNode<LayoutTransformAttrs> {
  std::string src_layout;
  std::string dst_layout;

  TVM_DECLARE_ATTRS(LayoutTransformAttrs, "relay.attrs.LayoutTransformAttrs") {
    TVM_ATTR_FIELD(src_layout)
        .describe("The source layout of the tensor. (e.g. NCHW)");
    TVM_ATTR_FIELD(dst_layout)
        .describe("The destination layout of the tensor. (e.g. NCHW16c)");
  }
};

/*! \brief Attributes for ShapeOf operator */
struct ShapeOfAttrs : public tvm::AttrsNode<ShapeOfAttrs> {
  DataType dtype;

  TVM_DECLARE_ATTRS(ShapeOfAttrs, "relay.attrs.ShapeOfAttrs") {
    TVM_ATTR_FIELD(dtype)
        .describe("Target data type")
        .set_default(NullValue<DataType>());
  }
};

struct SequenceMaskAttrs : public tvm::AttrsNode<SequenceMaskAttrs> {
  double mask_value;
  int axis;

  TVM_DECLARE_ATTRS(SequenceMaskAttrs, "relay.attrs.SequenceMaskAttrs") {
    TVM_ATTR_FIELD(mask_value).set_default(0)
      .describe("The masking value.");
    TVM_ATTR_FIELD(axis).set_default(0)
      .describe("The axis of the length dimension. Can only be 0 or 1.");
  }
};  // struct SequenceMaskAttrs.

/*! \brief Attributes for ndarray_size operator */
struct NdarraySizeAttrs : public tvm::AttrsNode<NdarraySizeAttrs> {
  DataType dtype;

  TVM_DECLARE_ATTRS(NdarraySizeAttrs, "relay.attrs.NdarraySizeAttrs") {
    TVM_ATTR_FIELD(dtype)
        .describe("Target data type")
        .set_default(NullValue<DataType>());
  }
};

/*! \brief Attributes used in one-hot operator */
struct OneHotAttrs : public tvm::AttrsNode<OneHotAttrs> {
  int depth;
  int axis;
  DataType dtype;

  TVM_DECLARE_ATTRS(OneHotAttrs, "relay.attrs.OneHotAttrs") {
    TVM_ATTR_FIELD(depth).set_default(1)
        .describe("Depth of the one hot dimension.");
    TVM_ATTR_FIELD(axis).set_default(-1)
        .describe("Axis to fill.");
    TVM_ATTR_FIELD(dtype).set_default(NullValue<DataType>())
        .describe("Output data type.");
  }
};  // struct OneHotAttrs

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_TRANSFORM_H_
