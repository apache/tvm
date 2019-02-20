/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/attrs/transform.h
 * \brief Transform operators.
 */
#ifndef TVM_RELAY_ATTRS_TRANSFORM_H_
#define TVM_RELAY_ATTRS_TRANSFORM_H_

#include <tvm/attrs.h>
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

  TVM_DECLARE_ATTRS(TakeAttrs, "relay.attrs.TakeAttrs") {
    TVM_ATTR_FIELD(axis).set_default(NullValue<Integer>())
        .describe("The axis over which to select values.");
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
  tvm::Expr start;
  tvm::Expr stop;
  tvm::Expr step;
  DataType dtype;

  TVM_DECLARE_ATTRS(ArangeAttrs, "relay.attrs.ArangeAttrs") {
    TVM_ATTR_FIELD(start).set_default(make_const(Float(32), 0))
        .describe("Start of interval. The interval includes this value.");
    TVM_ATTR_FIELD(stop)
        .describe("Stop of interval. The interval does not include this value.");
    TVM_ATTR_FIELD(step).set_default(make_const(Float(32), 1))
        .describe("Spacing between values.");
    TVM_ATTR_FIELD(dtype).set_default(NullValue<DataType>())
        .describe("Target data type.");
  }
};  // struct ArangeAttrs

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

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_TRANSFORM_H_
