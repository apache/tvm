/*!
 *  Copyright (c) 2017 by Contributors
 * \file tensor.h
 * \brief Auxiliary param for tensor primitive.
 */
#ifndef NNVM_TOP_TENSOR_H_
#define NNVM_TOP_TENSOR_H_

#include <dmlc/base.h>
#include <dmlc/parameter.h>
#include <nnvm/tuple.h>

namespace nnvm {
namespace top {

struct ConcatenateParam : public dmlc::Parameter<ConcatenateParam> {
  int axis;
  DMLC_DECLARE_PARAMETER(ConcatenateParam) {
    DMLC_DECLARE_FIELD(axis).set_lower_bound(0).set_default(1)
    .describe("the axis to be concated.");
  }
};

struct ExpandDimsParam : public dmlc::Parameter<ExpandDimsParam> {
  int axis;
  int num_newaxis;
  DMLC_DECLARE_PARAMETER(ExpandDimsParam) {
    DMLC_DECLARE_FIELD(axis)
    .describe("the axis to be expanded.");
    DMLC_DECLARE_FIELD(num_newaxis).set_lower_bound(1).set_default(1)
    .describe("Number of new axis to be inserted.");
  }
};

struct SplitParam : public dmlc::Parameter<SplitParam> {
  // numpy convention, only support indices, not support list.
  Tuple<int> indices_or_sections;
  int axis;
  // additional hint whether it is equal_split mode
  // deduced from indices_or_sections
  bool equal_split;

  DMLC_DECLARE_PARAMETER(SplitParam) {
    DMLC_DECLARE_FIELD(indices_or_sections)
        .describe("Number of outputs to be splitted");
    DMLC_DECLARE_FIELD(axis).set_lower_bound(0).set_default(1)
        .describe("the axis to be splitted.");
  }
};

enum TypeFlag {
  kFloat32 = 0,
  kFloat64 = 1,
  kFloat16 = 2,
  kUint8 = 3,
  kInt32 = 4,
  kInt8  = 5,
  kInt64 = 6,
  kInt16 = 7,
};

struct CastParam : public dmlc::Parameter<CastParam> {
  int dtype;
  DMLC_DECLARE_PARAMETER(CastParam) {
    DMLC_DECLARE_FIELD(dtype)
    .add_enum("float32", kFloat32)
    .add_enum("float64", kFloat64)
    .add_enum("float16", kFloat16)
    .add_enum("uint8", kUint8)
    .add_enum("int32", kInt32)
    .add_enum("int8", kInt8)
    .add_enum("int64", kInt64)
    .add_enum("int16", kInt16)
    .describe("Output data type.");
  }
};

struct ReshapeParam : public dmlc::Parameter<ReshapeParam> {
  Tuple<int64_t> shape;

  DMLC_DECLARE_PARAMETER(ReshapeParam) {
    DMLC_DECLARE_FIELD(shape);
  }
};

struct SqueezeParam : public dmlc::Parameter<SqueezeParam> {
  TShape axis;

  DMLC_DECLARE_PARAMETER(SqueezeParam) {
    DMLC_DECLARE_FIELD(axis).set_default(TShape())
    .describe("The axis to squeeze in the input tensor."
              " If set to None, all size=1 axes will be squeezed");
  }
};

struct ScalarParam : public dmlc::Parameter<ScalarParam> {
  double scalar;

  DMLC_DECLARE_PARAMETER(ScalarParam) {
    DMLC_DECLARE_FIELD(scalar);
  }
};

struct TransposeParam : public dmlc::Parameter<TransposeParam> {
  TShape axes;

  DMLC_DECLARE_PARAMETER(TransposeParam) {
    DMLC_DECLARE_FIELD(axes).set_default(TShape())
    .describe("Target axis order. By default the axes will be inverted.");
  }
};

struct BroadcastToParam : public dmlc::Parameter<BroadcastToParam> {
  TShape shape;

  DMLC_DECLARE_PARAMETER(BroadcastToParam) {
    DMLC_DECLARE_FIELD(shape).set_default(TShape())
      .describe("The shape of the desired array."
                " We can set the dim to zero if it's same as the original."
                " E.g `A = broadcast_to(B, shape=(10, 0, 0))` ");
  }
};

struct ReduceParam : public dmlc::Parameter<ReduceParam> {
  TShape axis;
  bool keepdims;
  bool exclude;

  DMLC_DECLARE_PARAMETER(ReduceParam) {
    DMLC_DECLARE_FIELD(axis).set_default(TShape())
        .describe(R"code(The axis or axes along which to perform the reduction.

      The default, `axis=()`, will compute over all elements into a
      scalar array with shape `(1,)`.

      If `axis` is int, a reduction is performed on a particular axis.

      If `axis` is a tuple of ints, a reduction is performed on all the axes
      specified in the tuple.

      If `exclude` is true, reduction will be performed on the axes that are
      NOT in axis instead.)code");

    DMLC_DECLARE_FIELD(keepdims).set_default(false)
      .describe("If this is set to `True`, the reduced axes are left "
                "in the result as dimension with size one.");
    DMLC_DECLARE_FIELD(exclude).set_default(false)
      .describe("Whether to perform reduction on axis that are NOT in axis instead.");
  }
};

}  // namespace top
}  // namespace nnvm

#endif  // NNVM_TOP_TENSOR_H_
