/*!
 *  Copyright (c) 2017 by Contributors
 * \file transform.cc
 * \brief Injective transformation of shape or type.
 */
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/compiler/util.h>
#include <nnvm/top/tensor.h>
#include <cctype>
#include "../op_common.h"
#include "../elemwise_op_common.h"
#include "topi/nn/flatten.h"
#include "topi/transform.h"

namespace nnvm {
namespace top {
using namespace tvm;
using namespace nnvm::compiler;

// flatten
inline bool FlattenInferShape(const NodeAttrs& attrs,
                              std::vector<TShape>* in_attrs,
                              std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U) << "Input: [data]";
  CHECK_EQ(out_attrs->size(), 1U);
  const TShape &dshape = (*in_attrs)[0];
  if (dshape.ndim() == 0) return false;
  uint32_t target_dim = 1;
  for (uint32_t i = 1; i < dshape.ndim(); ++i) {
    target_dim *= dshape[i];
  }
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0,
                           TShape({dshape[0], target_dim}));
  return true;
}

NNVM_REGISTER_OP(flatten)
.describe(R"code(Flattens the input into a 2-D array.

For an input array with shape ``(d1, d2, ..., dk)``, `flatten` operation reshapes
the input array into an output array of shape ``(d1, d2*...*dk)``.

Example::

    x = [[
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ],
    [   [1,2,3],
        [4,5,6],
        [7,8,9]
    ]],

    flatten(x) = [[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
       [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]]

)code" NNVM_ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<FInferShape>("FInferShape", FlattenInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.add_argument("data", "Tensor", "Input data.")
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    return Array<Tensor>{ topi::nn::flatten(inputs[0]) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    return MakeGradNode("reshape_like", n,
                        {ograds[0], n->inputs[0]});
})
.set_support_level(1);

// concatenate
DMLC_REGISTER_PARAMETER(ConcatenateParam);

inline bool ConcatenateInferShape(const NodeAttrs& attrs,
                                  std::vector<TShape>* in_shape,
                                  std::vector<TShape>* out_shape) {
  const ConcatenateParam& param = nnvm::get<ConcatenateParam>(attrs.parsed);
  TShape dshape;
  dim_t size = 0;
  bool has_zero = false;
  for (size_t i = 0; i < in_shape->size(); ++i) {
    TShape tmp = (*in_shape)[i];
    if (tmp.ndim()) {
      CHECK_LT(static_cast<dim_t>(param.axis), tmp.ndim())
          << "concat dim " << param.axis << " out of range of input shape " << tmp;
      has_zero = tmp[param.axis] == 0 || has_zero;
      size += tmp[param.axis];
      tmp[param.axis] = 0;
      shape_assign(&dshape, tmp);
    }
  }

  TShape tmp = (*out_shape)[0];
  if (tmp.ndim()) {
    CHECK_LT(static_cast<dim_t>(param.axis), tmp.ndim())
        << "concat dim " << param.axis << " out of range of input shape " << tmp;
    tmp[param.axis] = 0;
    shape_assign(&dshape, tmp);
  }

  if (dshape.ndim() == 0) return false;

  for (size_t i = 0; i < in_shape->size(); ++i) {
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, i, dshape);
  }

  if (!has_zero) dshape[param.axis] = size;
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, dshape);
  return dshape.Size() != 0;
}

NNVM_REGISTER_OP(concatenate)
.describe(R"code(Joins input arrays along a given axis.

The dimensions of the input arrays should be the same except the axis along
which they will be concatenated.
The dimension of the output array along the concatenated axis will be equal
to the sum of the corresponding dimensions of the input arrays.

Example::

   x = [[1,1],[2,2]]
   y = [[3,3],[4,4],[5,5]]
   z = [[6,6], [7,7],[8,8]]

   concatenate(x,y,z,dim=0) = [[ 1.,  1.],
                               [ 2.,  2.],
                               [ 3.,  3.],
                               [ 4.,  4.],
                               [ 5.,  5.],
                               [ 6.,  6.],
                               [ 7.,  7.],
                               [ 8.,  8.]]

   Note that you cannot concat x,y,z along dimension 1 since dimension
   0 is not the same for all the input arrays.

   concatenate(y,z,dim=1) = [[ 3.,  3.,  6.,  6.],
                             [ 4.,  4.,  7.,  7.],
                             [ 5.,  5.,  8.,  8.]]

)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor-or-Tensor[]", "List of arrays to concatenate")
.add_arguments(ConcatenateParam::__FIELDS__())
.set_attr_parser(ParamParser<ConcatenateParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<ConcatenateParam>)
.set_attr<FInferShape>("FInferShape", ConcatenateInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<-1, 1>)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const ConcatenateParam& param = nnvm::get<ConcatenateParam>(attrs.parsed);
    return Array<Tensor>{ topi::concatenate(inputs, param.axis) };
})
.set_num_outputs(1)
.set_num_inputs(kVarg)
.set_support_level(1);

// expand_dims
DMLC_REGISTER_PARAMETER(ExpandDimsParam);

inline bool ExpandDimsInferShape(const NodeAttrs& attrs,
                                 std::vector<TShape>* in_shape,
                                 std::vector<TShape>* out_shape) {
  const ExpandDimsParam& param = nnvm::get<ExpandDimsParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 1U);
  const TShape& dshape = in_shape->at(0);
  int ndim = static_cast<int>(dshape.ndim());
  CHECK(param.axis >= -ndim - 1 && param.axis <= ndim);
  int axis = param.axis < 0 ? ndim + param.axis + 1 : param.axis;
  std::vector<dim_t> oshape;
  for (int i = 0; i < axis; ++i) {
    oshape.push_back(dshape[i]);
  }
  for (int i = 0; i < param.num_newaxis; ++i) {
    oshape.push_back(1);
  }
  for (int i = axis; i < ndim; ++i) {
    oshape.push_back(dshape[i]);
  }
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0,
                           TShape(oshape.begin(), oshape.end()));
  return true;
}

NNVM_REGISTER_OP(expand_dims)
.describe(R"code(Inserts a new axis of size 1 into the array shape

For example, given ``x`` with shape ``(2,3,4)``, then ``expand_dims(x, axis=1, num_newaxis=5)``
will return a new array with shape ``(2,5,3,4)``.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input tensor")
.add_arguments(ExpandDimsParam::__FIELDS__())
.set_attr_parser(ParamParser<ExpandDimsParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<ExpandDimsParam>)
.set_attr<FInferShape>("FInferShape", ExpandDimsInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const ExpandDimsParam& param = nnvm::get<ExpandDimsParam>(attrs.parsed);
    return Array<Tensor>{ topi::expand_dims(inputs[0], param.axis, param.num_newaxis) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    const ExpandDimsParam& param = nnvm::get<ExpandDimsParam>(n->attrs.parsed);
    return std::vector<NodeEntry> {
      MakeNode("sum", n->attrs.name + "_grad", {ograds[0]},
               {{"axis", std::to_string(param.axis)}})
    };
})
.set_support_level(1);

NNVM_REGISTER_OP(expand_like)
  .describe(R"code(Expand an input array with the shape of second array.
This operation can always be composed of unsqueezing and expanding dims.
Examples::
  input = [ 12.  19.  27.]
  input.shape = (3,)
  new_shape_array = [[[1,2],[2,3],[1,3]],
                     [[1,4],[4,3],[5,2]],
                     [[7,1],[7,2],[7,3]]]
  new_shape_array.shape = (3, 3, 2)
  expand_like(input, [1,2], new_shape_array) =
                    [[[12,12],[12,12],[12,12]],
                     [[19,19],[19,19],[19,19]],
                     [[27,27],[27,27],[27,27]]]
)code" NNVM_ADD_FILELINE)
.add_argument("input", "Tensor", "Source input")
.add_argument("shape_like", "Tensor", "Input with new shape")
.add_arguments(IndicatorParam::__FIELDS__())
.set_attr_parser(ParamParser<IndicatorParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<IndicatorParam>)
.set_attr<nnvm::FInferShape>("FInferShape", AssignOutputAttr<TShape, 1, 0>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds) {
    const IndicatorParam& param = nnvm::get<IndicatorParam>(n->attrs.parsed);
    std::ostringstream axis;
    axis << param.axis;

    return std::vector<NodeEntry>{
      MakeNode("sum", n->attrs.name + "_grad",
               {ograds[0]},
               {{"axis", axis.str()},
                {"exclude", std::to_string(param.exclude)}}),
      MakeNode("zeros_like", n->attrs.name + "_zero_grad", {n->inputs[1]})
    };
  })
  .set_support_level(4);

// split
DMLC_REGISTER_PARAMETER(SplitParam);

inline void SplitParamParser(nnvm::NodeAttrs* attrs) {
  SplitParam param;
  param.Init(attrs->dict);
  if (!std::isdigit(attrs->dict.at("indices_or_sections")[0])) {
    param.equal_split = false;
  } else {
    CHECK_EQ(param.indices_or_sections.ndim(), 1);
    param.equal_split = true;
  }
  attrs->parsed = std::move(param);
}

inline bool SplitInferShape(const NodeAttrs& attrs,
                            std::vector<TShape>* in_shape,
                            std::vector<TShape>* out_shape) {
  const SplitParam& param = nnvm::get<SplitParam>(attrs.parsed);
  const TShape& dshape = (*in_shape)[0];
  if (dshape.ndim() == 0) return false;

  if (param.equal_split) {
    int num_outputs = param.indices_or_sections[0];
    CHECK_EQ(out_shape->size(), static_cast<size_t>(num_outputs));
    CHECK_LT(param.axis, dshape.ndim());
    TShape oshape = dshape;
    CHECK_EQ(oshape[param.axis] % num_outputs, 0)
        << "indices_or_sections need to be able to divide input.shape[axis]";
    oshape[param.axis] /= num_outputs;

    for (size_t i = 0; i < out_shape->size(); ++i) {
      NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, i, oshape);
    }
  } else {
    dim_t num_outputs = param.indices_or_sections.ndim() + 1;
    CHECK_EQ(out_shape->size(), static_cast<size_t>(num_outputs));
    CHECK_LT(param.axis, dshape.ndim());
    TShape oshape = dshape;
    dim_t begin = 0;
    for (dim_t i = 0; i < num_outputs - 1; ++i) {
      CHECK_GT(param.indices_or_sections[i], begin)
          << "indices_or_sections need to be a sorted ascending list";
      oshape[param.axis] = param.indices_or_sections[i] - begin;
      begin = param.indices_or_sections[i];
      NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, i, oshape);
    }
    CHECK_LT(begin, dshape[param.axis])
        << "The sum of sections must match the input.shape[axis]";
    oshape[param.axis] = dshape[param.axis] - begin;
    NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, num_outputs - 1, oshape);
  }
  return true;
}

inline uint32_t SplitNumOutputs(const NodeAttrs& attrs) {
  const SplitParam& param = nnvm::get<SplitParam>(attrs.parsed);
  if (param.equal_split) {
    return static_cast<uint32_t>(param.indices_or_sections[0]);
  } else {
    return static_cast<uint32_t>(param.indices_or_sections.ndim()) + 1;
  }
}

// Intentionally not add ParamGetAttrDict for indices_or_sections.
NNVM_REGISTER_OP(split)
.describe(R"code(Splits an array along a particular axis into multiple sub-arrays.

**Note** that `indices_or_sections` should evenly divide the length of the axis
along which to split the array.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Array to be splitted")
.add_arguments(SplitParam::__FIELDS__())
.set_attr_parser(SplitParamParser)
.set_attr<FInferShape>("FInferShape", SplitInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, -1>)
.set_num_inputs(1)
.set_num_outputs(SplitNumOutputs)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const SplitParam& param = nnvm::get<SplitParam>(attrs.parsed);
    if (param.equal_split) {
      return Array<Tensor>{
        topi::split_sections(inputs[0], param.indices_or_sections[0], param.axis) };
    } else {
      Array<Expr> indices;
      for (auto i : param.indices_or_sections) {
        indices.push_back(tvm::make_const(tvm::Int(32), i));
      }
      return Array<Tensor>{ topi::split(inputs[0], indices, param.axis) };
    }
})
.set_support_level(1);

// cast
DMLC_REGISTER_PARAMETER(CastParam);

inline bool CastInferType(const NodeAttrs& attrs,
                          std::vector<int>* in_attrs,
                          std::vector<int>* out_attrs) {
  const CastParam& param = nnvm::get<CastParam>(attrs.parsed);
  CHECK_EQ(out_attrs->size(), 1U);
  NNVM_ASSIGN_OUTPUT_TYPE(attrs, *out_attrs, 0, param.dtype);
  return true;
}

NNVM_REGISTER_OP(cast)
.describe(R"code(Cast the content of input to dtype.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input data array")
.add_arguments(CastParam::__FIELDS__())
.set_attr_parser(ParamParser<CastParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<CastParam>)
.set_attr<FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<FInferType>("FInferType", CastInferType)
.set_num_inputs(1)
.set_num_outputs(1)
.set_support_level(1);


// reshape
DMLC_REGISTER_PARAMETER(ReshapeParam);

inline bool ReshapeInferShape(const NodeAttrs& attrs,
                              std::vector<TShape>* in_attrs,
                              std::vector<TShape>* out_attrs) {
  const ReshapeParam& param = nnvm::get<ReshapeParam>(attrs.parsed);
  CHECK_GT(param.shape.ndim(), 0);
  CHECK_EQ(in_attrs->size(), 1U) << "Input: [data]";
  CHECK_EQ(out_attrs->size(), 1U);

  const TShape &dshape = (*in_attrs)[0];
  if (dshape.ndim() == 0) return false;

  const Tuple<int64_t>& target_shape = param.shape;
  std::vector<int64_t> oshape;
  dim_t src_idx = 0;
  int infer_idx = -1;

  for (dim_t i = 0; i < target_shape.ndim(); ++i) {
    int svalue = target_shape[i];
    // special flag handling for shape inference.
    if (svalue > 0) {
      oshape.push_back(svalue);
      ++src_idx;
    } else if (svalue == 0) {
      // keep same
      CHECK_LT(src_idx, dshape.ndim());
      oshape.push_back(dshape[src_idx++]);
    } else if (svalue == -1) {
      // inference based on rest
      CHECK_LT(infer_idx, 0)
          << "One and only one dim can be inferred";
      infer_idx = i;
      oshape.push_back(1);
      ++src_idx;
    } else if (svalue == -2) {
      // copy all remaining dims from source
      while (src_idx < dshape.ndim()) {
        oshape.push_back(dshape[src_idx++]);
      }
    } else if (svalue == -3) {
      // merge two dims from source
      CHECK_LT(src_idx + 1, dshape.ndim());
      dim_t d1 = dshape[src_idx++];
      dim_t d2 = dshape[src_idx++];
      oshape.push_back(d1 * d2);
    } else if (svalue == -4) {
      // split the source dim s into two dims
      // read the left dim and then the right dim (either can be -1)
      CHECK_LT(i + 2, target_shape.ndim());
      CHECK_LT(src_idx, dshape.ndim());
      dim_t d0 = dshape[src_idx++];
      int d1 = target_shape[++i];
      int d2 = target_shape[++i];
      CHECK(d1 != -1 || d2 != -1) << "Split dims cannot both be -1.";
      if (d1 == -1) d1 = d0 / d2;
      if (d2 == -1) d2 = d0 / d1;
      CHECK_EQ(d1 * d2, static_cast<int>(d0)) <<
          "Split dims " << d1 << ", " << d2 << " do not divide original dim " << d0;
      oshape.push_back(d1);
      oshape.push_back(d2);
    }
  }

  if (infer_idx >= 0) {
    if (dshape.Size() > 0) {
      int new_size = 1;
      for (int x : oshape) {
        new_size *= x;
      }
      oshape[infer_idx] = dshape.Size() / new_size;
    } else {
      oshape[infer_idx] = 0;
    }
  }
  TShape out_shape(oshape.begin(), oshape.end());
  CHECK_EQ(out_shape.Size(), dshape.Size())
      << "Target shape size is different to source. "
      << "Target: " << out_shape
      << "\nSource: " << dshape;
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, out_shape);
  return true;
}

NNVM_REGISTER_OP(reshape)
.describe(R"code(Reshapes the input array.

Given an array and a shape, this function returns a copy of the array in the new shape.
The shape is a tuple of integers such as (2,3,4). The size of the new shape should be same as the size of the input array.

Example::

  reshape([1,2,3,4], shape=(2,2)) = [[1,2], [3,4]]

To give user more convenience in without doing manual shape inference,
some dimensions of the shape can take special values from the set {0, -1, -2, -3, -4}.
The significance of each is explained below:

- ``0``  copy this dimension from the input to the output shape.

  Example::

  - input shape = (2,3,4), shape = (4,0,2), output shape = (4,3,2)
  - input shape = (2,3,4), shape = (2,0,0), output shape = (2,3,4)

- ``-1`` infers the dimension of the output shape by using the remainder of the input dimensions
  keeping the size of the new array same as that of the input array.
  At most one dimension of shape can be -1.

  Example::

  - input shape = (2,3,4), shape = (6,1,-1), output shape = (6,1,4)
  - input shape = (2,3,4), shape = (3,-1,8), output shape = (3,1,8)
  - input shape = (2,3,4), shape=(-1,), output shape = (24,)

- ``-2`` copy all/remainder of the input dimensions to the output shape.

  Example::

  - input shape = (2,3,4), shape = (-2,), output shape = (2,3,4)
  - input shape = (2,3,4), shape = (2,-2), output shape = (2,3,4)
  - input shape = (2,3,4), shape = (-2,1,1), output shape = (2,3,4,1,1)

- ``-3`` use the product of two consecutive dimensions of the input shape as the output dimension.

  Example::

  - input shape = (2,3,4), shape = (-3,4), output shape = (6,4)
  - input shape = (2,3,4,5), shape = (-3,-3), output shape = (6,20)
  - input shape = (2,3,4), shape = (0,-3), output shape = (2,12)
  - input shape = (2,3,4), shape = (-3,-2), output shape = (6,4)

- ``-4`` split one dimension of the input into two dimensions passed subsequent to -4 in shape (can contain -1).

  Example::

  - input shape = (2,3,4), shape = (-4,1,2,-2), output shape =(1,2,3,4)
  - input shape = (2,3,4), shape = (2,-4,-1,3,-2), output shape = (2,1,3,4)

)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input data.")
.add_arguments(ReshapeParam::__FIELDS__())
.set_attr_parser(ParamParser<ReshapeParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<ReshapeParam>)
.set_attr<FInferShape>("FInferShape", ReshapeInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    return Array<Tensor>{ topi::reshape(inputs[0], out_info[0]->shape) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds) {
    return std::vector<NodeEntry>{
      MakeNode("reshape_like", n->attrs.name + "_grad",
               {ograds[0], n->inputs[0]})
    };
})
.set_support_level(3);

NNVM_REGISTER_OP(reshape_like)
  .describe(R"code(Reshapes the input array by the size of another array.
For an input array with shape ``(d1, d2, ..., dk)``, `reshape_like` operation reshapes
the input array into an output array with the same shape as the second input array.
.. note::
    Sizes for both array should be compatible.
)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input data.")
.add_argument("shape_like", "Tensor", "Input data.")
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<FInferShape>(
  "FInferShape", [](const NodeAttrs& attrs,
                    std::vector<TShape>* in_attrs,
                    std::vector<TShape>* out_attrs) {
    CHECK_EQ(in_attrs->at(0).Size(), in_attrs->at(1).Size())
      << "Reshape inputs size should be compatible";
    NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, in_attrs->at(1));
    return true;
})
.set_attr<FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds) {
    return std::vector<NodeEntry>{
      MakeNode("reshape_like", n->attrs.name + "_grad", {ograds[0], n->inputs[0]}),
      MakeNode("zeros_like", n->attrs.name + "_zero_grad", { n->inputs[1]})
    };
})
.set_support_level(4);

// squeeze
DMLC_REGISTER_PARAMETER(SqueezeParam);

inline bool SqueezeShape(const nnvm::NodeAttrs& attrs,
                           std::vector<TShape>* in_attrs,
                           std::vector<TShape>* out_attrs) {
  const SqueezeParam& param = nnvm::get<SqueezeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const TShape& shp = (*in_attrs)[0];
  if (shp.ndim() == 0) return false;

  std::vector<int64_t> oshape;
  if (param.axis.ndim() == 0) {
    for (dim_t i = 0; i < shp.ndim(); ++i) {
      if (shp[i] != 1) {
        oshape.emplace_back(shp[i]);
      }
    }
  } else {
    std::unordered_set<dim_t> axis_checker;
    for (size_t i = 0; i < param.axis.ndim(); ++i) {
      if (param.axis[i] < 0) {
        int real_axis = param.axis[i] + static_cast<int>(shp.ndim());
        CHECK(real_axis < static_cast<int>(shp.ndim()) && real_axis >= 0);
        axis_checker.insert(real_axis);
      }
    }
    for (size_t i = 0; i < shp.ndim(); ++i) {
      if (axis_checker.find(i) == axis_checker.end()) {
        oshape.emplace_back(shp[i]);
      } else {
        CHECK_EQ(shp[i], 1) << "The squeezed axis must have shape 1!"
                            << "Want to squeeze " << i
                            << ", which has shape" << shp[i];
      }
    }
  }
  if (oshape.size() == 0) {
    // Handles the case where all axes are squeezed.
    oshape.push_back(1);
  }
  TShape out_shape(oshape.begin(), oshape.end());
  CHECK_EQ(out_shape.Size(), shp.Size())
      << "Target shape size is different to source. "
      << "Target: " << out_shape
      << "\nSource: " << shp;
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, out_shape);
  return true;
}

NNVM_REGISTER_OP(squeeze)
.describe(R"code(Squeeze axises in the array.

Examples::

  x = [[[0], [1], [2]]]
  x.shape = (1, 3, 1)

  squeeze(x) = [0, 1, 2]

  squeeze(x, 0) = [[0], [1], [2]]

  squeeze(x, (0, 2)) = [0, 1, 2]

)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Source input")
.add_arguments(SqueezeParam::__FIELDS__())
.set_attr_parser(ParamParser<SqueezeParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<SqueezeParam>)
.set_attr<nnvm::FInferShape>("FInferShape", SqueezeShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const SqueezeParam& param = nnvm::get<SqueezeParam>(attrs.parsed);
    auto axis = ShapeToArray(param.axis);
    return Array<Tensor>{ topi::squeeze(inputs[0], axis) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds) {
    return std::vector<NodeEntry>{
      MakeNode("reshape_like", n->attrs.name + "_grad",
               {ograds[0], n->inputs[0]})
    };
})
.set_support_level(1);

// tranpose
DMLC_REGISTER_PARAMETER(TransposeParam);

inline bool TransposeShape(const nnvm::NodeAttrs& attrs,
                           std::vector<TShape>* in_attrs,
                           std::vector<TShape>* out_attrs) {
  const TransposeParam& param = nnvm::get<TransposeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const TShape& shp = (*in_attrs)[0];
  if (shp.ndim() == 0) return false;

  TShape ret(shp.ndim());
  if (param.axes.ndim() == 0) {
    for (dim_t i = 0; i < shp.ndim(); ++i) {
      ret[i] = shp[shp.ndim() - 1 - i];
    }
  } else {
    CHECK_EQ(shp.ndim(), param.axes.ndim());
    for (size_t i = 0; i < shp.ndim(); ++i) {
      CHECK(param.axes[i] < shp.ndim());
      ret[i] = shp[param.axes[i]];
    }
  }
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, ret);
  return true;
}

NNVM_REGISTER_OP(transpose)
.describe(R"code(Permutes the dimensions of an array.

Examples::

  x = [[ 1, 2],
       [ 3, 4]]

  transpose(x) = [[ 1.,  3.],
                  [ 2.,  4.]]

  x = [[[ 1.,  2.],
        [ 3.,  4.]],

       [[ 5.,  6.],
        [ 7.,  8.]]]

  transpose(x) = [[[ 1.,  5.],
                   [ 3.,  7.]],

                  [[ 2.,  6.],
                   [ 4.,  8.]]]

  transpose(x, axes=(1,0,2)) = [[[ 1.,  2.],
                                 [ 5.,  6.]],

                                [[ 3.,  4.],
                                 [ 7.,  8.]]]
)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Source input")
.add_arguments(TransposeParam::__FIELDS__())
.set_attr_parser(ParamParser<TransposeParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<TransposeParam>)
.set_attr<nnvm::FInferShape>("FInferShape", TransposeShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_support_level(4)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const TransposeParam& param = nnvm::get<TransposeParam>(attrs.parsed);
    auto axes = ShapeToArray(param.axes);
    return Array<Tensor>{ topi::transpose(inputs[0], axes) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds) {
    const TransposeParam& param = nnvm::get<TransposeParam>(n->attrs.parsed);
    std::ostringstream oss; oss << param.axes;
    return std::vector<NodeEntry>{
      MakeNode("transpose", n->attrs.name + "_t", {ograds[0]}, {{"axes", oss.str()}})
    };
});

}  // namespace top
}  // namespace nnvm
