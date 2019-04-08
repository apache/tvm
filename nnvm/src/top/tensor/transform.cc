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
#include <sstream>
#include "../op_common.h"
#include "../elemwise_op_common.h"
#include "topi/nn/flatten.h"
#include "topi/transform.h"
#include "topi/elemwise.h"
#include "topi/detail/constant_utils.h"
#include "../../compiler/compile_engine.h"

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
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
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
  int axis = param.axis >= 0 ? param.axis : in_shape->at(0).ndim() + param.axis;
  for (size_t i = 0; i < in_shape->size(); ++i) {
    TShape tmp = (*in_shape)[i];
    if (tmp.ndim()) {
      CHECK_LT(static_cast<dim_t>(axis), tmp.ndim())
          << "concat dim " << axis << " out of range of input shape " << tmp;
      has_zero = tmp[axis] == 0 || has_zero;
      size += tmp[axis];
      tmp[axis] = 0;
      shape_assign(&dshape, tmp);
    }
  }

  TShape tmp = (*out_shape)[0];
  if (tmp.ndim()) {
    CHECK_LT(static_cast<dim_t>(axis), tmp.ndim())
        << "concat dim " << axis << " out of range of input shape " << tmp;
    tmp[axis] = 0;
    shape_assign(&dshape, tmp);
  }

  if (dshape.ndim() == 0) return false;

  for (size_t i = 0; i < in_shape->size(); ++i) {
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, i, dshape);
  }

  if (!has_zero) dshape[axis] = size;
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, dshape);
  return dshape.Size() != 0;
}

inline bool ConcatenateCorrectLayout(const NodeAttrs& attrs,
                                     std::vector<Layout> *ilayouts,
                                     const std::vector<Layout> *last_ilayouts,
                                     std::vector<Layout> *olayouts) {
  const ConcatenateParam& param = nnvm::get<ConcatenateParam>(attrs.parsed);
  CHECK_EQ(ilayouts->size(), last_ilayouts->size());
  CHECK_EQ(olayouts->size(), 1U);

  Layout layout;
  if (!ilayouts->at(0).defined()) {
    layout = last_ilayouts->at(0);
  } else if (param.axis >= static_cast<int>(ilayouts->at(0).ndim())) {
    CHECK(last_ilayouts->at(0).defined())
      << "Current input layout " << ilayouts->at(0)
      << " is invalid but last input layout is not "
         "defined for the first input.";
    layout = last_ilayouts->at(0);
  } else if (last_ilayouts->at(0).defined()
             && ilayouts->at(0)[param.axis]
                != last_ilayouts->at(0)[param.axis]) {
    layout = last_ilayouts->at(0);
  } else {
    layout = ilayouts->at(0);
  }

  for (size_t i = 0; i < ilayouts->size(); ++i) {
    NNVM_ASSIGN_LAYOUT(*ilayouts, i, layout);
  }
  NNVM_ASSIGN_LAYOUT(*olayouts, 0, layout);
  return true;
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

   concatenate(x,y,z,axis=0) = [[ 1.,  1.],
                               [ 2.,  2.],
                               [ 3.,  3.],
                               [ 4.,  4.],
                               [ 5.,  5.],
                               [ 6.,  6.],
                               [ 7.,  7.],
                               [ 8.,  8.]]

   Note that you cannot concat x,y,z along dimension 1 since dimension
   0 is not the same for all the input arrays.

   concatenate(y,z,axis=1) = [[ 3.,  3.,  6.,  6.],
                             [ 4.,  4.,  7.,  7.],
                             [ 5.,  5.,  8.,  8.]]

)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor-or-Tensor[]", "List of arrays to concatenate")
.add_arguments(ConcatenateParam::__FIELDS__())
.set_attr_parser(ParamParser<ConcatenateParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<ConcatenateParam>)
.set_attr<FInferShape>("FInferShape", ConcatenateInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<-1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ConcatenateCorrectLayout)
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
  CHECK(param.axis >= -ndim - 1 && param.axis <= ndim)
    << "with axis = " << param.axis << " ndim = " << ndim;
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
will return a new array with shape ``(2,1,1,1,1,1,3,4)``.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input tensor")
.add_arguments(ExpandDimsParam::__FIELDS__())
.set_attr_parser(ParamParser<ExpandDimsParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<ExpandDimsParam>)
.set_attr<FInferShape>("FInferShape", ExpandDimsInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
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
    return std::vector<NodeEntry> {
      MakeNode("collapse_sum", n->attrs.name + "_grad", {ograds[0], n->inputs[0]})
    };
})
.set_support_level(1);

NNVM_REGISTER_OP(expand_like)
  .describe(R"code(Expand an input array with the shape of second array.
This operation can be thought of as a composition of expand_dims and broadcast_to.
If the dimensions are already expanded then it just broadcasts.
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
// never transform layout of the second input array.
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds) {
    const IndicatorParam& param = nnvm::get<IndicatorParam>(n->attrs.parsed);
    std::ostringstream axis;
    axis << param.axis;

    if (param.axis.ndim() == 0 && !param.exclude) {
      // Special case needed because sum interprets axis=[] differently
      return std::vector<NodeEntry>{
        ograds[0],
        MakeNode("zeros_like", n->attrs.name + "_zero_grad", {n->inputs[1]})
      };
    }

    auto sum_node =
      MakeNode("sum", n->attrs.name + "_sum_grad",
               {ograds[0]},
               {{"axis", axis.str()},
                {"exclude", std::to_string(param.exclude)}});

    return std::vector<NodeEntry>{
      MakeNode("reshape_like", n->attrs.name + "_grad",
               {sum_node, n->inputs[0]}),
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

  auto axis = param.axis;
  if (axis < 0) {
    axis += dshape.ndim();
  }
  CHECK_LT(axis, dshape.ndim())
    << "axis should be within input dimension range but got " <<  axis;
  CHECK_GT(axis, -1)
    << "axis should be within input dimension range but got " <<  axis;

  if (param.equal_split) {
    int num_outputs = param.indices_or_sections[0];
    CHECK_EQ(out_shape->size(), static_cast<size_t>(num_outputs));
    TShape oshape = dshape;
    CHECK_EQ(oshape[axis] % num_outputs, 0)
        << "indices_or_sections need to be able to divide input.shape[axis] got sections "
        << num_outputs << " and dimension " << oshape[axis];
    oshape[axis] /= num_outputs;

    for (size_t i = 0; i < out_shape->size(); ++i) {
      NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, i, oshape);
    }
  } else {
    dim_t num_outputs = param.indices_or_sections.ndim() + 1;
    CHECK_EQ(out_shape->size(), static_cast<size_t>(num_outputs));
    TShape oshape = dshape;
    dim_t begin = 0;
    for (dim_t i = 0; i < num_outputs - 1; ++i) {
      CHECK_GT(param.indices_or_sections[i], begin)
          << "indices_or_sections need to be a sorted ascending list got "
          << param.indices_or_sections;
      oshape[axis] = param.indices_or_sections[i] - begin;
      begin = param.indices_or_sections[i];
      NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, i, oshape);
    }
    CHECK_LT(begin, dshape[axis])
        << "The sum of sections must match the input.shape[axis]";
    oshape[axis] = dshape[axis] - begin;
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
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, -1>)
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
      Array<Integer> indices;
      for (auto i : param.indices_or_sections) {
        indices.push_back(static_cast<int>(i));
      }
      return Array<Tensor>{ topi::split(inputs[0], indices, param.axis) };
    }
})
.set_support_level(3);

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
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseArbitraryLayout<1, 1>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const CastParam& param = nnvm::get<CastParam>(attrs.parsed);
    Type dtype = GetTVMType(param.dtype);
    return Array<Tensor>{ topi::cast(inputs[0], dtype) };
})
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
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
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

inline bool ReshapeLikeInferType(const NodeAttrs &attrs,
                                 std::vector<int> *in_attrs,
                                 std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  NNVM_ASSIGN_OUTPUT_TYPE(attrs, *out_attrs, 0, (*in_attrs)[0]);
  return true;
}

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
.set_attr<FInferType>("FInferType", ReshapeLikeInferType)
// never transform layout of the second input array.
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
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
      int real_axis;
      if (param.axis[i] < 0) {
        real_axis = param.axis[i] + static_cast<int>(shp.ndim());
      } else {
        real_axis = param.axis[i];
      }
      CHECK(real_axis < static_cast<int>(shp.ndim()) && real_axis >= 0);
      axis_checker.insert(real_axis);
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
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const SqueezeParam& param = nnvm::get<SqueezeParam>(attrs.parsed);
    auto axis = ShapeToIntArray(param.axis);
    return Array<Tensor>{ topi::squeeze(inputs[0], axis, true) };
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

// transpose
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

inline bool TransposeCorrectLayout(const NodeAttrs& attrs,
                                   std::vector<Layout> *ilayouts,
                                   const std::vector<Layout> *last_ilayouts,
                                   std::vector<Layout> *olayouts) {
  const TransposeParam& param = nnvm::get<TransposeParam>(attrs.parsed);
  CHECK_EQ(ilayouts->size(), 1U);
  CHECK_EQ(olayouts->size(), 1U);

  const Layout& input = last_ilayouts->at(0).defined()
                        ? last_ilayouts->at(0)
                        : ilayouts->at(0);

  NNVM_ASSIGN_LAYOUT(*ilayouts, 0, input);

  if (input.defined()) {
    std::ostringstream new_layout;
    if (param.axes.ndim() == 0) {
      for (size_t i = 0; i < input.ndim(); ++i) {
        new_layout << input.at(input.ndim() - 1 - i);
      }
    } else {
      CHECK_EQ(input.ndim(), param.axes.ndim());
      for (size_t i = 0; i < input.ndim(); ++i) {
        CHECK(param.axes[i] < static_cast<int>(input.ndim()));
        new_layout << input.at(param.axes[i]);
      }
    }
    NNVM_ASSIGN_LAYOUT(*olayouts, 0, Layout(new_layout.str()));
  }

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
.set_attr<FCorrectLayout>("FCorrectLayout", TransposeCorrectLayout)
.set_num_inputs(1)
.set_num_outputs(1)
.set_support_level(4)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const TransposeParam& param = nnvm::get<TransposeParam>(attrs.parsed);
    auto axes = ShapeToIntArray(param.axes);
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

// strided_slice
DMLC_REGISTER_PARAMETER(StridedSliceParam);

inline void StridedSliceParamParser(nnvm::NodeAttrs* attrs) {
  StridedSliceParam param;
  param.Init(attrs->dict);
  attrs->parsed = std::move(param);
}

inline bool StridedSliceInferShape(const NodeAttrs& attrs,
                            std::vector<TShape>* in_shape,
                            std::vector<TShape>* out_shape) {
  const StridedSliceParam& param = nnvm::get<StridedSliceParam>(attrs.parsed);
  const TShape& dshape = (*in_shape)[0];
  if (dshape.ndim() == 0) return false;
  TShape oshape = dshape;
  dim_t num_axis = dshape.ndim();

  std::vector<int64_t> begin_vec;
  std::copy(param.begin.begin(), param.begin.end(), std::back_inserter(begin_vec));
  for (dim_t i = begin_vec.size(); i < num_axis; ++i) {
    begin_vec.push_back(0);
  }

  std::vector<int64_t> end_vec;
  std::copy(param.end.begin(), param.end.end(), std::back_inserter(end_vec));
  for (dim_t i = end_vec.size(); i < num_axis; ++i) {
    end_vec.push_back(dshape[i]);
  }

  std::vector<int64_t> stride_vec;
  std::copy(param.stride.begin(), param.stride.end(), std::back_inserter(stride_vec));
  for (dim_t i = stride_vec.size(); i < num_axis; ++i) {
    stride_vec.push_back(1);
  }

  for (dim_t i = 0; i < num_axis; ++i) {
      int64_t begin_range = stride_vec[i] < 0 ? -1 : 0;
      int64_t end_range = stride_vec[i] < 0 ? dshape[i] - 1 : dshape[i];
      int64_t begin = begin_vec[i] < 0 ? dshape[i] + begin_vec[i] : begin_vec[i];
      int64_t end = end_vec[i] < 0 ? dshape[i] + end_vec[i] : end_vec[i];
      begin = std::min(std::max(begin, begin_range), end_range);
      end = std::min(std::max(end, begin_range), end_range);

      int interval = std::abs(end - begin);
      int slice_size = static_cast<int>((interval
                                       + std::abs(stride_vec[i]) - 1) / std::abs(stride_vec[i]));
      CHECK(stride_vec[i] < 0 ? (end < begin) : (begin < end))
        << ": Input [Begin=" << begin_vec[i] << ", End=" << end_vec[i]
        << "] is invalid for axis=" << i;
      oshape[i] = slice_size;
  }
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  return true;
}

NNVM_REGISTER_OP(strided_slice)
.describe(R"code(Strided slice of an array.

Examples::

  x = [[  1.,   4.,   7.,  10.],
       [  2.,   5.,   8.,  11.],
       [  3.,   6.,   9.,  12.]]

  strided_slice(x, begin=[0, 1], end=[2, 4], stride=[1, 1]) = [[ 4.,  7.,  10.],
                                                               [ 5.,  8.,  11.]]

  x = [[[ 1.,  2.],
        [ 3.,  4.]],

       [[ 5.,  6.],
        [ 7.,  8.]]]

  strided_slice(x, begin=[0, 0], end=[2, 2]) = [[[ 1.,  2.],
                                                 [ 3.,  4.]],

                                                [[ 5.,  6.],
                                                 [ 7.,  8.]]]
)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Array to be sliced")
.add_arguments(StridedSliceParam::__FIELDS__())
.set_attr_parser(StridedSliceParamParser)
.set_attr<FInferShape>("FInferShape", StridedSliceInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseArbitraryLayout<1, 1>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const StridedSliceParam& param = nnvm::get<StridedSliceParam>(attrs.parsed);
    Array<Integer> begin;
    Array<Integer> end;
    Array<Integer> stride;

    for (int64_t i : param.begin) {
      begin.push_back(static_cast<int>(i));
    }

    for (int64_t i : param.end) {
      end.push_back(static_cast<int>(i));
    }

    for (int64_t i : param.stride) {
      stride.push_back(static_cast<int>(i));
    }

    return Array<Tensor>{
      topi::strided_slice(inputs[0], begin, end, stride)
    };
})
.set_support_level(1);

// Flip
DMLC_REGISTER_PARAMETER(FlipParam);

NNVM_REGISTER_OP(flip)
.describe(R"code(Reverse the elements of an array.

Examples::

  x = [[ 1, 2],
       [ 3, 4]]

  flip(x) = [[ 3.,  4.],
             [ 1.,  2.]]

  x = [[[ 1.,  2.],
        [ 3.,  4.]],

       [[ 5.,  6.],
        [ 7.,  8.]]]

  flip(x) = [[[ 5.,  6.],
              [ 7.,  8.]],

             [[ 1.,  2.],
              [ 3.,  4.]]]

  flip(x, axis=1) = [[[ 3.,  4.],
                      [ 1.,  2.]],

                     [[ 7.,  8.],
                      [ 5.,  6.]]]
)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Source input")
.add_arguments(FlipParam::__FIELDS__())
.set_attr_parser(ParamParser<FlipParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<FlipParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_support_level(4)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const FlipParam& param = nnvm::get<FlipParam>(attrs.parsed);
    return Array<Tensor>{ topi::flip(inputs[0], param.axis) };
});


// take
DMLC_REGISTER_PARAMETER(TakeParam);

inline bool TakeInferShape(const NodeAttrs& attrs,
                           std::vector<TShape>* in_shape,
                           std::vector<TShape>* out_shape) {
  CHECK_EQ(in_shape->size(), 2U);
  CHECK_EQ(out_shape->size(), 1U);
  const TShape& dshape = (*in_shape)[0];
  const TShape& indicesshape = (*in_shape)[1];
  if (dshape.ndim() == 0) return false;
  if (indicesshape.ndim() == 0) return false;

  const TakeParam& param = nnvm::get<TakeParam>(attrs.parsed);
  TShape oshape((!param.axis ? 0: dshape.ndim() - 1) + indicesshape.ndim());
  if (!param.axis) {
    for (size_t j = 0; j < indicesshape.ndim(); ++j) {
      oshape[j] = indicesshape[j];
    }
  } else {
    int axis = param.axis.value();
    if (axis < 0) {
      axis += dshape.ndim();
    }
    CHECK_LT(axis, dshape.ndim());

    size_t posi = 0;
    for (size_t i = 0; i < dshape.ndim(); ++i) {
      if (static_cast<int>(i) == axis) {
        for (size_t j = 0; j < indicesshape.ndim(); ++j) {
          oshape[posi++] = indicesshape[j];
        }
      } else {
        oshape[posi++] = dshape[i];
      }
    }
  }
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, 0, dshape);
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, 1, indicesshape);
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  return dshape.Size() != 0;
}

inline bool TakeInferType(const NodeAttrs& attrs,
                          std::vector<int>* in_attrs,
                          std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK_EQ((*in_attrs)[1], kInt32);
  NNVM_ASSIGN_INPUT_TYPE(attrs, *in_attrs, 0, (*in_attrs)[0]);
  NNVM_ASSIGN_INPUT_TYPE(attrs, *in_attrs, 1, static_cast<int>(kInt32));
  NNVM_ASSIGN_OUTPUT_TYPE(attrs, *out_attrs, 0, (*in_attrs)[0]);
  return true;
}

inline bool TakeCorrectLayout(const NodeAttrs& attrs,
                              std::vector<Layout> *ilayouts,
                              const std::vector<Layout> *last_ilayouts,
                              std::vector<Layout> *olayouts) {
  CHECK_EQ(ilayouts->size(), last_ilayouts->size());
  CHECK_EQ(olayouts->size(), 1U);

  for (size_t i = 0; i < ilayouts->size(); ++i) {
    const Layout& input = last_ilayouts->at(i).defined() ?
                          last_ilayouts->at(i) : ilayouts->at(i);
    NNVM_ASSIGN_LAYOUT(*ilayouts, i, input);
  }

  return true;
}

NNVM_REGISTER_OP(take)
.describe(R"code(Take elements from an array along an axis.

When axis is not None, this function does the same thing as 'fancy' indexing
(indexing arrays using arrays); however, it can be easier to use if you need
elements along a given axis.

**Note** that when axis is none the flattened input array is used.

Examples::

  a = [[ 1, 2],
       [ 3, 4]]
  indices = [3, 0, 2]
  take(a, indices) = [ 4, 1, 3]

  a = [[ 1., 2.],
       [ 3., 4.]]
  indices = [1, 0]
  take(a, indices, axis=1) = [[ 2., 1.],
                              [ 4., 3.]]

  )code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Array to be indexed")
.add_argument("indices", "Tensor", "The indices of the values to extract")
.add_arguments(TakeParam::__FIELDS__())
.set_attr_parser(ParamParser<TakeParam>)
.set_attr<FInferShape>("FInferShape", TakeInferShape)
.set_attr<FInferType>("FInferType", TakeInferType)
.set_attr<FCorrectLayout>("FCorrectLayout", TakeCorrectLayout)
.set_num_inputs(2)
.set_num_outputs(1)
.set_support_level(3)
.set_attr<FTVMCompute>(
    "FTVMCompute", [](const NodeAttrs& attrs,
                      const Array<Tensor>& inputs,
                      const Array<Tensor>& out_info) {
      const TakeParam& param = nnvm::get<TakeParam>(attrs.parsed);
      if (!param.axis) {
        return Array<Tensor>{
            topi::take(inputs[0], inputs[1]) };
      } else {
        return Array<Tensor>{
            topi::take(inputs[0], inputs[1], param.axis.value()) };
      }
  });


// SliceLike
DMLC_REGISTER_PARAMETER(SliceLikeParam);

inline bool SliceLikeShape(const nnvm::NodeAttrs& attrs,
                           std::vector<TShape>* in_attrs,
                           std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const SliceLikeParam& param = nnvm::get<SliceLikeParam>(attrs.parsed);
  const TShape& src_shape = in_attrs->at(0);
  const TShape& target_shape = in_attrs->at(1);
  Tuple<dim_t> end_idx;
  end_idx = Tuple<dim_t>(src_shape);
  if (param.axis.ndim() == 0) {
    for (size_t i = 0; i < src_shape.ndim(); ++i) {
      if (i < target_shape.ndim()) {
        end_idx[i] = target_shape[i];
        CHECK_LE(end_idx[i], src_shape[i])
          << "End index of axis " << i << " exceeds input shape: "
          << end_idx[i] << " vs " << src_shape[i];
      }
    }
  } else {
    for (auto i : param.axis) {
      if (i < 0) {
        i = src_shape.ndim() + i;
      }
      CHECK_LT(i, target_shape.ndim())
        << "Axis " << i << " exceeds dimension "
        << target_shape.ndim()<< " of target_shape.";
      end_idx[i] = target_shape[i];
      CHECK_LE(end_idx[i], src_shape[i])
        << "End index of axis " << i << " exceeds input shape: "
        << end_idx[i] << " vs " << src_shape[i];
    }
  }
  TShape out_shape = TShape(std::move(end_idx));
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, out_shape);
  return true;
}

// Adapter function to make int array.
Array<Integer> GetIntArray(Array<Expr> arr) {
  for (size_t i = 0; i < arr.size(); ++i) {
    CHECK(!arr[i].defined() || arr[i].as<IntImm>())
        << "Expect an int array";
  }
  return Array<Integer>(arr.node_);
}

NNVM_REGISTER_OP(slice_like)
.describe(R"code(Slice the first input respect to the second input.
)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input data to be sliced.")
.add_argument("slice_like", "Tensor", "Tensor with target shape")
.set_num_inputs(2)
.set_num_outputs(1)
.add_arguments(SliceLikeParam::__FIELDS__())
.set_attr_parser(ParamParser<SliceLikeParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<SliceLikeParam>)
.set_attr<FInferShape>("FInferShape", SliceLikeShape)
.set_attr<FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseBinaryKeepLeftLayout)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const auto& param = nnvm::get<SliceLikeParam>(attrs.parsed);
    Array<Expr> src_shape = inputs[0]->shape;
    Array<Expr> target_shape = inputs[1]->shape;
    Array<Expr> begin_idx, end_idx, strides;
    for (size_t i = 0; i < src_shape.size(); ++i) {
      begin_idx.push_back(make_const(tvm::Int(32), 0));
      strides.push_back(make_const(tvm::Int(32), 1));
    }
    end_idx = Array<Expr>(src_shape);
    if (param.axis.ndim() == 0) {
      for (size_t i = 0; i < src_shape.size(); ++i) {
        if (i < target_shape.size()) {
          end_idx.Set(i, target_shape[i]);
          CHECK_LE(topi::GetConstInt(end_idx[i]),
                   topi::GetConstInt(src_shape[i]))
            << "End index of axis " << i << " exceeds input shape: "
            << topi::GetConstInt(end_idx[i]) << " vs "
            << topi::GetConstInt(src_shape[i]);
        }
      }
    } else {
      for (int axis : param.axis) {
        if (axis < 0) {
          axis = static_cast<int>(src_shape.size()) + axis;
        }
        end_idx.Set(static_cast<size_t>(axis), target_shape[axis]);
        CHECK_LE(topi::GetConstInt(end_idx[axis]),
                 topi::GetConstInt(src_shape[axis]))
          << "End index of axis " << axis << " exceeds input shape: "
          << topi::GetConstInt(end_idx[axis]) << " vs "
          << topi::GetConstInt(src_shape[axis]);
      }
    }
    return Array<Tensor>{
      topi::strided_slice(inputs[0],
                          GetIntArray(begin_idx),
                          GetIntArray(end_idx),
                          GetIntArray(strides))
    };
})
.set_attr<FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "slice_like"};
})
.set_support_level(4);

// where
inline bool WhereShape(const nnvm::NodeAttrs& attrs,
                       std::vector<TShape>* in_attrs,
                       std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 1U);
  const TShape& cond_shape = in_attrs->at(0);
  const TShape& x_shape = in_attrs->at(1);
  const TShape& y_shape = in_attrs->at(2);
  CHECK_EQ(x_shape, y_shape) << "x and y must have the same shape: "
                             << x_shape << " vs " << y_shape;
  if (cond_shape != x_shape) {
    CHECK_EQ(cond_shape.ndim(), 1)
      << "Shape of condition " << cond_shape
      << " must be either equal to x or has dimension of 1.";
  }
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, x_shape);
  return true;
}

inline bool WhereInferType(const NodeAttrs &attrs,
                           std::vector<int> *in_attrs,
                           std::vector<int> *out_attrs) {
  DTYPE_ASSIGN(out_attrs->at(0), in_attrs->at(1));
  return true;
}

inline bool WhereCorrectLayout(const NodeAttrs& attrs,
                               std::vector<Layout> *ilayouts,
                               const std::vector<Layout> *last_ilayouts,
                               std::vector<Layout> *olayouts) {
  CHECK_EQ(ilayouts->size(), last_ilayouts->size());
  CHECK_EQ(olayouts->size(), 1U);

  for (size_t i = 0; i < ilayouts->size(); ++i) {
    const Layout& input = last_ilayouts->at(i).defined() ?
                          last_ilayouts->at(i) : ilayouts->at(i);
    NNVM_ASSIGN_LAYOUT(*ilayouts, i, input);
  }

  return true;
}

NNVM_REGISTER_OP(where)
.describe(R"code(
Return the elements, either from x or y, depending on the condition.

Given three ndarrays, condition, x, and y, return an ndarray with the elements
from x or y, depending on the elements from condition are true or false.
x and y must have the same shape. If condition has the same shape as x,
each element in the output array is from x if the corresponding element
in the condition is true, and from y if false.

If condition does not have the same shape as x, it must be a 1D array whose
size is the same as x’s first dimension size. Each row of the output array
is from x’s row if the corresponding element from condition is true, and
from y’s row if false.

Note that all non-zero values are interpreted as True in condition.

Examples::

  x = [[1, 2], [3, 4]]
  y = [[5, 6], [7, 8]]
  cond = [[0, 1], [-1, 0]]
  where(cond, x, y) = [[5, 2], [3, 8]]


  cond = [1, 0]
  where(cond, x, y) = [[1, 2], [7, 8]]

)code" NNVM_ADD_FILELINE)
.add_argument("condition", "Tensor", "Condition array")
.add_argument("x", "Tensor", "First array to be selected")
.add_argument("y", "Tensor", "Second array to be selected")
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<FInferShape>("FInferShape", WhereShape)
.set_attr<FInferType>("FInferType", WhereInferType)
.set_attr<FCorrectLayout>("FCorrectLayout", WhereCorrectLayout)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    return Array<Tensor>{
      topi::where(inputs[0], inputs[1], inputs[2])
    };
  })
.set_attr<FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"condition", "x", "y"};
})
.set_support_level(4);

// gather_nd
inline bool GatherNDInferShape(const nnvm::NodeAttrs& attrs,
                               std::vector<TShape>* in_attrs,
                               std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const TShape& data_shape = in_attrs->at(0);
  const TShape& indices_shape = in_attrs->at(1);
  CHECK_GT(indices_shape.ndim(), 1) << "indices must have at least 2 dimensions";
  CHECK_LE(indices_shape[0], data_shape.ndim()) <<
      "dim 0 of indices must be no more than rank of data";
  std::vector<dim_t> oshape;
  for (size_t i = 1; i < indices_shape.ndim(); ++i) {
    oshape.push_back(indices_shape[i]);
  }
  for (size_t i = indices_shape[0]; i < data_shape.ndim(); ++i) {
    oshape.push_back(data_shape[i]);
  }
  if (oshape.size() == 0) {
    oshape.push_back(1);
  }
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0,
                           TShape(oshape.begin(), oshape.end()));
  return true;
}

inline bool GatherNDInferType(const NodeAttrs &attrs,
                              std::vector<int> *in_attrs,
                              std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  NNVM_ASSIGN_OUTPUT_TYPE(attrs, *out_attrs, 0, (*in_attrs)[0]);
  return true;
}

inline bool GatherNDCorrectLayout(const NodeAttrs& attrs,
                                  std::vector<Layout> *ilayouts,
                                  const std::vector<Layout> *last_ilayouts,
                                  std::vector<Layout> *olayouts) {
  CHECK_EQ(ilayouts->size(), last_ilayouts->size());
  CHECK_EQ(olayouts->size(), 1U);

  for (size_t i = 0; i < ilayouts->size(); ++i) {
    const Layout& input = last_ilayouts->at(i).defined() ?
                          last_ilayouts->at(i) : ilayouts->at(i);
    NNVM_ASSIGN_LAYOUT(*ilayouts, i, input);
  }

  return true;
}

NNVM_REGISTER_OP(gather_nd)
.describe(R"code(
Gather elements or slices from ``data`` into a tensor specified by ``indices``.

The shape of output tensor is inferred from ``indices``. Given ``data`` with
shape ``(X0, X1, ..., X_{N-1})`` and ``indices`` with shape ``(Y_0, ...,
Y_{M-1})``, the output will have shape ``(Y_1, ..., Y_{M-1}, X_{Y_0}, ...,
X_{N-1})`` when ``Y_0 < N``, or ``(Y_1, ..., Y_{M-1})`` when ``Y_0 == N``. The
operator is invalid when ``Y_0 > N``.

The element in output is defined as follows::

  output[y_1, ..., y_{M-1}, x_{Y_0}, ..., x_{N-1}] = data[indices[0, y_1, ..., y_{M-1}],
                                                     ...,
                                                     indices[Y_0-1, y_1, ..., y_{M-1}],
                                                     x_{Y_0}, ..., x_{N-1}]

Examples::

  data = [[0, 1], [2, 3]]
  indices = [[1], [0]]
  gather_nd(data, indices) = [2]

  data = [[0, 1], [2, 3]]
  indices = [[1, 1, 0], [0, 1, 0]]
  gather_nd(data, indices) = [2, 3, 0]

  data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
  indices = [[0, 1], [1, 0]]
  gather_nd(data, indices) = [[3, 4], [5, 6]]

)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input data.")
.add_argument("indices", "Tensor", "Indices of data")
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<FInferShape>("FInferShape", GatherNDInferShape)
.set_attr<FInferType>("FInferType", GatherNDInferType)
.set_attr<FCorrectLayout>("FCorrectLayout", GatherNDCorrectLayout)
.set_attr<FTVMCompute>(
    "FTVMCompute", [](const NodeAttrs& attrs,
                      const Array<Tensor>& inputs,
                      const Array<Tensor>& out_info) {
      return Array<Tensor>{
        topi::gather_nd(inputs[0], inputs[1]) };
  })
.set_attr<FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data", "indices"};
})
.set_support_level(3);

}  // namespace top
}  // namespace nnvm
