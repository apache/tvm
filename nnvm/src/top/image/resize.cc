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
 * \file resize.cc
 * \brief Property def of resize operators.
 */
#include <tvm/operation.h>
#include <tvm/expr.h>
#include <tvm/packed_func_ext.h>
#include <nnvm/layout.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include "../nn/nn_common.h"
#include "../op_common.h"
#include "../elemwise_op_common.h"
#include "topi/elemwise.h"
#include "topi/transform.h"
#include "topi/image/resize.h"
#include "resize.h"

namespace nnvm {
namespace top {
using tvm::Expr;
using tvm::Array;
using tvm::Tensor;
using nnvm::compiler::FTVMCompute;

DMLC_REGISTER_PARAMETER(ResizeParam);

inline bool ResizeInferShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape>* in_shape,
                             std::vector<TShape>* out_shape) {
  static const Layout kNCHW("NCHW");
  const ResizeParam& param = nnvm::get<ResizeParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 1U);
  CHECK_EQ(out_shape->size(), 1U);
  TShape dshape = (*in_shape)[0];
  if (dshape.ndim() ==  0) return false;
  dshape = ConvertLayout(dshape, param.layout, kNCHW);

  TShape oshape = dshape;
  oshape[2] = param.size[0];
  oshape[3] = param.size[1];

  oshape = ConvertLayout(oshape, kNCHW, param.layout);
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);

  return true;
}

inline bool ResizeLayout(const NodeAttrs& attrs,
                         std::vector<Layout> *in_layouts,
                         const std::vector<Layout> *last_in_layouts,
                         std::vector<Layout> *out_layouts) {
  const ResizeParam& param = nnvm::get<ResizeParam>(attrs.parsed);
  CHECK_EQ(in_layouts->size(), 1U);
  CHECK_EQ(out_layouts->size(), 1U);
  const Layout layout(param.layout);
  NNVM_ASSIGN_LAYOUT(*in_layouts, 0, layout);
  NNVM_ASSIGN_LAYOUT(*out_layouts, 0, layout);
  return true;
}

NNVM_REGISTER_OP(resize)
.describe(R"(Perform resize to input array with nearest neighbour or bilinear interpolation.

- **data**: data is 4D array of shape
            (batch_size, channels, in_height, in_width) for NCHW
            (batch_size, in_height, in_width, channels) for NHWC

- **out**: Output is 4D array of shape
           for layout NCHW
           (batch_size, channels, size[0], size[1])

           for layout NHWC
           (batch_size, size[0], size[1], channels)

)" NNVM_ADD_FILELINE)
.add_argument("data", "4D Tensor", "Input data.")
.add_arguments(ResizeParam::__FIELDS__())
.set_attr_parser(ParamParser<ResizeParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<ResizeParam>)
.set_attr<FInferShape>("FInferShape", ResizeInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ResizeLayout)
.set_num_outputs(1)
.set_num_inputs(1)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
  const ResizeParam& param = nnvm::get<ResizeParam>(attrs.parsed);
  Array<Expr> oshape;
  if (param.layout == "NCHW") {
    oshape.push_back(out_info[0]->shape[2]);
    oshape.push_back(out_info[0]->shape[3]);
  } else {
    oshape.push_back(out_info[0]->shape[1]);
    oshape.push_back(out_info[0]->shape[2]);
  }

  return Array<Tensor>{ topi::image::resize(inputs[0], oshape, param.layout,
                                             param.align_corners, param.method)};
})
.set_support_level(2);

}  // namespace top
}  // namespace nnvm
