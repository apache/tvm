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
 * \file nms.cc
 * \brief Property def of SSD non-maximum suppression operator.
 */

#include <tvm/expr.h>
#include <tvm/packed_func_ext.h>
#include <nnvm/op.h>
#include <nnvm/top/nn.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/compiler/op_attr_types.h>
#include "../op_common.h"
#include "../elemwise_op_common.h"

namespace nnvm {
namespace top {
using compiler::FTVMCompute;
using tvm::Tensor;
using tvm::Array;

DMLC_REGISTER_PARAMETER(NonMaximumSuppressionParam);

bool NMSShape(const NodeAttrs& attrs,
              std::vector<TShape> *in_attrs,
              std::vector<TShape> *out_attrs) {
  const NonMaximumSuppressionParam& param =
    nnvm::get<NonMaximumSuppressionParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 2U) << "Inputs: [data, valid_count]";
  TShape dshape = in_attrs->at(0);
  TShape vshape = in_attrs->at(1);
  CHECK_EQ(dshape.ndim(), 3U) << "Input data should be 3-D.";
  CHECK_EQ(vshape.ndim(), 1U) << "Input valid count should be 1-D.";
  CHECK_EQ(dshape[2], 6U) << "Data input should have shape "
    "(batch_size, num_anchors, 6).";
  CHECK_EQ(dshape[0], vshape[0]) << "batch_size mismatch.";
  out_attrs->clear();
  if (param.return_indices) {
    TShape oshape = TShape(2);
    oshape[0] = dshape[0];
    oshape[1] = dshape[1];
    NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, oshape);
  } else {
    NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, dshape);
  }
  return true;
}

inline bool NMSInferType(const NodeAttrs &attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  DTYPE_ASSIGN(out_attrs->at(0), in_attrs->at(0));
  return true;
}

inline bool NMSInferLayout(const NodeAttrs& attrs,
                           std::vector<Layout> *ilayouts,
                           const std::vector<Layout> *last_ilayouts,
                           std::vector<Layout> *olayouts) {
  static const Layout kNCHW("NCHW");
  CHECK_EQ(ilayouts->size(), 2U);
  CHECK_EQ(olayouts->size(), 1U);
  NNVM_ASSIGN_LAYOUT(*ilayouts, 0, kNCHW);
  NNVM_ASSIGN_LAYOUT(*ilayouts, 1, kNCHW);
  return true;
}

NNVM_REGISTER_OP(non_max_suppression)
  .describe(R"doc("Non-maximum suppression."
)doc" NNVM_ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NonMaximumSuppressionParam>)
.set_attr<FGetAttrDict>("FGetAttrDict",
                        ParamGetAttrDict<NonMaximumSuppressionParam>)
.add_arguments(NonMaximumSuppressionParam::__FIELDS__())
.add_argument("data", "Tensor", "Input data.")
.add_argument("valid_count", "Tensor", "Number of valid anchor boxes.")
.set_attr<FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data", "valid_count"};
})
.set_attr<FInferShape>("FInferShape", NMSShape)
.set_attr<FInferType>("FInferType", NMSInferType)
.set_attr<FCorrectLayout>("FCorrectLayout", NMSInferLayout)
.set_support_level(4);

}  // namespace top
}  // namespace nnvm

