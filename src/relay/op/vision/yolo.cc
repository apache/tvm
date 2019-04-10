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
 *  Copyright (c) 2018 by Contributors
 * \file yolo.cc
 * \brief Yolo related operators
 */
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/vision.h>
#include <topi/vision/reorg.h>
#include <vector>
#include "../op_common.h"
#include "../type_relations.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(YoloReorgAttrs);

/*!
* \brief YoloReorgRel Output type and shape relation evaluation function.
* \param num_inputs Number of input types in the args.
* \param attrs The additional attributes of the operator.
* \param reporter The reporter to report solution to.
* \return false if This relation cannot be resolved. true if this relation has been resolved.
*/
bool YoloReorgRel(const Array<Type>& types,
                  int num_inputs,
                  const Attrs& attrs,
                  const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  const YoloReorgAttrs* param = attrs.as<YoloReorgAttrs>();
  CHECK(param != nullptr);

  CHECK(data->shape.size() == 4) << "Yolo reorg supports only 4 dimension.";
  std::vector<IndexExpr>&& oshape = AsVector(data->shape);
  oshape[1] = oshape[1] * param->stride * param->stride;
  oshape[2] = oshape[2] / param->stride;
  oshape[3] = oshape[3] / param->stride;
  reporter->Assign(types[1], TensorTypeNode::make(oshape, data->dtype));
  return true;
}

Expr MakeYoloReorg(Expr data,
                   Integer stride) {
  auto attrs = make_node<YoloReorgAttrs>();
  attrs->stride = stride;
  static const Op& op = Op::Get("vision.yolo_reorg");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op.vision._make.yolo_reorg")
.set_body_typed(MakeYoloReorg);


RELAY_REGISTER_OP("vision.yolo_reorg")
.describe(R"doc("Yolo reorg operation. This layer reorganize the output.
Its function is mostly shape transform.")doc" TVM_ADD_FILELINE)
.add_argument("data", "Tensor", "The input tensor.")
.set_num_inputs(1)
.set_support_level(5)
.set_attrs_type_key("relay.attrs.YoloReorgAttrs")
.add_type_rel("YoloReorg", YoloReorgRel)
.set_attr<FTVMCompute>("FTVMCompute", [](const Attrs& attrs,
                                         const Array<Tensor>& inputs,
                                         const Type& out_type,
                                         const Target& target) {
  const auto* params = attrs.as<YoloReorgAttrs>();
  CHECK(params != nullptr);
  return Array<Tensor>{ topi::vision::reorg(inputs[0], params->stride) };
});

}  // namespace relay
}  // namespace tvm
