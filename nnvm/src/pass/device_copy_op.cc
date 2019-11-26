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

#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>

#include "../top/elemwise_op_common.h"
#include "../top/op_common.h"

namespace nnvm {
namespace op {

NNVM_REGISTER_OP(device_copy_op)
.describe(R"code(
Copy data from one tensor to another. The source and destination might be
on different devices.
)code" NNVM_ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", nnvm::top::ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", nnvm::top::ElemwiseType<1, 1>)
.set_attr<FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<nnvm::FCorrectLayout>(
  "FCorrectLayout", nnvm::top::ElemwiseArbitraryLayout<1, 1>);

}  // namespace op
}  // namespace nnvm
