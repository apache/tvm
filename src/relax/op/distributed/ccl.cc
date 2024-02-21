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
#include "tvm/relax/attrs/ccl.h"

#include "utils.h"

namespace tvm {
namespace relax {
namespace distributed {

StructInfo InferDistStructInfoAllReduce(const Call& call, const BlockBuilder& ctx) {
  Array<DTensorStructInfo> input_dtensor_sinfos = GetInputDTensorStructInfo(call, ctx);
  ICHECK(input_dtensor_sinfos.size() == 1);
  DTensorStructInfo input_dtensor_sinfo = input_dtensor_sinfos[0];
  TensorStructInfo tensor_sinfo = input_dtensor_sinfo->tensor_sinfo;
  DeviceMesh device_mesh = input_dtensor_sinfo->device_mesh;
  // FIXME: this is a hack where there's only 1d mesh
  return DTensorStructInfo(tensor_sinfo, device_mesh,
                           Placement::FromText(std::string(device_mesh->shape.size(), 'R')));
}

TVM_REGISTER_OP("relax.ccl.allreduce")
    .set_attr<FInferStructInfo>("dist.FInferStructInfo", InferDistStructInfoAllReduce);

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
