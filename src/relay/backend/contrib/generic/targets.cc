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
 * \file relay/backend/contrib/generic/codegen.cc
 *
 * \brief this file contains the targets for generic scale4edge codegen.
 */

#include <tvm/relay/transform.h>
#include <tvm/target/target.h>

namespace tvm {

namespace relay {
namespace contrib {
namespace generic {
    tvm::transform::Pass RelayToTIR(String target_name);
}  // namespace generic
}  // namespace contrib
}  // namespace relay

TVM_REGISTER_TARGET_KIND("ultra_trail", kDLCPU)
    .set_attr<FTVMRelayToTIR>("RelayToTIR", relay::contrib::generic::RelayToTIR("ultra_trail"));

TVM_REGISTER_TARGET_KIND("rb_npu", kDLCPU)
    .set_attr<FTVMRelayToTIR>("RelayToTIR", relay::contrib::generic::RelayToTIR("rb_npu"));

}  // namespace tvm
