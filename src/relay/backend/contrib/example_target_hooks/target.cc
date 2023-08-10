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

#include <tvm/relay/transform.h>
#include <tvm/target/target.h>

namespace tvm {

using FTVMTIRToRuntime = tvm::runtime::TypedPackedFunc<runtime::Module(IRModule, Target)>;

namespace relay {
namespace contrib {
namespace example_target_hooks {
tvm::transform::Pass RelayToTIR();
runtime::Module TIRToRuntime(IRModule mod, Target target);
}  // namespace example_target_hooks
}  // namespace contrib
}  // namespace relay

TVM_REGISTER_TARGET_KIND("example_target_hook", kDLCPU)
    .set_attr<Bool>("use_device_api", Bool(true))
    .set_attr<relay::transform::FTVMRelayToTIR>(attr::kRelayToTIR,
                                                relay::contrib::example_target_hooks::RelayToTIR())
    .set_attr<FTVMTIRToRuntime>("TIRToRuntime", relay::contrib::example_target_hooks::TIRToRuntime)
    .add_attr_option<Integer>("example_attribute", Integer(0));

}  // namespace tvm
