
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

#include "../../../../target/parsers/cpu.h"
#include "compiler_attrs.h"

namespace tvm {

namespace relay {
namespace contrib {
namespace cmsisnn {

tvm::transform::Pass RelayToTIR();
runtime::Module TIRToRuntime(IRModule mod, Target target);
using FTVMTIRToRuntime = tvm::runtime::TypedPackedFunc<runtime::Module(IRModule, Target)>;

TVM_REGISTER_TARGET_KIND("cmsis-nn", kDLCPU)
    .add_attr_option<Array<String>>("mattr")
    .add_attr_option<String>("mcpu")
    .add_attr_option<Bool>("debug_last_error")
    .set_attr<relay::transform::FTVMRelayToTIR>(tvm::attr::kRelayToTIR, RelayToTIR())
    .set_attr<FTVMTIRToRuntime>("TIRToRuntime", TIRToRuntime)
    .set_target_parser(tvm::target::parsers::cpu::ParseTarget);

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
