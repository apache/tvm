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
 * \file relay/backend/contrib/uma/targets.cc
 *
 * \brief this file contains the targets for the Universal Modular Accelerator Interface (UMA).
 */

#include <tvm/relay/transform.h>
#include <tvm/target/target.h>

namespace tvm {

using FTVMTIRToRuntime = tvm::runtime::TypedPackedFunc<runtime::Module(IRModule, Target)>;

namespace relay {
namespace contrib {
namespace uma {
transform::Pass RelayToTIR(String target_name);
runtime::Module TIRToRuntime(IRModule mod, Target target);
}  // namespace uma
}  // namespace contrib
}  // namespace relay

TVM_REGISTER_GLOBAL("relay.backend.contrib.uma.RegisterTarget")
    .set_body_typed([](String target_name, Map<String, ObjectRef> attr_options) -> bool {
      // create only new target and init only once
      for (const String registered_target_name : TargetKindRegEntry::ListTargetKinds()) {
        if (registered_target_name == target_name) {
          LOG(FATAL) << "TVM UMA Error: Target is already registered: " << target_name;
        }
      }

      auto target_kind =
          TargetKindRegEntry::RegisterOrGet(target_name)
              .set_name()
              .set_default_device_type(kDLCPU)
              .add_attr_option<Array<String>>("keys")
              .add_attr_option<String>("tag")
              .add_attr_option<String>("device")
              .add_attr_option<String>("model")
              .add_attr_option<Array<String>>("libs")
              .add_attr_option<Target>("host")
              .add_attr_option<Integer>("from_device")
              .set_attr<relay::transform::FTVMRelayToTIR>(
                  attr::kRelayToTIR, relay::contrib::uma::RelayToTIR(target_name))
              .set_attr<FTVMTIRToRuntime>("TIRToRuntime", relay::contrib::uma::TIRToRuntime);

      // target kind attrs inventory
      auto kind = TargetKind::Get(target_name).value();
      auto list_attrs = TargetKindRegEntry::ListTargetKindOptions(kind);

      for (auto& attr_option : attr_options) {
        auto option_name = attr_option.first;
        auto default_value = attr_option.second;
        if (list_attrs.find(option_name) != list_attrs.end()) {
          LOG(FATAL) << "TVM UMA Error: Attribute is already registered: " << option_name;
        }
        if (default_value->IsInstance<StringObj>()) {
          target_kind.add_attr_option<String>(option_name, Downcast<String>(default_value));
        } else if (default_value->IsInstance<IntImmNode>()) {
          target_kind.add_attr_option<Integer>(option_name, Downcast<Integer>(default_value));
        } else {
          LOG(FATAL) << "TypeError: Only String, Integer, or Bool are supported. "
                     << "Given attribute option type: " << attr_option.second->GetTypeKey();
        }
      }
      return true;
    });

}  // namespace tvm
