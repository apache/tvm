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
#ifndef TVM_SCRIPT_IR_BUILDER_IR_UTILS_H_
#define TVM_SCRIPT_IR_BUILDER_IR_UTILS_H_

#include <tvm/script/ir_builder/ir/frame.h>

namespace tvm {
namespace script {
namespace ir_builder {
namespace ir {

inline IRModuleFrame FindModuleFrame(const String& method) {
  IRBuilder builder = IRBuilder::Current();
  if (Optional<IRModuleFrame> frame = builder->FindFrame<IRModuleFrame>()) {
    const Optional<IRModuleFrame>& last_module_frame = builder->GetLastFrame<IRModuleFrame>();
    if (last_module_frame.defined() && last_module_frame.value() == frame) {
      return frame.value();
    }
  } else {
    LOG(FATAL) << "ValueError: IRModule frame not find. Please ensure '" << method
               << "' is called under I.ir_module()";
  }
  LOG(FATAL) << "ValueError: '" << method << "' must be called immediately under I.ir_module()";
  throw;
}

inline IRModuleFrame FindModuleFrame() {
  IRBuilder builder = IRBuilder::Current();
  if (Optional<IRModuleFrame> frame = builder->FindFrame<IRModuleFrame>()) {
    return frame.value();
  } else {
    LOG(FATAL) << "ValueError: IRModule frame not find. Please ensure it"
               << " is called under I.ir_module()";
  }
  throw;
}

}  // namespace ir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_IR_BUILDER_IR_UTILS_H_
