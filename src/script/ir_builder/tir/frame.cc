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
#include <tvm/script/ir_builder/tir/frame.h>
#include <tvm/tir/function.h>

#include "../../../tir/ir/script/script_complete.h"
#include "./utils.h"

namespace tvm {
namespace script {
namespace ir_builder {
namespace tir {

void PrimFuncFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  tvm::tir::PrimFunc func(
      /*params=*/args,
      /*body=*/AsStmt(stmts),
      /*ret_type=*/ret_type.value_or(TupleType::Empty()),
      /*buffer_map=*/buffer_map,
      /*preflattened_buffer_map=*/preflattened_buffer_map,
      /*attrs=*/attrs.defined() ? DictAttrs(attrs.value()) : NullValue<DictAttrs>());
  func = tvm::tir::ScriptComplete(func, root_alloc_buffers);
  IRBuilder builder = IRBuilder::Current();
  if (builder->frames.empty()) {
    ICHECK(!builder->result.defined()) << "ValueError: Builder.result has already been set";
    builder->result = func;
  } else if (Optional<ir::IRModuleFrame> opt_frame = builder->FindFrame<ir::IRModuleFrame>()) {
    ir::IRModuleFrame frame = opt_frame.value();
    frame->global_vars.push_back(GlobalVar(name.value_or("")));
    frame->functions.push_back(func);
  } else {
    LOG(FATAL) << "ValueError: Cannot find where to insert PrimFunc";
  }
}

TVM_REGISTER_NODE_TYPE(TIRFrameNode);
TVM_REGISTER_NODE_TYPE(PrimFuncFrameNode);

}  // namespace tir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
