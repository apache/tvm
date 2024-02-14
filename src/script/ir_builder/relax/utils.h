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
#ifndef TVM_SCRIPT_IR_BUILDER_RELAX_UTILS_H_
#define TVM_SCRIPT_IR_BUILDER_RELAX_UTILS_H_

#include <tvm/relax/struct_info_functor.h>
#include <tvm/script/ir_builder/relax/frame.h>
#include <tvm/script/ir_builder/relax/ir.h>

#include <string>

namespace tvm {
namespace script {
namespace ir_builder {
namespace relax {

inline FunctionFrame FindFunctionFrame(const String& method) {
  if (Optional<FunctionFrame> frame = IRBuilder::Current()->FindFrame<FunctionFrame>()) {
    return frame.value();
  }
  LOG(FATAL) << "ValueError: Function frame not find. Please ensure '" << method
             << "' is called under R.function()";
  throw;
}

inline IfFrame FindIfFrame(const String& method) {
  if (Optional<IfFrame> frame = IRBuilder::Current()->GetLastFrame<IfFrame>()) {
    return frame.value();
  } else {
    LOG(FATAL) << "ValueError: IfThenElse frame not find. Please ensure '" << method
               << "' is called under R.if_()";
  }
  throw;
}

inline tvm::relax::BlockBuilder GetBlockBuilder() {
  Optional<FunctionFrame> frame = IRBuilder::Current()->FindFrame<FunctionFrame>();
  CHECK(frame.defined()) << "ValueError: Relax Function frame not find. Please ensure "
                            "assignment is called under R.function()";
  return frame.value()->block_builder;
}

inline BlockFrame CheckBlockFrameExistAndUnended() {
  // We check if the current block is "ended" - if a block is ended, it is not allowed to emit new
  // bindings into this block, and we should throw exceptions.

  Optional<BlockFrame> block_frame = IRBuilder::Current()->GetLastFrame<BlockFrame>();
  CHECK(block_frame.defined()) << "ValueError: Block frame not find";
  CHECK(!block_frame.value()->block_ended)
      << "ValueError: New binding is not allowed after dataflow block output.";
  return block_frame.value();
}

inline tvm::relax::SeqExpr GetSeqExprForBranch(const SeqExprFrame& frame, String* var_name) {
  // Step 0. Check frame type
  std::string method;
  if (frame->IsInstance<ThenFrameNode>()) {
    method = "R.Then";
  } else if (frame->IsInstance<ElseFrameNode>()) {
    method = "R.Else";
  } else {
    ICHECK(false) << "TypeError: Unsupported frame type: " << frame->GetTypeKey();
  }

  // Step 1. Check non-empty block and last binding is non-dataflow
  CHECK(!frame->binding_blocks.empty())
      << "Empty body is not allowed for '" << method << "' statements.";
  const tvm::relax::BindingBlock& last_block = frame->binding_blocks.back();
  CHECK(!last_block->bindings.empty()) << "Blocks are expected to be non-empty.";

  // Step 2. Collect body from the last binding.
  tvm::relax::Expr body;
  const tvm::relax::Binding& last_binding = last_block->bindings.back();
  if (const auto* var_binding = last_binding.as<tvm::relax::VarBindingNode>()) {
    CHECK(!var_binding->var->IsInstance<tvm::relax::DataflowVarNode>())
        << "A non-dataflow var is expected in the last binding of '" << method << "'.";
    body = var_binding->value;
    *var_name = var_binding->var->name_hint();
  } else if (const auto* match_cast = last_binding.as<tvm::relax::MatchCastNode>()) {
    CHECK(!match_cast->var->IsInstance<tvm::relax::DataflowVarNode>())
        << "A non-dataflow var is expected in the last binding of '" << method << "'.";
    body = var_binding->value;
    *var_name = match_cast->var->name_hint();
  } else {
    ICHECK(false) << "TypeError: Unsupported binding type: " << last_binding->GetTypeKey();
  }

  // Step 3. Re-collect binding blocks to remove the last binding.
  Array<tvm::relax::BindingBlock> new_blocks(frame->binding_blocks.begin(),
                                             frame->binding_blocks.end() - 1);
  Array<tvm::relax::Binding> last_block_bindings(last_block->bindings.begin(),
                                                 last_block->bindings.end() - 1);
  new_blocks.push_back(tvm::relax::BindingBlock(last_block_bindings));

  return tvm::relax::SeqExpr(new_blocks, body);
}

}  // namespace relax
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_IR_BUILDER_RELAX_UTILS_H_
