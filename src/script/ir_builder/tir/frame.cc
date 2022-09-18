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

void BlockFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  Array<tvm::tir::Buffer> tir_alloc_buffers;
  for (const tvm::tir::Buffer& buffer : alloc_buffers) {
    tir_alloc_buffers.push_back(buffer);
  }
  Map<String, ObjectRef> attrs = annotations.value_or({});
  if (int detect_access = (!reads.defined()) | (!writes.defined() << 1)) {
    attrs.Set("tir.script_parsing_detect_access", tvm::IntImm(DataType::Int(64), detect_access));
  }
  tvm::tir::Block block(iter_vars, reads.value_or(Array<tvm::tir::BufferRegion>()),
                        writes.value_or(Array<tvm::tir::BufferRegion>()), name, AsStmt(stmts), init,
                        tir_alloc_buffers, match_buffers, attrs);
  if (no_realize) {
    CHECK(iter_values.empty())
        << "ValueError: Block bindings are not allowed when `no_realize=True`";
    CHECK(!predicate.defined()) << "ValueError: `T.where` is not allowed when `no_realize=True`";
    AddToParent(block);
  } else {
    AddToParent(tvm::tir::BlockRealize(iter_values, predicate.value_or(Bool(true)), block));
  }
}

void BlockInitFrameNode::EnterWithScope() {
  BlockFrame frame = FindBlockFrame("T.init");
  if (frame->init.defined()) {
    LOG(FATAL) << "ValueError: Duplicate block init declaration";
  }
  TIRFrameNode::EnterWithScope();
}

void BlockInitFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  BlockFrame frame = FindBlockFrame("T.init");
  frame->init = AsStmt(stmts);
}

void ForFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(this->f_make_for_loop(vars, doms, AsStmt(stmts)));
}

void AssertFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tir::AssertStmt(condition, message, AsStmt(stmts)));
}

void LetFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tir::LetStmt(var, value, AsStmt(stmts)));
}

void RealizeFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tir::AttrStmt(buffer_slice->buffer, "realize_scope",
                                 tvm::tir::StringImm(storage_scope),
                                 tvm::tir::BufferRealize(buffer_slice->buffer, buffer_slice->region,
                                                         condition, AsStmt(stmts))));
}

void LaunchThreadFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tir::AttrStmt(iter_var, attr_key, extent, AsStmt(stmts)));
}

TVM_REGISTER_NODE_TYPE(TIRFrameNode);
TVM_REGISTER_NODE_TYPE(PrimFuncFrameNode);
TVM_REGISTER_NODE_TYPE(BlockFrameNode);
TVM_REGISTER_NODE_TYPE(BlockInitFrameNode);
TVM_REGISTER_NODE_TYPE(ForFrameNode);
TVM_REGISTER_NODE_TYPE(AssertFrameNode);
TVM_REGISTER_NODE_TYPE(LetFrameNode);
TVM_REGISTER_NODE_TYPE(RealizeFrameNode);
TVM_REGISTER_NODE_TYPE(LaunchThreadFrameNode);

}  // namespace tir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
