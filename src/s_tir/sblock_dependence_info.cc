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

#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/sblock_dependence_info.h>
#include <tvm/s_tir/utils.h>

namespace tvm {
namespace tir {

TVM_FFI_STATIC_INIT_BLOCK() { SBlockDependenceInfoNode::RegisterReflection(); }

/**
 * @brief A helper class to collect and build SBlock Dependences using SBlockScope class
 */
class SBlockDependenceInfoCollector : private StmtVisitor {
 public:
  static void Collect(SBlockDependenceInfoNode* self, const Stmt& stmt) {
    SBlockDependenceInfoCollector collector(self);
    collector.VisitStmt(stmt);
  }

  explicit SBlockDependenceInfoCollector(SBlockDependenceInfoNode* self)
      : self_(self), block_frames_{} {
    block_frames_.emplace_back();
  }

  void MakeSBlockScope(StmtSRef scope) {
    ffi::Array<StmtSRef> child_block_srefs = std::move(block_frames_.back());
    self_->sref2scope[scope] = SBlockScope(child_block_srefs);
  }

  void VisitStmt_(const SBlockRealizeNode* realize) final {
    block_frames_.emplace_back();
    const SBlockNode* block = realize->block.get();
    // Recursive visit
    VisitStmt(block->body);  // `block->init` is not visited
    // Create SBlockInfo for the block
    auto sref = self_->stmt2ref.at(block);
    MakeSBlockScope(sref);
    // Update parent scope
    block_frames_.pop_back();
    block_frames_.back().push_back(sref);
  }

  void VisitStmt_(const SeqStmtNode* seq_stmt) final {
    // Set `seq_index` information for SeqStmtNode
    StmtVisitor::VisitStmt_(seq_stmt);
    SetSeqIndexInChildren(self_->stmt2ref, seq_stmt, false);
  }

  SBlockDependenceInfoNode* self_;
  /*! \brief The stack frames of blocks in the DFS visit. */
  std::vector<ffi::Array<StmtSRef>> block_frames_;
};

SBlockDependenceInfo::SBlockDependenceInfo() {
  data_ = ffi::make_object<SBlockDependenceInfoNode>();
}

SBlockDependenceInfo::SBlockDependenceInfo(IRModule mod) {
  ObjectPtr<SBlockDependenceInfoNode> n = ffi::make_object<SBlockDependenceInfoNode>();
  SBlockDependenceInfoNode* self = n.get();
  n->stmt2ref = SRefTreeCreator::Create(mod, /* include_loops */ false);

  for (const auto& kv : mod->functions) {
    const BaseFunc& base_func = kv.second;
    if (auto opt = base_func.as<PrimFunc>()) {
      auto func = opt.value();
      SBlockDependenceInfoCollector::Collect(self, func->body);
    }
  }
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.SBlockDependenceInfo",
           [](IRModule mod) -> SBlockDependenceInfo { return SBlockDependenceInfo(mod); })
      .def_method("s_tir.SBlockDependenceInfoGetSBlockScope",
                  &SBlockDependenceInfoNode::GetSBlockScope)
      .def("s_tir.SBlockDependenceInfoGetSRef",
           [](SBlockDependenceInfo self, Stmt stmt) -> ffi::Optional<StmtSRef> {
             auto it = self->stmt2ref.find(stmt.get());
             return it != self->stmt2ref.end() ? it->second : ffi::Optional<StmtSRef>(std::nullopt);
           });
}

}  // namespace tir
}  // namespace tvm
