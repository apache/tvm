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
 * \file s_tir/analysis/find_anchor_sblock.cc
 * \brief Find the "anchor block" of a given module
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {

Stmt GetEnclosingLoop(const SBlockNode* block, Stmt func_body) {
  struct GetRootSeqStmt : public StmtVisitor {
    void VisitStmt_(const SeqStmtNode* seq) override { result = seq; }
    const SeqStmtNode* result;
  };

  struct BlockFinder : public StmtVisitor {
    explicit BlockFinder(const SBlockNode* tgt) : target(tgt) {}

    void VisitStmt_(const SBlockNode* block) override {
      if (block == target) {
        found = true;
      }
    }

    const SBlockNode* target;
    bool found = false;
  };

  GetRootSeqStmt seq_finder;
  seq_finder(func_body);

  TVM_FFI_ICHECK(seq_finder.result);

  for (auto stmt : seq_finder.result->seq) {
    if (stmt->IsInstance<ForNode>()) {
      BlockFinder finder(block);
      finder(stmt);
      if (finder.found) {
        return stmt;
      }
    }
  }

  TVM_FFI_THROW(InternalError) << "Enclosing loop not found for a block "
                               << ffi::GetRef<SBlock>(block);
  TVM_FFI_UNREACHABLE();
}

const SBlockNode* FindAnchorBlock(const IRModule& mod) {
  struct ReductionSBlockCollector : public StmtVisitor {
    void VisitStmt_(const SBlockNode* block) override {
      if (block->init) {
        blocks.push_back(block);
      }
      StmtVisitor::VisitStmt(block->body);
    }
    std::vector<const SBlockNode*> blocks;
  };

  if (auto prim_func = FindEntryFunc(mod, nullptr)) {
    ReductionSBlockCollector collector;
    collector(prim_func->body);

    const auto& candidates = collector.blocks;

    if (candidates.empty()) {
      return nullptr;
    } else if (candidates.size() == 1) {
      return candidates[0];
    }

    double best_flops = -1;
    int best_idx = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
      auto loop = GetEnclosingLoop(candidates[i], prim_func->body);
      auto flops = s_tir::EstimateTIRFlops(loop);
      if (flops > best_flops) {
        best_flops = flops;
        best_idx = i;
      }
    }
    return candidates[best_idx];
  }
  return nullptr;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.analysis.find_anchor_sblock", [](const IRModule& mod) {
    auto ret = FindAnchorBlock(mod);
    if (ret) {
      return ffi::Optional<SBlock>(ffi::GetRef<SBlock>(ret));
    }
    return ffi::Optional<SBlock>(std::nullopt);
  });
}

}  // namespace tir
}  // namespace tvm
