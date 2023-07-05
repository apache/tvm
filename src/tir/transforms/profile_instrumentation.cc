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
 * \file profile_instrumentation.cc
 */
// Insert profile intrinsic at loop and function level. During codegen,
// these instruction can be replaced with a call to a target specific handler
// and can be used to capture profiling information such as processor cycles.

#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {
namespace lwp {

TVM_REGISTER_PASS_CONFIG_OPTION("tir.lwp_disable_func_prof", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.lwp_max_depth", Integer);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.lwp_min_height", Integer);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.instr_siblings", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.reset_start_id", Bool);

static int32_t start_id = 0;

struct LoopInfo {
  LoopInfo() = default;
  LoopInfo(unsigned i, unsigned d, unsigned h = 0) : id(i), depth(d), height(h) {
    has_siblings = false;
    has_parallel = false;
  }
  unsigned id;
  int32_t depth;
  int32_t height;
  bool has_siblings;
  // Set to 'true' if ForKind::kParallel is set for the current loop or one of its ancestor
  bool has_parallel;
};

using LoopInfoMap = std::unordered_map<const ForNode*, LoopInfo>;
// Traverse loops depth first and assign them a unique number.
class LoopAnalyzer : public StmtExprVisitor {
 public:
  LoopInfoMap Analyze(const Stmt& stmt) {
    this->VisitStmt(stmt);
    return loops;
  }
  void VisitStmt_(const ForNode* op) final {
    LoopInfo loop_info(start_id, 0);
    start_id++;
    loop_info.height = TraverseLoop(op->body, 0);
    loops[op] = loop_info;
  }

  unsigned TraverseLoop(const Stmt& stmt, unsigned parent_depth, bool has_parallel = false) {
    if (stmt->IsInstance<SeqStmtNode>()) {
      std::vector<const ForNode*> siblings;
      unsigned height = 0;
      bool has_loop = false;
      const SeqStmtNode* n = stmt.as<SeqStmtNode>();
      for (Stmt s : n->seq) {
        if (s->IsInstance<ForNode>()) {
          has_loop = true;
          const ForNode* f = s.as<ForNode>();
          LoopInfo loop_info(start_id, parent_depth + 1);
          start_id++;
          bool parent_parallel = false;
          if (has_parallel) {
            loop_info.has_parallel = true;
            parent_parallel = true;
          } else if (f->kind == ForKind::kParallel) {
            // has_parallel for the current loop is being set to 'false' since the
            // intrinsic is added outside of the loop. The instrumentation isn't
            // allowed for the subsequent nested loops.
            loop_info.has_parallel = false;
            parent_parallel = true;
          }
          siblings.push_back(f);
          height = std::max(height, TraverseLoop(f->body, parent_depth + 1, parent_parallel));
          loop_info.height = height;
          loops[f] = loop_info;
        }
      }
      if (siblings.size() > 1) {
        for (auto* l : siblings) {
          loops[l].has_siblings = true;
        }
      }
      height = has_loop ? height + 1 : height;
      return height;  // Parent's height : max of all children's height
    } else if (stmt->IsInstance<IfThenElseNode>()) {
      const IfThenElseNode* n = stmt.as<IfThenElseNode>();
      unsigned height = TraverseLoop(n->then_case, parent_depth, has_parallel);
      if (n->else_case) {
        height = std::max(height, TraverseLoop(n->else_case.value(), parent_depth, has_parallel));
      }
      return height;
    } else if (stmt->IsInstance<ForNode>()) {
      const ForNode* f = stmt.as<ForNode>();
      LoopInfo loop_info(start_id, parent_depth + 1);
      start_id++;
      bool parent_parallel = false;
      if (has_parallel) {
        loop_info.has_parallel = true;
        parent_parallel = true;
      } else if (f->kind == ForKind::kParallel) {
        // has_parallel for the current loop is being set to 'false' since the
        // intrinsic is added outside of the loop. The instrumentation isn't
        // allowed for the subsequent nested loops.
        loop_info.has_parallel = false;
        parent_parallel = true;
      }
      unsigned height = TraverseLoop(f->body, parent_depth + 1, parent_parallel);
      loop_info.height = height;
      loops[f] = loop_info;
      return height + 1;
    } else if (stmt->IsInstance<LetStmtNode>()) {
      const LetStmtNode* n = stmt.as<LetStmtNode>();
      return TraverseLoop(n->body, parent_depth, has_parallel);
    } else if (stmt->IsInstance<AttrStmtNode>()) {
      const AttrStmtNode* n = stmt.as<AttrStmtNode>();
      return TraverseLoop(n->body, parent_depth, has_parallel);
    } else if (stmt->IsInstance<AllocateNode>()) {
      const AllocateNode* n = stmt.as<AllocateNode>();
      return TraverseLoop(n->body, parent_depth, has_parallel);
    } else {
      return 0;  // inner-most loop
    }
  }

 private:
  LoopInfoMap loops;
};

class InstrumentIntrin : public StmtMutator {
 public:
  InstrumentIntrin(int32_t max_depth, int32_t min_height, bool instr_siblings)
      : max_instr_depth_(max_depth),
        min_instr_height_(min_height),
        instr_siblings_(instr_siblings) {}

  void GetLoopInfo(PrimFuncNode* op) {
    LoopAnalyzer analzer;
    loops_ = std::move(analzer.Analyze(op->body));
  }

  Stmt VisitStmt_(const SeqStmtNode* op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    return SeqStmt::Flatten(stmt);
  }

  Stmt VisitStmt_(const ForNode* op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    if (loops_.count(op) < 1) return stmt;

    LoopInfo loop_info = loops_[op];

    if (loop_info.has_parallel) {
      return stmt;
    }

    // Exclude inner-most loops from instrumentation. The inner-most loop has
    // height '0' and it increases as we move outward in the loop nest.
    if (loop_info.height < min_instr_height_) {
      return stmt;
    }

    // Only instrument loops with a sibling
    if (instr_siblings_ && !loop_info.has_siblings) {
      return stmt;
    }

    // If instr_siblings_ is set, ignore max depth for instrumentation
    if (!instr_siblings_ && loop_info.depth > max_instr_depth_) {
      return stmt;
    }
    PrimExpr id = static_cast<int32_t>(loop_info.id);
    PrimExpr start_call = Call(DataType::Handle(), builtin::start_profile_intrinsic(), {id});
    PrimExpr end_call = Call(DataType::Handle(), builtin::end_profile_intrinsic(), {id});
    const Stmt start_profile = Evaluate(start_call);
    const Stmt end_profile = Evaluate(end_call);
    Stmt new_stmt = SeqStmt({start_profile, stmt, end_profile});
    return new_stmt;
  }

 private:
  LoopInfoMap loops_;
  int32_t max_instr_depth_;
  int32_t min_instr_height_;
  bool instr_siblings_;
};

class CheckParallelLoops : public StmtExprVisitor {
 public:
  bool HasParallelLoops(const Stmt& stmt) {
    this->VisitStmt(stmt);
    return has_parallel;
  }

 private:
  void VisitStmt_(const ForNode* op) final {
    if (op->kind == ForKind::kParallel) {
      has_parallel = true;
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }
  }

  bool has_parallel = false;
};

PrimFunc AddProfileBuiltins(PrimFunc func, int32_t max_instr_depth, int32_t min_instr_height,
                            bool instr_siblings, bool disable_func_instrumentation) {
  auto* func_ptr = func.CopyOnWrite();

  PrimExpr e = start_id++;
  if (!disable_func_instrumentation) {
    PrimExpr start_call = Call(DataType::Handle(), builtin::start_profile_intrinsic(), {e});
    PrimExpr end_call = Call(DataType::Handle(), builtin::end_profile_intrinsic(), {e});
    const Stmt start_profile = Evaluate(start_call);
    const Stmt end_profile = Evaluate(end_call);
    func_ptr->body = SeqStmt({start_profile, std::move(func_ptr->body), end_profile});
  }
  InstrumentIntrin p(max_instr_depth, min_instr_height, instr_siblings);
  p.GetLoopInfo(func_ptr);
  func_ptr->body = p(std::move(func_ptr->body));
  return func;
}

}  // namespace lwp

namespace transform {
Pass InstrumentProfileIntrinsics() {
  auto pass_func = [](IRModule m, PassContext ctx) {
    auto* mptr = m.CopyOnWrite();

    // All loops with depth <= max_instr_depth are instrumented. By default,
    // only outer-most loops are instrumented which has a depth of 0.
    // In addition, loops with siblings are also instrumented provided
    // their loop depth is >= min_instr_height. This is done to avoid
    // instrumenting inner-most loops.
    auto max_instr_depth = ctx->GetConfig<Integer>("tir.lwp_max_depth", Integer(0)).value();
    auto min_instr_height = ctx->GetConfig<Integer>("tir.lwp_min_height", Integer(1)).value();
    bool instr_siblings = ctx->GetConfig<Bool>("tir.instr_siblings", Bool(true)).value();
    bool disable_func_instrumentation =
        ctx->GetConfig<Bool>("tir.lwp_disable_func_prof", Bool(false)).value();
    bool reset_start_id = ctx->GetConfig<Bool>("tir.reset_start_id", Bool(false)).value();
    if (reset_start_id) lwp::start_id = 0;
    std::vector<std::pair<GlobalVar, PrimFunc>> updates;
    for (const auto& kv : mptr->functions) {
      if (auto func = kv.second.as<PrimFunc>()) {
        auto updated_func = lwp::AddProfileBuiltins(func.value(), max_instr_depth.IntValue(),
                                                    min_instr_height.IntValue(), instr_siblings,
                                                    disable_func_instrumentation);
        updates.push_back({kv.first, updated_func});
      }
    }
    for (const auto& pair : updates) {
      mptr->AddUnchecked(pair.first, pair.second);
    }
    return m;
  };

  return tvm::transform::CreateModulePass(pass_func, 0, "tir.InstrumentProfileIntrinsics", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InstrumentProfileIntrinsics")
    .set_body_typed(InstrumentProfileIntrinsics);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
