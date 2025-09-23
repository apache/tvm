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

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

class IrregularLoopAnnotator : public StmtMutator {
 public:
  static Stmt Annotate(const Stmt& body) { return IrregularLoopAnnotator().VisitStmt(body); }

 private:
  IrregularLoopAnnotator() = default;

  Stmt VisitStmt_(const ForNode* op) final {
    bool cur_has_jump = has_jump_;
    has_jump_ = false;
    For res = Downcast<For>(StmtMutator::VisitStmt_(op));
    if (has_jump_) {
      CHECK(op->kind == ForKind::kSerial)
          << "Loop kind " << op->kind << " is invalid for irregular loop " << op->loop_var;
      for (const char* key : {attr::pragma_auto_unroll_max_step, attr::pragma_unroll_explicit,
                              attr::pragma_loop_partition_hint, attr::software_pipeline_stage}) {
        CHECK(!res->annotations.count(key))
            << "Annotation `" << key << "` is invalid for irregular loop " << op->loop_var;
      }
      res.CopyOnWrite()->annotations.Set(attr::irregular_loop_mark, 1);
    }
    std::swap(cur_has_jump, has_jump_);
    return res;
  }

  Stmt VisitStmt_(const WhileNode* op) final {
    bool cur_has_jump = has_jump_;
    has_jump_ = false;
    Stmt res = StmtMutator::VisitStmt_(op);
    std::swap(cur_has_jump, has_jump_);
    return res;
  }

  Stmt VisitStmt_(const EvaluateNode* op) final {
    if (const CallNode* call = op->value.as<CallNode>()) {
      if (call->op.same_as(builtin::continue_loop()) || call->op.same_as(builtin::break_loop())) {
        has_jump_ = true;
      }
    }
    return ffi::GetRef<Evaluate>(op);
  }

  bool has_jump_{false};
};

namespace transform {

Pass AnnotateIrregularLoop() {
  auto pass_func = [](PrimFunc func, IRModule mod, PassContext ctx) -> PrimFunc {
    func.CopyOnWrite()->body = IrregularLoopAnnotator::Annotate(func->body);
    return func;
  };

  return CreatePrimFuncPass(pass_func, 0, "tir.AnnotateIrregularLoop", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.transform.AnnotateIrregularLoop", AnnotateIrregularLoop);
}

}  // namespace transform
}  // namespace tir
}  // namespace tvm
