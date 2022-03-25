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
 * \file tir/contrib/ethosu/passes.cc
 *
 * \brief Passes used in TIR lowering for the microNPU compiler.
 */
#include <tvm/tir/builtin.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {
namespace contrib {
namespace ethosu {

/*!
 * \brief This mutator moves allocates to the top of the body of the main
 * function.
 *
 * Note: This pass can currently only be run in conjunction with the
 * LowerToTIR() pass as it expects a single primitive function called
 * "main" that is being offloaded to the NPU.
 *
 * For example,
 * Before:
 *   allocate {
 *       extern_call(...)
 *           allocate {
 *               extern_call(...)
 *           }
 *   }
 *
 * After:
 *   allocate {
 *       allocate {
 *           extern_call(...)
 *           extern_call(...)
 *       }
 *  }
 */
class HoistAllocatesMutator : public StmtExprMutator {
 public:
  HoistAllocatesMutator() {}

  PrimFunc operator()(PrimFunc main_func) {
    Stmt new_main_func_body = this->VisitStmt(main_func->body);

    // Write all allocates that were removed in reverse order
    for (auto it = allocates_.rbegin(); it != allocates_.rend(); it++) {
      Allocate current_alloc = *it;
      if (it != allocates_.rbegin()) {
        new_main_func_body = SeqStmt({new_main_func_body});
      }
      new_main_func_body =
          Allocate(current_alloc->buffer_var, current_alloc->dtype, current_alloc->extents,
                   current_alloc->condition, new_main_func_body, current_alloc->annotations,
                   current_alloc->span);
    }

    PrimFunc new_main_func =
        PrimFunc(main_func->params, new_main_func_body, main_func->ret_type, main_func->buffer_map,
                 main_func->preflattened_buffer_map, main_func->attrs);
    return new_main_func;
  }

 private:
  Stmt VisitStmt_(const AllocateNode* op) override {
    allocates_.push_back(GetRef<Allocate>(op));

    // Skip the allocate node itself
    if (const auto* seq = op->body.as<SeqStmtNode>()) {
      // Traverse the allocate body recursively and flatten
      Array<Stmt> new_stmts;
      new_stmts.reserve(seq->seq.size());
      for (const Stmt& old_stmt : seq->seq) {
        new_stmts.push_back(VisitStmt(old_stmt));
      }
      return SeqStmt::Flatten(new_stmts);
    } else {
      return VisitStmt(op->body);
    }
  }

  /*! A stack to store allocates as they are visited. */
  std::vector<Allocate> allocates_;
};

/*!
 * \brief A pass to hoist allocate nodes to the top of the body of the main function.
 *
 * \return tvm::transform::Pass
 */
tvm::transform::Pass HoistAllocates() {
  auto pass_func = [=](PrimFunc f, IRModule mod, tvm::transform::PassContext ctx) {
    ICHECK(mod->GetGlobalVars().size() == 1 && mod->ContainGlobalVar("main"))
        << "Expected a single primitive function called 'main'. Please run the HoistAllocates pass "
           "in conjunction with the LowerToTIR() pass.";
    return HoistAllocatesMutator()(f);
  };
  return tvm::tir::transform::CreatePrimFuncPass(pass_func, 0, "tir.contrib.ethos-u.HoistAllocates",
                                                 {});
}

TVM_REGISTER_GLOBAL("tir.contrib.ethos-u.HoistAllocates").set_body_typed(HoistAllocates);

}  // namespace ethosu
}  // namespace contrib
}  // namespace tir
}  // namespace tvm
