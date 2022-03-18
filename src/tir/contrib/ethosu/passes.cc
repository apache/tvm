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
 * For example,
 *               allocate {
 *                   extern_call(...) {
 *                       allocate {
 *     Before:               extern_call(...)
 *                       }
 *                   }
 *               }
 *
 *               allocate {
 *                   allocate {
 *                      extern_call(...)
 *     After:           extern_call(...)
 *                   }
 *               }
 */
class HoistAllocatesMutator : public StmtExprMutator {
 public:
  HoistAllocatesMutator() {}

  IRModule operator()(IRModule mod) {
    GlobalVar gv = mod->GetGlobalVar("main");
    PrimFunc main_func = Downcast<PrimFunc>(mod->Lookup(gv));
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
    mod->Update(gv, new_main_func);
    return mod;
  }

 private:
  Stmt VisitStmt_(const AllocateNode* op) override {
    allocates_.push_back(GetRef<Allocate>(op));

    // Skip the allocate node itself
    const auto* seq = op->body.as<SeqStmtNode>();
    ICHECK(seq) << "Expected a sequence statement but got " << op->body->GetTypeKey() << ".";

    // Traverse the allocate body recursively and flatten
    Array<Stmt> new_stmts;
    new_stmts.reserve(seq->seq.size());
    for (const Stmt& old_stmt : seq->seq) {
      new_stmts.push_back(VisitStmt(old_stmt));
    }
    return SeqStmt::Flatten(new_stmts);
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
  auto pass_func = [=](IRModule mod, tvm::transform::PassContext ctx) {
    return HoistAllocatesMutator()(mod);
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tir.contrib.ethos-u.HoistAllocates", {});
}

TVM_REGISTER_GLOBAL("tir.contrib.ethos-u.HoistAllocates").set_body_typed(HoistAllocates);

}  // namespace ethosu
}  // namespace contrib
}  // namespace tir
}  // namespace tvm
