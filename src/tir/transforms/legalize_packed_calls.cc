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
 * \file make_packed_call.cc
 * \brief Rewrite packed calls in AOT so that the arguments are packed
 */
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>

#include "ir_utils.h"

namespace tvm {
namespace tir {

using InputMap =
    std::unordered_map<PrimExpr, bool, runtime::ObjectPtrHash, runtime::ObjectPtrEqual>;
/**
 * This is a legalization pass only used in AOT. Traverse the TIR graph to legalize
 * packed calls by making its argument wrapped in TVMValues (by using tvm_set_struct built-in)
 */
class PackedCallLegalizer : public StmtExprMutator {
 public:
  Stmt Legalize(const InputMap& params, tir::Stmt body) {
    inputs_ = params;
    return StmtExprMutator::VisitStmt(body);
  }

  Stmt VisitStmt_(const EvaluateNode* op) final {
    if (tir::is_const_int(op->value)) return StmtExprMutator::VisitStmt_(op);
    const CallNode* call = op->value.as<CallNode>();
    // Given a packed call f(A,B,C), we need a set of new statements
    // let A_packed = set_struct(tvm_value1, A)
    // let B_packed = set_struct(tvm_value2, B)
    // let C_packed = set_struct(tvm_value3, C)
    // call_packed(f, A_packed, B_packed, C_packed)
    std::vector<Stmt> new_stmts;
    if (call) {
      if (call->op.same_as(builtin::tvm_call_cpacked())) {
        Array<PrimExpr> packed_args{call->args[0]};
        std::vector<tir::Var> tvm_values;
        for (unsigned i = 1; i < call->args.size(); i++) {
          // No need to pack inputs of the prim_func
          if (inputs_[call->args[i]] == true) {
            packed_args.push_back(call->args[i]);
          } else {
            // Pack the argument inside a TVMValue
            std::stringstream ss;
            ss << "tvm_value_" << tvm_value_index_++;
            auto sid_array = tir::Var(ss.str(), DataType::Handle());
            tvm_values.push_back(sid_array);

            new_stmts.push_back(tir::Evaluate(
                tvm::tir::Call(DataType::Handle(), tvm::tir::builtin::tvm_struct_set(),
                               {sid_array, 0, tir::builtin::kArrData, call->args[i]})));
            packed_args.push_back(sid_array);
          }
        }
        // Evaluate the packed call
        new_stmts.push_back(tir::Evaluate(tir::Call(call->dtype, call->op, packed_args)));
        tir::Stmt call_stmt = tir::SeqStmt(new_stmts);

        // Allocate the TVMValues on the stack and define the variables
        for (auto v : tvm_values) {
          call_stmt = LetStmt(v, StackAlloca("array", 1), call_stmt);
        }
        return call_stmt;
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

 private:
  InputMap inputs_;      // Store the inputs to the primfunc that don't need to be packed.
  int tvm_value_index_;  // Index of the actual tvm_value variable
};

namespace transform {

Pass LegalizePackedCalls() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();

    // Create the
    InputMap inputs;
    for (auto i : f->params) {
      inputs[i] = true;
    }
    n->body = PackedCallLegalizer().Legalize(inputs, std::move(n->body));
    return std::move(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LegalizePackedCalls", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LegalizePackedCalls").set_body_typed(LegalizePackedCalls);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
