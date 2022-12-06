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
  PackedCallLegalizer(IRModule m, const InputMap& inputs) : mod_{m}, inputs_{inputs} {}

  Stmt Legalize(tir::Stmt body) { return StmtExprMutator::VisitStmt(body); }

  Stmt VisitStmt_(const EvaluateNode* op) final {
    if (tir::is_const_int(op->value)) return StmtExprMutator::VisitStmt_(op);
    const CallNode* call = op->value.as<CallNode>();
    // Given a packed call f(A,B,C), we need a set of new statements
    // let A_packed = set_struct(tvm_value1, A)
    // let B_packed = set_struct(tvm_value2, B)
    // let C_packed = set_struct(tvm_value3, C)
    // call_packed(f, A_packed, B_packed, C_packed)
    if (call) {
      if (call->op.same_as(builtin::tvm_call_cpacked())) {
        Array<PrimExpr> packed_args{call->args[0]};
        VLOG(2) << "Legalize call:" << call;
        BaseFunc base_func = mod_->Lookup(Downcast<StringImm>(call->args[0])->value);
        const PrimFuncNode* prim_func = base_func.as<PrimFuncNode>();
        VLOG(2) << " to func " << base_func;
        for (unsigned i = 1; i < call->args.size() - 1; i++) {
          // No need to pack inputs of the prim_func
          if (inputs_[call->args[i]] == true) {
            packed_args.push_back(call->args[i]);
          } else {
            // Stack-allocate a DLTensor for this parameter. Note that LowerTVMBuiltin will collect
            // all such stack-allocated tensors and minimize the storage needed by reusing
            // DLTensors.
            Array<PrimExpr> call_args{call->args[i]};
            tvm::runtime::Map<tvm::tir::Var, tvm::tir::Buffer>::iterator param_buf_it;
            if (prim_func != nullptr) {
              auto param_var = prim_func->params[i - 1];
              param_buf_it = prim_func->buffer_map.find(param_var);
            }
            if (prim_func != nullptr && param_buf_it != prim_func->buffer_map.end()) {
              Buffer param = (*param_buf_it).second;
              PrimExpr shape = tvm::tir::Call(
                  DataType::Handle(), tvm::tir::builtin::tvm_stack_make_shape(), param->shape);
              Cast var_type(param->dtype, IntImm(DataType::Int(32), 0));
              call_args.push_back(shape /* shape */);
              call_args.push_back(make_zero(DataType::Handle()) /* strides */);
              call_args.push_back(tvm::IntImm(DataType::UInt(32), param->shape.size()) /* ndim */);
              call_args.push_back(var_type /* carries dtype */);
              call_args.push_back(param->elem_offset /* elem_offset */);
            } else {
              // When the PrimFunc cannot be found, most DLTensor information cannot be populated.
              PrimExpr shape = tvm::tir::Call(
                  DataType::Handle(), tvm::tir::builtin::tvm_stack_make_shape(), Array<PrimExpr>());
              Cast var_type(DataType::Handle(), IntImm(DataType::Int(32), 0));
              call_args.push_back(shape /* shape */);
              call_args.push_back(make_zero(DataType::Handle()) /* strides */);
              call_args.push_back(tvm::IntImm(DataType::UInt(32), 0) /* ndim */);
              call_args.push_back(var_type /* carries dtype */);
              call_args.push_back(tvm::IntImm(DataType::UInt(64), 0) /* elem_offset */);
            }
            packed_args.push_back(tvm::tir::Call(
                DataType::Handle(), tvm::tir::builtin::tvm_stack_make_array(), call_args));
          }
        }
        packed_args.push_back(call->args[call->args.size() - 1]);  // push device_context
        // Evaluate the packed call
        return tir::Evaluate(tir::Call(call->dtype, call->op, packed_args));
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

 private:
  IRModule mod_;
  InputMap inputs_;  // Store the inputs to the primfunc that don't need to be packed.
};

namespace transform {

Pass LegalizePackedCalls() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();

    // Note which Var are inputs and exclude them from packing.
    InputMap inputs;
    for (auto i : f->params) {
      inputs[i] = true;
    }
    n->body = PackedCallLegalizer(m, inputs).Legalize(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LegalizePackedCalls", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LegalizePackedCalls").set_body_typed(LegalizePackedCalls);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
