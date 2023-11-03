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
 * \file make_unpacked_api.cc Lower PrimFunc to a standard C function API.
 */
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_set>
#include <utility>
#include <vector>

#include "arg_binder.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

namespace {

class SubroutineCallRewriter : public StmtExprMutator {
 public:
  static Optional<Stmt> Apply(const std::unordered_set<const GlobalVarNode*>& external_methods,
                              Stmt stmt) {
    SubroutineCallRewriter rewriter(external_methods);
    stmt = rewriter.VisitStmt(std::move(stmt));
    if (rewriter.made_change_) {
      return stmt;
    } else {
      return NullOpt;
    }
  }

 private:
  explicit SubroutineCallRewriter(const std::unordered_set<const GlobalVarNode*>& external_methods)
      : external_methods_(external_methods) {}

  PrimExpr VisitExpr_(const CallNode* op) override {
    auto node = Downcast<Call>(StmtExprMutator::VisitExpr_(op));

    if (auto gvar = node->op.as<GlobalVarNode>()) {
      if (external_methods_.count(gvar)) {
        Array<PrimExpr> args = node->args.Map([](const PrimExpr& arg) -> PrimExpr {
          if (auto* as_call = arg.as<CallNode>()) {
            if (as_call->op.same_as(builtin::tvm_stack_make_array())) {
              PrimExpr data_ptr = as_call->args[0];
              return data_ptr;
            }
          }
          return arg;
        });

        if (!args.same_as(node->args) || node->dtype != DataType::Int(32)) {
          auto write_ptr = node.CopyOnWrite();
          write_ptr->dtype = DataType::Int(32);
          write_ptr->args = args;
          made_change_ = true;
        }
      }
    }

    return std::move(node);
  }
  const std::unordered_set<const GlobalVarNode*>& external_methods_;
  bool made_change_{false};
};

}  // namespace

PrimFunc MakeUnpackedAPI(PrimFunc func) {
  // A function with an explicit calling convention has already been
  // lowered, and should not be modified.
  if (auto opt = func->GetAttr<Integer>(tvm::attr::kCallingConv)) {
    if (CallingConv(opt.value()->value) != CallingConv::kDefault) {
      return func;
    }
  }

  // Internal function calls do not need API updates
  auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
  if (!global_symbol.defined()) {
    return func;
  }

  Target target = [&]() {
    auto opt = func->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(opt) << "MakeUnpackedAPI required the function to be annotated with tvm::attr::kTarget ("
                << tvm::attr::kTarget << "), but the function only has attributes " << func->attrs;
    return opt.value();
  }();
  int target_device_type = target->GetTargetDeviceType();

  // A function without a host target has already been lowered.
  Target target_host;
  if (auto opt = target->GetHost()) {
    target_host = opt.value();
  } else {
    return func;
  }

  auto* func_ptr = func.CopyOnWrite();

  // Setup device context
  Integer device_type(target_device_type);
  Integer device_id(0);
  ObjectRef node = String("default");
  const Stmt nop = Evaluate(0);
  std::vector<Stmt> device_init;

  // Collect variables and buffers to map between
  Array<Var> args;

  for (const Var& param : func->params) {
    // Ideally all func params should have Buffers defined in the buffer_map
    // We should look to insert buffer_maps for all PrimFuncs that are returned
    // to the core compiler.
    if (func->buffer_map.find(param) != func->buffer_map.end()) {
      args.push_back(func->buffer_map[param]->data);
    } else {
      args.push_back(param);
    }
  }

  if (func->buffer_map.size()) {
    device_init.push_back(AttrStmt(node, attr::device_id, device_id, nop));
    device_init.push_back(AttrStmt(node, attr::device_type, device_type, nop));
  }

  Stmt body = MergeNest(device_init, SeqStmt({func_ptr->body, Evaluate(ret(Integer(0)))}));

  func_ptr->body = body;
  func_ptr->params = args;
  func_ptr->ret_type = PrimType(DataType::Int(32));
  func_ptr->buffer_map = Map<Var, Buffer>();

  // return the function.
  return WithAttrs(std::move(func), {{tvm::attr::kTarget, target_host}});
}

namespace transform {

Pass MakeUnpackedAPI() {
  auto pass_func = [](IRModule mod, PassContext ctx) {
    std::unordered_set<const GlobalVarNode*> external_methods;
    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto* prim_func = base_func.as<PrimFuncNode>()) {
        if (prim_func->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
          external_methods.insert(gvar.get());
        }
      }
    }

    IRModule updates;

    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto opt = base_func.as<PrimFunc>()) {
        auto func = opt.value();

        if (auto body = SubroutineCallRewriter::Apply(external_methods, func->body)) {
          func.CopyOnWrite()->body = body.value();
        }

        func = MakeUnpackedAPI(std::move(func));
        if (!func.same_as(base_func)) {
          updates->Add(gvar, func);
        }
      }
    }

    if (updates->functions.size()) {
      mod.CopyOnWrite()->Update(updates);
    }
    return mod;
  };

  return tvm::transform::CreateModulePass(pass_func, 0, "tir.MakeUnpackedAPI", {});
}

TVM_REGISTER_GLOBAL("tir.transform.MakeUnpackedAPI").set_body_typed(MakeUnpackedAPI);
}  // namespace transform
}  // namespace tir
}  // namespace tvm
