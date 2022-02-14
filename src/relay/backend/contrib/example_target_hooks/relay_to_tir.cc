
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
#include <tvm/relay/attrs/call.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/memory.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>

#include "../../../op/call/call.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace example_target_hooks {

class ConvertAddToSubtract : public MixedModeMutator {
 public:
  explicit ConvertAddToSubtract(IRModule ir_module, Target host_target)
      : ir_module_(ir_module),
        host_target_(host_target),
        custom_target_(Target("example_target_hook")) {}

  IRModule Mutate() {
    GlobalVar main_global_var = ir_module_->GetGlobalVar("main");
    Function main = GetRef<Function>(ir_module_->Lookup(main_global_var).as<FunctionNode>());
    Function mutated_main = WithFields(main, main->params, VisitExpr(main->body));

    ir_module_->Update(main_global_var, mutated_main);

    return ir_module_;
  }

 private:
  tir::Load LoadIndex(const tir::Buffer& buffer, const PrimExpr& index) {
    return tir::Load(DataType::Float(32), buffer->data, index, tir::const_true());
  }

  void ReplaceAddWithSubtractPrimFunc(const GlobalVar& new_global_var, const Function& func) {
    tir::Buffer x_buffer = tir::decl_buffer({8}, DataType::Float(32), "x");
    tir::Buffer y_buffer = tir::decl_buffer({8}, DataType::Float(32), "y");
    tir::Buffer out_buffer = tir::decl_buffer({8}, DataType::Float(32));

    tir::Var x_var("x", DataType::Handle());
    tir::Var y_var("y", DataType::Handle());
    tir::Var out_var("out", DataType::Handle());

    Map<String, ObjectRef> dict_attrs;
    dict_attrs.Set("global_symbol", new_global_var->name_hint);
    dict_attrs.Set("tir.noalias", Bool(true));

    te::Var index("index", DataType::Int(32));
    tir::Sub indexed_sub = tir::Sub(LoadIndex(x_buffer, index), LoadIndex(y_buffer, index));
    tir::Stmt math_body = tir::Store(out_buffer->data, indexed_sub, index, tir::const_true());
    tir::Stmt math_loop = tir::For(index, 0, 8, tir::ForKind::kSerial, math_body);

    Map<tir::Var, tir::Buffer> buffer_map = {
        {x_var, x_buffer},
        {y_var, y_buffer},
        {out_var, out_buffer},
    };

    tir::PrimFunc replacement_func = tir::PrimFunc({x_var, y_var, out_var}, math_loop, VoidType(),
                                                   buffer_map, DictAttrs(dict_attrs));

    // Switch to TIRToRuntime hook for testing
    Bool tir_to_runtime = func->GetAttr<Bool>("tir_to_runtime").value_or(Bool(false));
    if (tir_to_runtime) {
      replacement_func = WithAttr(replacement_func, ::tvm::attr::kTarget, custom_target_);
    } else {
      replacement_func = WithAttr(replacement_func, ::tvm::attr::kTarget, host_target_);
    }

    ir_module_->Add(new_global_var, replacement_func);
  }

  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    if (const CallNode* call = post.as<CallNode>()) {
      auto* func = call->op.as<FunctionNode>();
      if (func == nullptr) {
        return post;
      }

      auto func_name = func->GetAttr<String>(::tvm::attr::kGlobalSymbol);
      if (func_name.defined() && func_name == "replace_add_with_subtract") {
        // Introduce a new global var to map the function to and copy the source type
        // over for InferType
        GlobalVar new_global_var(func_name.value());
        new_global_var->checked_type_ = func->checked_type();
        ReplaceAddWithSubtractPrimFunc(new_global_var, GetRef<Function>(func));

        // Since we are replacing the Relay function with a call to a TIR function, we must use the
        // call_lowered op.
        CallLoweredAttrs attrs;
        attrs.metadata.Set("relay_attrs", call->attrs);
        ICHECK(call->type_args.empty()) << "lowered functions cannot be polymorphic";
        return CallLowered(std::move(new_global_var), call->args, std::move(attrs), call->span);
      }
    }

    return post;
  }

 public:
  IRModule ir_module_;
  Target host_target_;
  Target custom_target_;
};

transform::Pass RelayToTIR() {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [=](IRModule ir_module, transform::PassContext pass_context) {
        auto relay_to_tir = ConvertAddToSubtract(ir_module, Target("c"));
        return relay_to_tir.Mutate();
      };
  return tvm::transform::CreateModulePass(pass_func, 0, "RelayToTIR", {});
}

}  // namespace example_target_hooks
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
