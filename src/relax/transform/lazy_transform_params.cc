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

/*! \file src/relax/transform/lazy_transform_params.cc */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include <optional>
#include <unordered_map>

#include "utils.h"

namespace tvm {
namespace relax {

namespace {
std::optional<int64_t> GetNumInputParams(const FunctionNode* func) {
  if (auto opt_int_imm = func->GetAttr<IntImm>(attr::kNumInput)) {
    int64_t num_input_params = opt_int_imm.value()->value;
    CHECK_GE(num_input_params, 0) << "ValueError: "
                                  << "Annotation for attr::kNumInput (\"" << attr::kNumInput
                                  << "\") must be non-negative, but was " << num_input_params;
    CHECK_LE(static_cast<size_t>(num_input_params), func->params.size())
        << "ValueError: "
        << "Annotation for attr::kNumInput (\"" << attr::kNumInput << "\") specifies "
        << num_input_params << " parameters to be provided at runtime, "
        << "but the function only accepts " << func->params.size() << " parameters in total";
    return num_input_params;
  } else {
    return std::nullopt;
  }
}

class LazyInputMutator : public ExprMutator {
 public:
  Expr VisitExpr_(const FunctionNode* func) override {
    if (plan_.has_value()) {
      return ExprMutator::VisitExpr_(func);
    }

    int64_t num_input_params = GetNumInputParams(func).value_or(0);

    std::unordered_map<Var, size_t> param_lookup;
    for (size_t i = num_input_params; i < func->params.size(); i++) {
      param_lookup.insert({func->params[i], i - num_input_params});
    }

    Var fget_param("fget_param",
                   FuncStructInfo({PrimStructInfo(DataType::Int(64)), ObjectStructInfo()},
                                  ObjectStructInfo()));

    Array<Var> new_params(func->params.begin(), func->params.begin() + num_input_params);
    new_params.push_back(fget_param);

    auto array_externally_visible_vars =
        DefinableTIRVarsInStructInfo(TupleStructInfo(new_params.Map(GetStructInfo)));
    std::unordered_set<tir::Var> externally_visible_vars(array_externally_visible_vars.begin(),
                                                         array_externally_visible_vars.end());
    StructInfo new_ret_struct_info =
        EraseToWellDefined(func->ret_struct_info, [&](const tir::Var& var) -> Optional<PrimExpr> {
          if (externally_visible_vars.count(var)) {
            return var;
          } else {
            return NullOpt;
          }
        });

    auto node = GetRef<Function>(func);
    node.CopyOnWrite()->params = new_params;
    node.CopyOnWrite()->ret_struct_info = new_ret_struct_info;
    node = WithAttr(node, attr::kNumInput, Integer(num_input_params + 1));

    plan_ = FunctionPlan{std::move(param_lookup), fget_param};
    auto output = Downcast<Function>(ExprMutator::VisitExpr_(node.get()));
    plan_.reset();
    return output;
  }

  Expr VisitExpr_(const VarNode* op) override {
    if (plan_) {
      Var var = GetRef<Var>(op);
      if (auto it = plan_->param_lookup.find(var); it != plan_->param_lookup.end()) {
        auto untyped =
            builder_->Emit(relax::Call(plan_->fget_param,
                                       {
                                           PrimValue(IntImm(DataType::Int(64), it->second)),
                                           StringImm(var->name_hint()),
                                       }),
                           var->name_hint() + "_untyped");
        return builder_->EmitMatchCast(untyped, GetStructInfo(var), var->name_hint());
      }
    }

    return ExprMutator::VisitExpr_(op);
  }

 private:
  struct FunctionPlan {
    std::unordered_map<Var, size_t> param_lookup;
    Expr fget_param;
  };
  std::optional<FunctionPlan> plan_;
};

class LazyOutputMutator : public ExprMutator {
 public:
  Expr VisitExpr_(const FunctionNode* func) override {
    if (plan_.has_value()) {
      return ExprMutator::VisitExpr_(func);
    }

    std::unordered_map<Var, std::vector<size_t>> output_lookup;
    std::vector<std::tuple<size_t, Expr>> inline_outputs;
    auto define_lookup = [&](size_t output_index, Expr output_value) {
      if (auto var = output_value.as<Var>()) {
        output_lookup[var.value()].push_back(output_index);
      } else {
        inline_outputs.push_back({output_index, output_value});
      }
    };

    auto func_body = Downcast<SeqExpr>(func->body);
    if (auto tuple_output = func_body->body.as<TupleNode>()) {
      for (size_t i = 0; i < tuple_output->fields.size(); i++) {
        define_lookup(i, tuple_output->fields[i]);
      }
    } else {
      define_lookup(0, func_body->body);
    }

    Var fset_output("fset_output",
                    FuncStructInfo({PrimStructInfo(DataType::Int(64)), ObjectStructInfo()},
                                   TupleStructInfo(Array<StructInfo>{}), /* purity = */ false));
    plan_ = FunctionPlan{std::move(output_lookup), fset_output};

    std::optional<int64_t> num_input_params = GetNumInputParams(func);

    auto new_params = func->params;
    new_params.insert(new_params.begin() + num_input_params.value_or(func->params.size()),
                      fset_output);

    BindingBlock start_of_func = [&]() {
      Array<Binding> propagated_params;
      for (auto param : func->params) {
        GenerateSetOutputCalls(param, [&](const auto& fset_output_call) {
          Var void_output("_void", TupleStructInfo(Array<StructInfo>{}));
          propagated_params.push_back(VarBinding(void_output, fset_output_call));
        });
      }
      return BindingBlock(propagated_params);
    }();
    BindingBlock end_of_func = [&]() {
      Array<Binding> propagated_params;
      for (const auto& [output_index, expr] : inline_outputs) {
        Call fset_output_call(fset_output,
                              {PrimValue(IntImm(DataType::Int(64), output_index)), expr});
        Var void_output("_void", TupleStructInfo(Array<StructInfo>{}));
        propagated_params.push_back(VarBinding(void_output, fset_output_call));
      }
      return BindingBlock(propagated_params);
    }();

    Array<BindingBlock> new_blocks = func_body->blocks;
    new_blocks.insert(new_blocks.begin(), start_of_func);
    new_blocks.push_back(end_of_func);
    Expr new_body = SeqExpr(new_blocks, Tuple(Array<Expr>{}));

    auto node = GetRef<Function>(func);
    {
      auto write_ptr = node.CopyOnWrite();
      write_ptr->params = new_params;
      write_ptr->body = new_body;
      write_ptr->is_pure = false;
    }
    if (num_input_params.has_value()) {
      node = WithAttr(node, attr::kNumInput, Integer(num_input_params.value() + 1));
    }

    auto output = Downcast<Function>(ExprMutator::VisitExpr_(node.get()));
    plan_.reset();
    return output;
  }

  void VisitBinding(const Binding& binding) override {
    ExprMutator::VisitBinding(binding);
    GenerateSetOutputCalls(binding->var, [this](const auto& fset_output_call) {
      builder_->Emit(fset_output_call, "_void");
    });
  }

 private:
  template <typename Callback>
  void GenerateSetOutputCalls(const Var& var, Callback callback) {
    if (plan_.has_value()) {
      if (auto it = plan_->output_lookup.find(var); it != plan_->output_lookup.end()) {
        for (auto output_index : it->second) {
          callback(
              Call(plan_->fset_output, {PrimValue(IntImm(DataType::Int(64), output_index)), var}));
        }
      }
    }
  }

  struct FunctionPlan {
    std::unordered_map<Var, std::vector<size_t>> output_lookup;
    Expr fset_output;
  };
  std::optional<FunctionPlan> plan_;
};
}  // namespace

Function WithLazyInputs(Function func) {
  LazyInputMutator mutator;

  func = Downcast<Function>(mutator.VisitExpr(func));
  func = Downcast<Function>(EliminateCommonSubexpr(func));
  func = Downcast<Function>(RemoveAllUnused(func));
  return func;
}

Function WithLazyOutputs(Function func) {
  LazyOutputMutator mutator;

  func = Downcast<Function>(mutator.VisitExpr(func));
  return func;
}

namespace transform {

Pass LazyGetInput() {
  auto pass_func = [](Function func, IRModule, PassContext) -> Function {
    if (!func->GetAttr<String>(tvm::attr::kGlobalSymbol).defined()) {
      return func;
    }
    return WithLazyInputs(func);
  };
  return CreateFunctionPass(/*pass_function=*/pass_func,
                            /*opt_level=*/0,
                            /*pass_name=*/"LazyGetInput",
                            /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.LazyGetInput").set_body_typed(LazyGetInput);

Pass LazySetOutput() {
  auto pass_func = [](Function func, IRModule, PassContext) -> Function {
    if (!func->GetAttr<String>(tvm::attr::kGlobalSymbol).defined()) {
      return func;
    }
    return WithLazyOutputs(func);
  };
  return CreateFunctionPass(/*pass_function=*/pass_func,
                            /*opt_level=*/0,
                            /*pass_name=*/"LazySetOutput",
                            /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.LazySetOutput").set_body_typed(LazySetOutput);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
