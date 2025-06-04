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
 * \file relax/ir/transform.cc
 * \brief Relax specific transformation passes.
 */
#include <dmlc/thread_local.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/rvalue_ref.h>
#include <tvm/node/repr_printer.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info_functor.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {
namespace transform {

TVM_REGISTER_PASS_CONFIG_OPTION("relax.fallback_device_type", IntImm);

class FunctionPass;

/*!
 * \brief Function-level passes are used to implement various global
 * optimizations for a given Relax IRModule. It fetches one function at a time
 * from the function list in the IRModule for optimization.
 *
 * Note that the scope of passes at this level is a Relax function. Therefore,
 * we cannot add or delete a function through these passes as they are not aware
 * of the global information.
 */
class FunctionPassNode : public tvm::transform::PassNode {
 public:
  /* \brief The pass meta data.*/
  PassInfo pass_info;

  /*! \brief The packed pass function sketches the real optimization. For
   * instance, we can implement a pass that works on a Relax function as a
   * `pass_func` and let it run on a given IRModule. The same `pass_func` will
   * then be applied on each function in the IRModule.
   */
  std::function<Function(Function, IRModule, PassContext)> pass_func;

  FunctionPassNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("pass_info", &pass_info); }

  /*!
   * \brief Run a function pass on given pass context.
   *
   * \param mod The IRModule that an optimization pass is applied on.
   * \param pass_ctx The context that an optimization pass executes on.
   *
   * \return Return the updated IRModule.
   */
  IRModule operator()(IRModule mod, const PassContext& pass_ctx) const final;

  /*!
   * \brief Get the pass information/meta data.
   */
  PassInfo Info() const override { return pass_info; }

  static constexpr const char* _type_key = "relax.FunctionPass";
  TVM_DECLARE_FINAL_OBJECT_INFO(FunctionPassNode, PassNode);

 private:
};

class FunctionPass : public Pass {
 public:
  /*!
   * \brief The constructor
   * \param pass_func The packed function which implements a pass.
   * \param pass_info The pass info.
   */
  TVM_DLL FunctionPass(std::function<Function(Function, IRModule, PassContext)> pass_func,
                       PassInfo pass_info);

  TVM_DEFINE_OBJECT_REF_METHODS(FunctionPass, Pass, FunctionPassNode);
};

FunctionPass::FunctionPass(std::function<Function(Function, IRModule, PassContext)> pass_func,
                           PassInfo pass_info) {
  auto n = make_object<FunctionPassNode>();
  n->pass_func = std::move(pass_func);
  n->pass_info = std::move(pass_info);
  data_ = std::move(n);
}

// Perform IRModule -> IRModule optimizations at the Function level.
IRModule FunctionPassNode::operator()(IRModule mod, const PassContext& pass_ctx) const {
  DiagnosticContext previous = DiagnosticContext::Default(mod);

  if (pass_ctx->diag_ctx) {
    DiagnosticContext tmp = pass_ctx->diag_ctx.value();
    pass_ctx->diag_ctx = previous;
    previous = tmp;
  } else {
    pass_ctx->diag_ctx = previous;
  }

  ICHECK(pass_ctx->diag_ctx)
      << "The diagnostic context was set at the top of this block this is a bug.";

  const PassInfo& pass_info = Info();

  ICHECK(mod.defined());

  VLOG_CONTEXT << pass_info->name;
  VLOG(0) << "Executing function pass with opt level: " << pass_info->opt_level;
  VLOG(1) << "Input module:" << std::endl << mod;

  IRModule updated_mod = mod->ShallowCopy();

  std::vector<std::pair<GlobalVar, Function> > updates;
  for (const auto& it : updated_mod->functions) {
    // only picks up relax::Function
    if (auto* n = it.second.as<FunctionNode>()) {
      Function func = GetRef<Function>(n);
      auto updated_func = pass_func(func, updated_mod, pass_ctx);
      updates.push_back({it.first, updated_func});
    }
  }

  for (const auto& pair : updates) {
    updated_mod->Add(pair.first, pair.second, true);
  }

  ICHECK(pass_ctx->diag_ctx)
      << "The diagnostic context was set at the top of this block, this is a bug.";

  pass_ctx->diag_ctx.value().Render();
  pass_ctx->diag_ctx = previous;

  VLOG(1) << "Output module:" << std::endl << updated_mod;

  return updated_mod;
}

Pass CreateFunctionPass(std::function<Function(Function, IRModule, PassContext)> pass_func,
                        int opt_level, String name, tvm::Array<String> required, bool traceable) {
  PassInfo pass_info = PassInfo(opt_level, name, required, traceable);
  return FunctionPass(std::move(pass_func), pass_info);
}

TVM_REGISTER_NODE_TYPE(FunctionPassNode);

TVM_FFI_REGISTER_GLOBAL("relax.transform.MakeFunctionPass")
    .set_body_typed(
        [](ffi::TypedFunction<Function(ffi::RValueRef<Function>, IRModule, PassContext)> pass_func,
           PassInfo pass_info) {
          auto wrapped_pass_func = [pass_func](Function func, IRModule mod, PassContext ctx) {
            return pass_func(ffi::RValueRef<Function>(std::move(func)), mod, ctx);
          };
          return FunctionPass(wrapped_pass_func, pass_info);
        });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<FunctionPassNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const FunctionPassNode*>(ref.get());
      const PassInfo info = node->Info();
      p->stream << "Run Function pass: " << info->name << " at the optimization level "
                << info->opt_level;
    });

class DataflowBlockPass;

/*!
 * \brief DataflowBlock-level passes are used to implement various dataflow block
 * optimizations for a given Relax IRModule. It fetches one dataflow block at a time
 * from the functions in an IRModule, and yields a rewritten DataflowBlock.
 *
 * Note that the scope of passes at this level is a Relax DataflowBlock. Therefore,
 * we cannot modify the global scope Vars and symbolic shape Vars defined inside the dataflow block.
 */
class DataflowBlockPassNode : public tvm::transform::PassNode {
 public:
  /* \brief The pass meta data.*/
  PassInfo pass_info;

  /*! \brief The packed pass function sketches the real optimization. For
   * instance, we can implement a pass that works on a Relax DataflowBlock as a
   * `pass_func` and let it run on a given IRModule. The same `pass_func` will
   * then be applied on each DataflowBlock in the IRModule.
   */
  std::function<DataflowBlock(DataflowBlock, IRModule, PassContext)> pass_func;

  DataflowBlockPassNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("pass_info", &pass_info); }

  IRModule operator()(IRModule mod, const PassContext& pass_ctx) const final;

  PassInfo Info() const override { return pass_info; }

  static constexpr const char* _type_key = "relax.DataflowBlockPass";
  TVM_DECLARE_FINAL_OBJECT_INFO(DataflowBlockPassNode, PassNode);
};

/*! \brief Helper to apply the passed function to dataflow blocks.*/
class DataflowBlockMutator : public ExprMutator {
 public:
  DataflowBlockMutator(std::function<DataflowBlock(DataflowBlock, IRModule, PassContext)> pass_func,
                       IRModule mod, PassContext pass_ctx)
      : pass_func_(pass_func), mod_(mod), pass_ctx_(pass_ctx) {}

  /*!
   * \brief Rewrite the DataflowBlockNode with pass_func_
   *
   * This function will check that there are no rewrites of the global scope Vars
   * and symbolic shape Vars defined inside the dataflow block.
   */
  BindingBlock VisitBindingBlock_(const DataflowBlockNode* n) final {
    // collect Global Scope Vars and Symbolic Vars inside the DataflowBlock
    Map<String, Var> global_scope_vars;
    Map<String, tir::Var> symbolic_vars;
    for (const Binding& binding : n->bindings) {
      Var var = binding->var;
      if (const auto* match_cast = binding.as<MatchCastNode>()) {
        auto collected_vars = SymbolicVarCollector::Collect(match_cast->struct_info);
        for (const tir::VarNode* var : collected_vars) {
          symbolic_vars.Set(var->name_hint, GetRef<tir::Var>(var));
        }
      }
      if (!var.as<DataflowVarNode>()) {
        global_scope_vars.Set(var->name_hint(), var);
      }
    }

    // apply pass_func_ to the DataflowBlock
    DataflowBlock block = GetRef<DataflowBlock>(n);
    DataflowBlock updated_block = pass_func_(block, mod_, pass_ctx_);

    // raise error if there are updates of recorded Global Scope Vars and Symbolic Vars
    for (const Binding& binding : updated_block->bindings) {
      Var var = binding->var;
      if (const auto* match_cast = binding.as<MatchCastNode>()) {
        auto collected_vars = SymbolicVarCollector::Collect(match_cast->struct_info);
        for (const tir::VarNode* var : collected_vars) {
          if (symbolic_vars.count(var->name_hint) > 0) {
            tir::Var old_var = symbolic_vars[var->name_hint];
            ICHECK(var == old_var.get())
                << "Error: DataflowBlock Pass should not rewrite any Symbolic Var.";
            symbolic_vars.erase(var->name_hint);
          }
        }
      }
      if (!var.as<DataflowVarNode>() && global_scope_vars.count(var->name_hint()) > 0) {
        ICHECK(var.same_as(global_scope_vars[var->name_hint()]))
            << "Error: DataflowBlock Pass should not rewrite any GlobalScope Var.";
        global_scope_vars.erase(var->name_hint());
      }
    }
    ICHECK(global_scope_vars.empty() && symbolic_vars.empty())
        << "Error: DataflowBlock Pass should not delete any GlobalScope/Symbolic Var.";

    return std::move(updated_block);
  }

 private:
  class SymbolicVarCollector : public StructInfoVisitor {
   public:
    static std::unordered_set<const tir::VarNode*> Collect(const StructInfo& info) {
      SymbolicVarCollector collector;
      collector.VisitStructInfo(info);
      return std::move(collector.symbolic_vars_);
    }

   private:
    void VisitStructInfoExprField(const PrimExpr& expr) final {
      if (const tir::VarNode* sym_var = expr.as<tir::VarNode>()) {
        symbolic_vars_.insert(sym_var);
      }
    }

   private:
    std::unordered_set<const tir::VarNode*> symbolic_vars_;
  };

  std::function<DataflowBlock(DataflowBlock, IRModule, PassContext)> pass_func_;
  IRModule mod_;
  PassContext pass_ctx_;
};

class DataflowBlockPass : public Pass {
 public:
  /*!
   * \brief The constructor
   * \param pass_func The packed function which implements a pass.
   * \param pass_info The pass info.
   */
  TVM_DLL DataflowBlockPass(
      std::function<DataflowBlock(DataflowBlock, IRModule, PassContext)> pass_func,
      PassInfo pass_info);

  TVM_DEFINE_OBJECT_REF_METHODS(DataflowBlockPass, Pass, DataflowBlockPassNode);
};

DataflowBlockPass::DataflowBlockPass(
    std::function<DataflowBlock(DataflowBlock, IRModule, PassContext)> pass_func,
    PassInfo pass_info) {
  auto n = make_object<DataflowBlockPassNode>();
  n->pass_func = std::move(pass_func);
  n->pass_info = std::move(pass_info);
  data_ = std::move(n);
}

// Perform IRModule -> IRModule transformations at the DataflowBlock level.
IRModule DataflowBlockPassNode::operator()(IRModule mod, const PassContext& pass_ctx) const {
  DiagnosticContext previous = DiagnosticContext::Default(mod);

  if (pass_ctx->diag_ctx) {
    DiagnosticContext tmp = pass_ctx->diag_ctx.value();
    pass_ctx->diag_ctx = previous;
    previous = tmp;
  } else {
    pass_ctx->diag_ctx = previous;
  }

  ICHECK(pass_ctx->diag_ctx)
      << "The diagnostic context was set at the top of this block, this is a bug.";

  const PassInfo& pass_info = Info();

  ICHECK(mod.defined());

  VLOG_CONTEXT << pass_info->name;
  VLOG(0) << "Executing DataflowBlock pass with opt level: " << pass_info->opt_level;
  VLOG(1) << "Input module:" << std::endl << mod;

  IRModule updated_mod = mod->ShallowCopy();

  DataflowBlockMutator dataflow_block_mutator(pass_func, updated_mod, pass_ctx);
  std::vector<std::pair<GlobalVar, Function> > updates;
  for (const auto& it : updated_mod->functions) {
    // only picks up relax::Function
    if (auto* n = it.second.as<FunctionNode>()) {
      Function func = GetRef<Function>(n);
      Function updated_func = Downcast<Function>(dataflow_block_mutator.VisitExpr(func));
      updates.push_back({it.first, updated_func});
    }
  }

  for (const auto& pair : updates) {
    updated_mod->Add(pair.first, pair.second, true);
  }

  ICHECK(pass_ctx->diag_ctx)
      << "The diagnostic context was set at the top of this block this is a bug.";

  pass_ctx->diag_ctx.value().Render();
  pass_ctx->diag_ctx = previous;

  VLOG(1) << "Output module:" << std::endl << updated_mod;

  return updated_mod;
}

Pass CreateDataflowBlockPass(
    std::function<DataflowBlock(DataflowBlock, IRModule, PassContext)> pass_func, int opt_level,
    String name, tvm::Array<String> required, bool traceable) {
  PassInfo pass_info = PassInfo(opt_level, name, required, traceable);
  return DataflowBlockPass(std::move(pass_func), pass_info);
}

TVM_REGISTER_NODE_TYPE(DataflowBlockPassNode);

TVM_FFI_REGISTER_GLOBAL("relax.transform.MakeDataflowBlockPass")
    .set_body_typed(
        [](ffi::TypedFunction<DataflowBlock(ffi::RValueRef<DataflowBlock>, IRModule, PassContext)>
               pass_func,
           PassInfo pass_info) {
          auto wrapped_pass_func = [pass_func](DataflowBlock func, IRModule mod, PassContext ctx) {
            return pass_func(ffi::RValueRef<DataflowBlock>(std::move(func)), mod, ctx);
          };
          return DataflowBlockPass(wrapped_pass_func, pass_info);
        });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<DataflowBlockPassNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const DataflowBlockPassNode*>(ref.get());
      const PassInfo info = node->Info();
      p->stream << "Run DataflowBlock pass: " << info->name << " at the optimization level "
                << info->opt_level;
    });
}  // namespace transform
}  // namespace relax
}  // namespace tvm
