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
 * \file relay/ir/transform.cc
 * \brief Relay specific transformation passes.
 */
#include <dmlc/thread_local.h>
#include <tvm/node/repr_printer.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace relay {
namespace transform {

TVM_REGISTER_PASS_CONFIG_OPTION("relay.fallback_device_type", IntImm);

class FunctionPass;

/*!
 * \brief Function-level passes are used to implement various global
 * optimizations for a given Relay module. It fetches one function at a time
 * from the function list in the module for optimization.
 *
 * Note that the scope of passes at this level is a Relay function. Therefore,
 * we cannot add or delete a function through these passes as they are not aware
 * of the global information.
 */
class FunctionPassNode : public PassNode {
 public:
  /* \brief The pass meta data.*/
  PassInfo pass_info;

  /*! \brief The packed pass function sketches the real optimization. For
   * instance, we can implement a pass that works on a Relay function as a
   * `pass_func` and let it run on a given module. The same `pass_func` will
   * then be applied on each function in the module.
   */
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func;

  FunctionPassNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("pass_info", &pass_info); }

  /*!
   * \brief Run a function pass on given pass context.
   *
   * \param mod The module that an optimization pass is applied on.
   * \param mod The context that an optimization pass executes on.
   *
   * \return Return the updated module.
   */
  IRModule operator()(IRModule mod, const PassContext& pass_ctx) const final;

  /*!
   * \brief Get the pass information/meta data.
   */
  PassInfo Info() const override { return pass_info; }

  static constexpr const char* _type_key = "relay.FunctionPass";
  TVM_DECLARE_FINAL_OBJECT_INFO(FunctionPassNode, PassNode);
};

class FunctionPass : public Pass {
 public:
  /*!
   * \brief The constructor
   * \param pass_func The packed function which implements a pass.
   * \param pass_info The pass info.
   */
  TVM_DLL FunctionPass(
      runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func,
      PassInfo pass_info);

  TVM_DEFINE_OBJECT_REF_METHODS(FunctionPass, Pass, FunctionPassNode);
};

FunctionPass::FunctionPass(
    runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func,
    PassInfo pass_info) {
  auto n = make_object<FunctionPassNode>();
  n->pass_func = std::move(pass_func);
  n->pass_info = std::move(pass_info);
  data_ = std::move(n);
}

// Perform Module -> Module optimizations at the Function level.
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
  VLOG(1) << "Input module:" << std::endl << PrettyPrint(mod);

  IRModule updated_mod = mod->ShallowCopy();

  std::vector<std::pair<GlobalVar, Function>> updates;
  for (const auto& kv : mod->functions) {
    // only process optimizable Relay Functions
    if (const auto* function_node = AsOptimizableFunctionNode(kv.second)) {
      Function updated_func = pass_func(GetRef<Function>(function_node), updated_mod, pass_ctx);
      updates.push_back({kv.first, std::move(updated_func)});
    }
  }

  for (const auto& pair : updates) {
    updated_mod->Add(pair.first, pair.second, true);
  }

  ICHECK(pass_ctx->diag_ctx)
      << "The diagnostic context was set at the top of this block this is a bug.";

  pass_ctx->diag_ctx.value().Render();
  pass_ctx->diag_ctx = previous;

  VLOG(1) << "Output module:" << std::endl << PrettyPrint(updated_mod);

  // TODO(@jroesch): move away from eager type checking for performance reasons
  // make issue.
  return transform::InferType()(updated_mod);
}

Pass CreateFunctionPass(
    const runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)>& pass_func,
    int opt_level, String name, tvm::Array<String> required, bool traceable) {
  PassInfo pass_info = PassInfo(opt_level, name, required, traceable);
  return FunctionPass(pass_func, pass_info);
}

TVM_REGISTER_NODE_TYPE(FunctionPassNode);

TVM_REGISTER_GLOBAL("relay._transform.MakeFunctionPass")
    .set_body_typed(
        [](runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func,
           PassInfo pass_info) { return FunctionPass(pass_func, pass_info); });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<FunctionPassNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const FunctionPassNode*>(ref.get());
      const PassInfo info = node->Info();
      p->stream << "Run Function pass: " << info->name << " at the optimization level "
                << info->opt_level;
    });

}  // namespace transform
}  // namespace relay
}  // namespace tvm
