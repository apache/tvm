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
 * \file tirx/ir/transform.cc
 * \brief TIR specific transformation passes.
 */
#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/rvalue_ref.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tirx {
namespace transform {

// Register build pipeline related options
TVM_FFI_STATIC_INIT_BLOCK() {
  ::tvm::transform::PassContext::RegisterConfigOption<bool>(tvm::tirx::attr::kNoAlias);
  ::tvm::transform::PassContext::RegisterConfigOption<bool>("tirx.disable_assert");
  ::tvm::transform::PassContext::RegisterConfigOption<bool>("tirx.disable_vectorize");
  ::tvm::transform::PassContext::RegisterConfigOption<bool>("tirx.enable_buffer_level_predication");
  ::tvm::transform::PassContext::RegisterConfigOption<bool>("tirx.disable_cse_tir");
  ::tvm::transform::PassContext::RegisterConfigOption<bool>("tirx.enable_debug");
  ::tvm::transform::PassContext::RegisterConfigOption<bool>("tirx.disable_storage_rewrite");
  ::tvm::transform::PassContext::RegisterConfigOption<bool>(tvm::tirx::attr::kIsEntryFunc);
  ::tvm::transform::PassContext::RegisterConfigOption<ffi::Array<ffi::Array<ffi::ObjectRef>>>(
      "tirx.add_lower_pass");
  ::tvm::transform::PassContext::RegisterConfigOption<bool>("tirx.debug_keep_trivial_loop");
  ::tvm::transform::PassContext::RegisterConfigOption<bool>("tirx.use_async_copy");
  ::tvm::transform::PassContext::RegisterConfigOption<bool>("tirx.merge_static_smem");
  ::tvm::transform::PassContext::RegisterConfigOption<int64_t>("tirx.vtcm_capacity");
  ::tvm::transform::PassContext::RegisterConfigOption<bool>("tirx.enable_fast_math");
}

/*!
 * \brief Function level pass that applies transformations to all
 *        TIR functions within the module.
 */
class FunctionPassNode : public PassNode {
 public:
  explicit FunctionPassNode(PassInfo pass_info) : pass_info(std::move(pass_info)) {}
  explicit FunctionPassNode(ffi::UnsafeInit) : pass_info(ffi::UnsafeInit{}) {}

  /* \brief The pass meta data.*/
  PassInfo pass_info;

  /*! \brief The pass function called on each. */
  std::function<ffi::Optional<Function>(Function, IRModule, PassContext)> pass_func;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FunctionPassNode>().def_ro("pass_info", &FunctionPassNode::pass_info);
  }

  /*!
   * \brief Run a function pass on given pass context.
   *
   * \param mod The module that an optimization pass is applied on.
   * \param pass_ctx The context that an optimization pass executes on.
   *
   * \return Return the updated module.
   */
  IRModule operator()(IRModule mod, const PassContext& pass_ctx) const final;

  /*!
   * \brief Get the pass information/meta data.
   */
  PassInfo Info() const override { return pass_info; }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.FunctionPass", FunctionPassNode, PassNode);
};

class FunctionPass : public Pass {
 public:
  /*!
   * \brief The constructor
   * \param pass_func The packed function which implements a pass.
   * \param pass_info The pass info.
   */
  TVM_DLL FunctionPass(
      std::function<ffi::Optional<Function>(Function, IRModule, PassContext)> pass_func,
      PassInfo pass_info);

  explicit FunctionPass(ffi::ObjectPtr<FunctionPassNode> node) : Pass(std::move(node)) {
    TVM_FFI_ICHECK(data_ != nullptr);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FunctionPass, Pass, FunctionPassNode);
};

FunctionPass::FunctionPass(
    std::function<ffi::Optional<Function>(Function, IRModule, PassContext)> pass_func,
    PassInfo pass_info)
    : Pass(ffi::UnsafeInit{}) {
  auto n = ffi::make_object<FunctionPassNode>(std::move(pass_info));
  n->pass_func = std::move(pass_func);
  data_ = std::move(n);
}

// Perform Module -> Module optimizations at the Function level.
IRModule FunctionPassNode::operator()(IRModule mod, const PassContext& pass_ctx) const {
  TVM_FFI_ICHECK(mod.defined());
  std::vector<GlobalVar> deleted_list;

  IRModuleNode* mod_ptr = mod.CopyOnWrite();
  auto* func_dict = mod_ptr->functions.CopyOnWrite();
  // directly loop over the underlying dict
  for (auto& kv : *func_dict) {
    // only picks up tirx::Function
    if (auto opt_func = kv.second.as<Function>()) {
      // reset the original Any state so the value contains only copy
      // use move semantics as follows to avoid only copy.
      kv.second.reset();
      Function func = *std::move(opt_func);
      auto updated = pass_func(std::move(func), mod, pass_ctx);
      kv.second = Any(std::move(updated));
      if (kv.second == nullptr) {
        deleted_list.push_back(kv.first.as_or_throw<GlobalVar>());
      }
    }
  }

  // Automatic removal of None.  This uses IRModuleNode::Remove
  // instead of manipulating func_dict directly, to ensure that both
  // the function map and the global_var_map_ are correctly updated.
  for (const auto& gv : deleted_list) {
    mod_ptr->Remove(gv);
  }
  return mod;
}

Pass CreateFunctionPass(
    std::function<ffi::Optional<Function>(Function, IRModule, PassContext)> pass_func,
    int opt_level, ffi::String name) {
  PassInfo pass_info = PassInfo(opt_level, name);
  return FunctionPass(std::move(pass_func), pass_info);
}

TVM_FFI_STATIC_INIT_BLOCK() { FunctionPassNode::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tirx.transform.CreateFunctionPass", [](ffi::TypedFunction<ffi::Optional<Function>(
                                                  ffi::RValueRef<Function>, IRModule, PassContext)>
                                                  pass_func,
                                              PassInfo pass_info) {
        auto wrapped_pass_func = [pass_func](Function func, IRModule mod, PassContext ctx) {
          return pass_func(ffi::RValueRef<Function>(std::move(func)), mod, ctx);
        };
        return FunctionPass(wrapped_pass_func, pass_info);
      });
}

// Pattern A (RM): auto-default repr from reflection.

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
