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
 * \file relay/backend/build_module.cc
 * \brief Code generation for TVM's graph executor.
 */
#include <tvm/driver/driver_api.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/memory_pools.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/executor.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/qnn/transform.h>
#include <tvm/relay/runtime.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/device_api.h>
#include <tvm/target/compilation_config.h>

#include <memory>

#include "../../driver/internal_driver_api.h"
#include "../../target/func_registry_generator.h"
#include "../../target/metadata_module.h"
#include "../../target/source/codegen_source_base.h"
#include "te_compiler.h"
#include "utils.h"

namespace tvm {
namespace relay {
namespace transform {
Pass LabelOps();
}
namespace backend {

using namespace tvm::relay::transform;

/*!
 * \brief Output of building module
 */
struct BuildOutput {
  std::string graph_json;
  runtime::Module mod;
  std::unordered_map<std::string, tvm::runtime::NDArray> params;
};

struct ExecutorCodegen {
  void Init(runtime::Module* m, const Array<Target>& raw_targets) {
    CallFunc("init", m, raw_targets);
  }

  void Codegen(IRModule mod, const Function& func, String mod_name) {
    CallFunc("codegen", mod, func, mod_name);
  }

  virtual void UpdateOutput(BuildOutput* ret) = 0;

  Map<String, FunctionInfo> GetFunctionMetadata() {
    return CallFunc<Map<String, FunctionInfo>>("get_function_metadata", nullptr);
  }

  std::unordered_map<std::string, tvm::runtime::NDArray> GetParams() {
    std::unordered_map<std::string, tvm::runtime::NDArray> ret;
    auto names = CallFunc<Array<runtime::String>>("list_params_name", nullptr);
    for (const auto& expr : names) {
      // Implicit cast from runtime::String to std::string
      std::string key = expr;
      ret[key] = CallFunc<runtime::NDArray>("get_param_by_name", key);
    }
    return ret;
  }

  Array<tvm::runtime::Module> GetExternalModules() {
    return CallFunc<Array<tvm::runtime::Module>>("get_external_modules", nullptr);
  }

  Map<Target, IRModule> GetIRModule() {
    return CallFunc<Map<Target, IRModule>>("get_irmodule", nullptr);
  }

  Array<String> ListDevices() { return CallFunc<Array<String>>("get_devices"); }

  relay::backend::ExecutorCodegenMetadata GetExecutorCodegenMetadata() {
    return CallFunc<relay::backend::ExecutorCodegenMetadata>("get_executor_codegen_metadata");
  }
  virtual ~ExecutorCodegen() {}

 protected:
  tvm::runtime::Module mod;
  template <typename R, typename... Args>
  R CallFunc(const std::string& name, Args... args) {
    auto pf = mod.GetFunction(name, false);
    return pf(std::forward<Args>(args)...);
  }
  template <typename... Args>
  void CallFunc(const std::string& name, Args... args) {
    auto pf = mod.GetFunction(name, false);
    pf(std::forward<Args>(args)...);
    return;
  }
};

struct AOTCodegen : ExecutorCodegen {
  AOTCodegen() {
    auto pf = GetPackedFunc("relay.build_module._AOTExecutorCodegen");
    mod = (*pf)();
  }

  void UpdateOutput(BuildOutput* ret) override { ret->graph_json = ""; }

  ~AOTCodegen() {}
};

/*!
 * \brief GraphCodegen module wrapper
 *
 */
struct GraphCodegen : ExecutorCodegen {
  GraphCodegen() {
    auto pf = GetPackedFunc("relay.build_module._GraphExecutorCodegen");
    mod = (*pf)();
  }
  void UpdateOutput(BuildOutput* ret) override { ret->graph_json = GetGraphJSON(); }

  std::string GetGraphJSON() { return CallFunc<std::string>("get_graph_json", nullptr); }

  ~GraphCodegen() {}
};

/*!
 * \brief Executor codegen factory function
 */
std::unique_ptr<ExecutorCodegen> MakeExecutorCodegen(String executor_str) {
  std::unique_ptr<ExecutorCodegen> ret;
  if (executor_str == runtime::kTvmExecutorGraph) {
    ret = std::make_unique<GraphCodegen>();
  } else if (executor_str == runtime::kTvmExecutorAot) {
    ret = std::make_unique<AOTCodegen>();
  } else {
    CHECK(false) << "Executor " << executor_str << " not supported";
  }
  return ret;
}

/*!
 * \brief Relay build module
 *
 */
class RelayBuildModule : public runtime::ModuleNode {
 public:
  RelayBuildModule() = default;

  /*!
   * \brief Get member function to front-end
   * \param name The name of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The corresponding member function.
   */
  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "get_graph_json") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetGraphJSON(); });
    } else if (name == "get_module") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetModule(); });
    } else if (name == "build") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.num_args, 8);
        this->Build(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);
      });
    } else if (name == "list_params") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->ListParamNames(); });
    } else if (name == "get_params") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetParams(); });
    } else if (name == "set_params") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        Map<String, Constant> params = args[0];
        for (const auto& kv : params) {
          this->SetParam(kv.first, kv.second->data);
        }
      });
    } else if (name == "get_devices") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->executor_codegen_->ListDevices();
      });
    } else if (name == "get_irmodule") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->executor_codegen_->GetIRModule();
      });
    } else if (name == "get_external_modules") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->executor_codegen_->GetExternalModules();
      });
    } else if (name == "get_function_metadata") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->executor_codegen_->GetFunctionMetadata();
      });
    } else if (name == "get_executor_codegen_metadata") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->executor_codegen_->GetExecutorCodegenMetadata();
      });
    } else if (name == "optimize") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.num_args, 2);
        *rv = this->Optimize(args[0], args[1]);
      });
    } else {
      LOG(FATAL) << "Unknown packed function: " << name;
      return PackedFunc([sptr_to_self, name](TVMArgs args, TVMRetValue* rv) {});
    }
  }

  /*!
   * \brief Get the GraphJSON for runtime
   *
   * \return const std::string graph_json
   */
  const std::string& GetGraphJSON() { return ret_.graph_json; }

  /*!
   * \brief Get the Module object
   *
   * \return runtime::Module
   */
  runtime::Module GetModule() { return ret_.mod; }

  /*!
   * \brief List all paramter names
   *
   * \return Array<runtime::String> names of params
   */
  Array<runtime::String> ListParamNames() {
    Array<runtime::String> ret;
    for (const auto& kv : params_) {
      ret.push_back(kv.first);
    }
    return ret;
  }

  /*!
   * \brief Get params dictionary
   *
   * \return Map<String, Constant> params dictionary
   */
  Map<String, Constant> GetParams() {
    Map<String, Constant> ret;
    for (const auto& kv : ret_.params) {
      ret.Set(kv.first, Constant(kv.second));
    }
    return ret;
  }

  /*!
   * \brief Set the parameters
   *
   * \param name name of parameter
   * \param data_in input DLTensor
   */
  void SetParam(const std::string& name, runtime::NDArray data_in) { params_[name] = data_in; }

  /*!
   * \brief type key
   *
   * \return const char*
   */
  const char* type_key() const final { return "RelayBuildModule"; }

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final { return runtime::ModulePropertyMask::kRunnable; }

  /*!
   * \brief Build relay IRModule for graph executor
   *
   * \param mod Relay IRModule
   * \param raw_targets List of available targets for kernels.
   * \param executor Executor to target
   * \param runtime Runtime to codegen for
   * \param mod_name Name of the module
   */
  void Build(IRModule mod, const Array<Target>& raw_targets, const tvm::Target& target_host,
             const Executor& executor, const Runtime& runtime,
             const WorkspaceMemoryPools& workspace_memory_pools,
             const ConstantMemoryPools& constant_memory_pools, const String mod_name) {
    VLOG_CONTEXT << "Build";
    executor_ = executor;
    runtime_ = runtime;
    workspace_memory_pools_ = workspace_memory_pools;
    constant_memory_pools_ = constant_memory_pools;
    config_ = CompilationConfig(PassContext::Current(), raw_targets);
    VLOG(1) << "Using compilation config:" << std::endl << config_;
    BuildRelay(std::move(mod), mod_name);
  }

 protected:
  /*!
   * \brief Optimize a Relay IRModule.
   *
   * \param relay_module The input IRModule where optmization will be applied on.
   * \param raw_targets List of available targets for kernels.
   *
   * \return relay::IRModule The updated Relay IR module after optimization.
   */
  IRModule Optimize(IRModule relay_module, const Array<Target>& raw_targets) {
    VLOG_CONTEXT << "Optimize";
    config_ = CompilationConfig(PassContext ::Current(), raw_targets);
    VLOG(1) << "Using compilation config:" << std::endl << config_;
    return OptimizeImpl(std::move(relay_module));
  }

  IRModule OptimizeImpl(IRModule relay_module) {
    ICHECK(relay_module.defined()) << "The IRModule must be defined for the Relay compiler.";

    backend::BindParamsInModule(relay_module, params_);

    Array<Pass> pass_seqs =
        GetPassPrefix(/*is_homogenous=*/config_->primitive_targets.size() == 1, /*is_vm=*/false);
    transform::PassContext pass_ctx = PassContext::Current();

    if (config_->optional_homogeneous_target.defined()) {
      // This pass currently only supports the homogeneous case.
      pass_seqs.push_back(transform::SplitArgs(
          config_->optional_homogeneous_target->GetAttr<Integer>("max_function_args", 0)
              .value()
              .IntValue()));
    }

    // Always plan devices so the remaining passes don't need to distinguish homogeneous vs
    // hetrogenous execution.
    pass_seqs.push_back(transform::PlanDevices(config_));

    // Fuse the operations if it is needed.
    pass_seqs.push_back(transform::FuseOps());

    // Create a sequential pass and perform optimizations.
    transform::Pass seq = transform::Sequential(pass_seqs);
    if (config_->optional_homogeneous_target.defined()) {
      With<Target> tctx(config_->optional_homogeneous_target);
      relay_module = seq(relay_module);
    } else {
      relay_module = seq(relay_module);
    }

    // Do layout rewrite for auto-scheduler.
    if (backend::IsAutoSchedulerEnabled() && config_->optional_homogeneous_target.defined()) {
      Pass major_pass = transform::AutoSchedulerLayoutRewrite();
      bool enable_layout_rewrite_targets =
          config_->optional_homogeneous_target->GetTargetDeviceType() == kDLCPU ||
          config_->optional_homogeneous_target->GetAttr<String>("device", "") == "mali";
      if (enable_layout_rewrite_targets && pass_ctx.PassEnabled(major_pass->Info())) {
        With<Target> tctx(config_->optional_homogeneous_target);
        relay_module = major_pass(relay_module);
        // Defuse ops to fold constants, then fuse them again
        relay_module = transform::DefuseOps()(relay_module);
        relay_module = transform::FoldConstant()(relay_module);
        relay_module = transform::FuseOps()(relay_module);
      }
    }
    if (backend::IsMetaScheduleEnabled() && config_->optional_homogeneous_target.defined()) {
      Pass major_pass = transform::MetaScheduleLayoutRewrite();
      bool enable_layout_rewrite_targets =
          config_->optional_homogeneous_target->GetTargetDeviceType() == kDLCPU ||
          config_->optional_homogeneous_target->GetAttr<String>("device", "") == "mali";
      if (enable_layout_rewrite_targets && pass_ctx.PassEnabled(major_pass->Info())) {
        With<Target> tctx(config_->optional_homogeneous_target);
        relay_module = major_pass(relay_module);
        // Defuse ops to fold constants, then fuse them again
        relay_module = transform::DefuseOps()(relay_module);
        relay_module = transform::FoldConstant()(relay_module);
        relay_module = transform::FuseOps()(relay_module);
      }
    }

    relay_module = transform::InferType()(relay_module);

    // Inline the functions that have been lifted by the module scope.
    //
    // TODO(@zhiics) Note that we need to be careful about the subgraphs with
    // global function calls. We should make sure that these callees are also
    // inline functions. However, this should be very unlikely for accelerators
    // and vendor-provided libraries. So we don't handle for now.
    relay_module = transform::Inline()(relay_module);
    relay_module = transform::InferType()(relay_module);
    relay_module = transform::LabelOps()(relay_module);
    relay_module = transform::AnnotateMemoryScope()(relay_module);

    ICHECK(relay_module.defined());

    return relay_module;
  }

  /*!
   * \brief Compile a Relay IR module to runtime module.
   *
   * \param relay_module The Relay IR module.
   * \param params The parameters.
   */
  void BuildRelay(IRModule relay_module, const String& mod_name) {
    // Relay IRModule -> IRModule optimizations.
    IRModule module = WithAttrs(
        relay_module, {{tvm::attr::kExecutor, executor_}, {tvm::attr::kRuntime, runtime_}});
    relay_module = OptimizeImpl(std::move(module));

    // Get the updated function and new IRModule to build.
    // Instead of recreating the IRModule, we should look at the differences between this and the
    // incoming IRModule to see if we can just pass (IRModule, Function) to the code generator.
    Function func = Downcast<Function>(relay_module->Lookup("main"));
    IRModule func_module = WithAttrs(IRModule::FromExpr(func),
                                     {{tvm::attr::kExecutor, executor_},
                                      {tvm::attr::kRuntime, runtime_},
                                      {tvm::attr::kWorkspaceMemoryPools, workspace_memory_pools_},
                                      {tvm::attr::kConstantMemoryPools, constant_memory_pools_}});

    // Generate code for the updated function.
    executor_codegen_ = MakeExecutorCodegen(executor_->name);
    executor_codegen_->Init(nullptr, config_->primitive_targets);
    executor_codegen_->Codegen(func_module, func, mod_name);
    executor_codegen_->UpdateOutput(&ret_);
    ret_.params = executor_codegen_->GetParams();

    auto lowered_funcs = executor_codegen_->GetIRModule();

    // No need to build for external functions.
    Target ext_dev("ext_dev");
    if (lowered_funcs.find(ext_dev) != lowered_funcs.end()) {
      lowered_funcs.Set(ext_dev, IRModule());
    }

    const Target& host_target = config_->host_virtual_device->target;
    const runtime::PackedFunc* pf = runtime::Registry::Get("codegen.LLVMModuleCreate");
    // When there is no lowered_funcs due to reasons such as optimization.
    if (lowered_funcs.size() == 0) {
      if (host_target->kind->name == "llvm") {
        CHECK(pf != nullptr) << "Unable to create empty module for llvm without llvm codegen.";
        // If we can decide the target is LLVM, we then create an empty LLVM module.
        ret_.mod = (*pf)(host_target->str(), "empty_module");
      } else {
        // If we cannot decide the target is LLVM, we create an empty CSourceModule.
        // The code content is initialized with ";" to prevent complaining
        // from CSourceModuleNode::SaveToFile.
        ret_.mod = tvm::codegen::CSourceModuleCreate(";", "", Array<String>{});
      }
    } else {
      ret_.mod = tvm::TIRToRuntime(lowered_funcs, host_target);
    }

    auto ext_mods = executor_codegen_->GetExternalModules();
    ret_.mod = tvm::codegen::CreateMetadataModule(ret_.params, ret_.mod, ext_mods, host_target,
                                                  runtime_, executor_,
                                                  executor_codegen_->GetExecutorCodegenMetadata());
    // Remove external params which were stored in metadata module.
    for (tvm::runtime::Module mod : ext_mods) {
      auto pf_var = mod.GetFunction("get_const_vars");
      if (pf_var != nullptr) {
        Array<String> variables = pf_var();
        for (size_t i = 0; i < variables.size(); i++) {
          auto it = ret_.params.find(variables[i].operator std::string());
          if (it != ret_.params.end()) {
            VLOG(1) << "constant '" << variables[i] << "' has been captured in external module";
            ret_.params.erase(it);
          }
        }
      }
    }
  }

 protected:
  std::unique_ptr<ExecutorCodegen> executor_codegen_;
  /*! \brief Executor to build for */
  Executor executor_;
  /*! \brief Runtime to codegen for */
  Runtime runtime_;
  /*! \brief Workspace memory pools to codegen for */
  WorkspaceMemoryPools workspace_memory_pools_;
  /*! \brief Constant memory pools to codegen for */
  ConstantMemoryPools constant_memory_pools_;
  /*! \brief parameters */
  std::unordered_map<std::string, runtime::NDArray> params_;
  /*! \brief building output */
  BuildOutput ret_;
  /*! \brief Collects all the targets and scopes we need during compilation. */
  CompilationConfig config_;
};

runtime::Module RelayBuildCreate() {
  auto exec = make_object<RelayBuildModule>();
  return runtime::Module(exec);
}

TVM_REGISTER_GLOBAL("relay.build_module._BuildModule").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = RelayBuildCreate();
});

TVM_REGISTER_GLOBAL("relay.build_module.BindParamsByName")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      Map<String, Constant> params = args[1];
      std::unordered_map<std::string, runtime::NDArray> params_;
      for (const auto& kv : params) {
        params_[kv.first] = kv.second->data;
      }
      *rv = relay::backend::BindParamsByName(args[0], params_);
    });

}  // namespace backend
}  // namespace relay
}  // namespace tvm
