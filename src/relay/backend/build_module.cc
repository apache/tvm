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
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/qnn/transform.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/device_api.h>

#include <memory>

#include "../../target/func_registry_generator.h"
#include "../../target/source/codegen_source_base.h"
#include "compile_engine.h"
#include "utils.h"

namespace tvm {
namespace relay {
namespace transform {
Pass LabelOps();
}
namespace backend {

using TargetsMap = Map<tvm::Integer, tvm::Target>;
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
  void Init(runtime::Module* m, TargetsMap targets) { CallFunc("init", m, targets); }

  void Codegen(const Function& func, String mod_name) { CallFunc("codegen", func, mod_name); }

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

  std::unordered_map<std::string, int64_t> GetParamIds() {
    std::unordered_map<std::string, int64_t> ret;
    auto names = CallFunc<Array<runtime::String>>("list_params_name", nullptr);
    for (const auto& expr : names) {
      // Implicit cast from runtime::String to std::string
      std::string key = expr;
      ret[key] = CallFunc<int64_t>("get_param_id", key);
    }
    return ret;
  }

  Array<tvm::runtime::Module> GetExternalModules() {
    return CallFunc<Array<tvm::runtime::Module>>("get_external_modules", nullptr);
  }

  Map<String, IRModule> GetIRModule() {
    return CallFunc<Map<String, IRModule>>("get_irmodule", nullptr);
  }

  runtime::Metadata GetMetadata() { return CallFunc<runtime::Metadata>("get_metadata"); }
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
  /*!
   * \brief Get member function to front-end
   * \param name The name of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The corresponding member function.
   */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "get_graph_json") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetGraphJSON(); });
    } else if (name == "get_module") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetModule(); });
    } else if (name == "build") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.num_args, 5);
        this->Build(args[0], args[1], args[2], args[3], args[4]);
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
    } else if (name == "optimize") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.num_args, 2);
        *rv = this->Optimize(args[0], args[1], this->params_);
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

  /*!
   * \brief Build relay IRModule for graph executor
   *
   * \param mod Relay IRModule
   * \param target Target device
   * \param target_host Host target device
   */
  void Build(IRModule mod, const TargetsMap& targets, const tvm::Target& target_host,
             const String executor, const String mod_name) {
    // Create protected variable targets_ from ground up
    targets_ = targets;
    target_host_ = target_host;
    executor_ = executor;
    CheckAndUpdateHostConsistency(&targets_, &target_host_);
    BuildRelay(mod, params_, mod_name);
    // Clear compile engine so that tuning schedules can be changed between runs. See issue #6096.
    CompileEngine::Global()->Clear();
  }

 protected:
  /*!
   * \brief Optimize a Relay IRModule.
   *
   * \param relay_module The input IRModule where optmization will be applied on.
   * \param targets The device type to `Target` mapping.
   * \param params The param name to value mapping.
   *
   * \return relay::IRModule The updated Relay IR module after optimization.
   */
  IRModule Optimize(IRModule relay_module, const TargetsMap& targets,
                    const std::unordered_map<std::string, runtime::NDArray>& params) {
    ICHECK(relay_module.defined()) << "The IRModule must be defined for the Relay compiler.";

    if (params.size()) {
      ICHECK(relay_module->ContainGlobalVar("main")) << "Missing the main entry function";
      GlobalVar main_glb_var = relay_module->GetGlobalVar("main");
      Function main_func = Downcast<Function>(relay_module->Lookup(main_glb_var));
      auto new_main = BindParamsByName(main_func, params);
      IRModuleNode* relay_module_ptr = relay_module.CopyOnWrite();
      relay_module_ptr->Update(main_glb_var, new_main);
    }

    Array<Pass> pass_seqs;
    Array<runtime::String> entry_functions{"main"};
    pass_seqs.push_back(transform::RemoveUnusedFunctions(entry_functions));
    pass_seqs.push_back(transform::ToBasicBlockNormalForm());

    // Run all dialect legalization passes.
    pass_seqs.push_back(relay::qnn::transform::Legalize());

    // Legalize pass is restricted to homogeneous execution for now.
    if (targets.size() == 1) {
      pass_seqs.push_back(transform::Legalize());
    }

    pass_seqs.push_back(transform::SimplifyInference());

    // Convert Dynamic ops to static versions
    pass_seqs.push_back(transform::DynamicToStatic());

    PackedFunc fskip = PackedFunc([](TVMArgs args, TVMRetValue* rv) {
      Expr expr = args[0];
      *rv = false;
      if (expr.as<CallNode>()) {
        auto call_node = expr.as<CallNode>();
        auto op_node = call_node->op.as<OpNode>();
        if (op_node->name == "cast") {
          auto attrs = call_node->attrs.as<CastAttrs>();
          if (attrs->dtype == DataType::Int(32)) {
            *rv = true;
          }
        }
      }
    });
    pass_seqs.push_back(transform::EliminateCommonSubexpr(fskip));
    pass_seqs.push_back(transform::SimplifyExpr());
    pass_seqs.push_back(transform::CombineParallelConv2D(3));
    pass_seqs.push_back(transform::CombineParallelDense(3));
    pass_seqs.push_back(transform::CombineParallelBatchMatmul(3));
    pass_seqs.push_back(transform::FoldConstant());
    pass_seqs.push_back(transform::FoldScaleAxis());
    pass_seqs.push_back(transform::CanonicalizeCast());
    pass_seqs.push_back(transform::CanonicalizeOps());

    // Alter layout transformation is only applied to homogeneous execution yet.
    if (targets.size() == 1) {
      pass_seqs.push_back(transform::InferType());
      pass_seqs.push_back(transform::AlterOpLayout());
    }

    // Fast math optimizations.
    pass_seqs.push_back(transform::FastMath());
    pass_seqs.push_back(transform::FoldConstant());

    // Create a sequential pass and perform optimizations.
    transform::Pass seq = transform::Sequential(pass_seqs);
    if (targets.size() == 1) {
      const auto& it = targets.begin();
      With<Target> tctx((*it).second);
      relay_module = seq(relay_module);
    } else {
      relay_module = seq(relay_module);
    }

    // Handle heterogeneous compilation.
    transform::PassContext pass_ctx = PassContext::Current();
    if (targets_.size() > 1) {
      Optional<Integer> opt_fallback_dev =
          pass_ctx->GetConfig("relay.fallback_device_type", Integer(static_cast<int>(kDLCPU)));
      auto fallback_dev = opt_fallback_dev.value();
      ICHECK_GT(fallback_dev->value, 0U);
      relay_module = RunDeviceAnnotationPass(relay_module, fallback_dev->value);
    }

    // Fuse the operations if it is needed.
    relay_module = transform::FuseOps()(relay_module);

    // Do layout rewrite for auto-scheduler.
    if (backend::IsAutoSchedulerEnabled() && targets.size() == 1) {
      const auto& target = (*targets.begin()).second;
      Pass major_pass = transform::AutoSchedulerLayoutRewrite();
      bool enable_layout_rewrite_targets =
          target->kind->device_type == kDLCPU || target->GetAttr<String>("device", "") == "mali";
      if (enable_layout_rewrite_targets && pass_ctx.PassEnabled(major_pass->Info())) {
        With<Target> tctx(target);
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

    ICHECK(relay_module.defined());

    return relay_module;
  }

  /*!
   * \brief Create a default type.
   * \param device_type The device type index.
   * \return the default target for the device.
   */
  Target CreateDefaultTarget(int device_type) {
    std::string name = runtime::DeviceName(device_type);
    if (name == "cpu") return Target("llvm");
    if (name == "cuda") return Target("cuda");
    return Target(name);
  }

  /*!
   * \brief Update the target and fallback device required for heterogeneous
   * compilation. CPU is used as the fallback device if it wasn't provided.
   * Meanwhile, a CPU device type and "llvm" pair will be added to the target
   * dictionary in this case.
   *
   * \param fallback_device The fallback device for heterogeneous execution.
   */
  void UpdateHeterogeneousInputs(int fallback_device) {
    std::unordered_map<int64_t, tvm::Target> tmp_map;
    for (const auto& kv : targets_) {
      tmp_map[kv.first->value] = kv.second;
    }
    if (tmp_map.count(fallback_device) == 0) {
      targets_.Set(fallback_device, CreateDefaultTarget(fallback_device));
    }
  }

  /*!
   * \brief Execute the device annotation passes to update the input program and
   *        target information.
   *
   * \param relay_module The input Relay module.
   * \param fallback_device The fallback device for heterogeneous execution.
   *
   * \return updated_module The updated module after device annotation.
   */
  IRModule RunDeviceAnnotationPass(const IRModule& relay_module, int fallback_device) {
    UpdateHeterogeneousInputs(fallback_device);
    auto rewrite = transform::RewriteAnnotatedOps(fallback_device);
    auto updated_module = rewrite(relay_module);
    ICHECK(updated_module.defined());

    tvm::Map<Expr, Integer> device_map;
    for (const auto& it : updated_module->functions) {
      device_map = relay::CollectDeviceInfo(it.second);
      if (!device_map.empty()) break;
    }

    if (device_map.empty()) {
      tvm::Map<Expr, Integer> annotation_map;
      for (const auto& it : relay_module->functions) {
        annotation_map = relay::CollectDeviceAnnotationOps(it.second);
        if (!annotation_map.empty()) break;
      }
      // None op is annotated but they are fallen back to the default device.
      if (annotation_map.empty()) {
        targets_.Set(0, CreateDefaultTarget(fallback_device));
      } else {
        // All ops are annotated to the same device type.
        int64_t dev_type = -1;
        for (auto kv : annotation_map) {
          dev_type = kv.second->value;
          break;
        }
        for (auto kv : annotation_map) {
          ICHECK_EQ(kv.second->value, dev_type) << "Expressions in the function are "
                                                << "annotated with various device types,"
                                                << "but not device copy operators "
                                                << "found. Please check the "
                                                << "RewriteAnnotation pass.";
        }
        targets_.Set(0, CreateDefaultTarget(dev_type));
      }
    }
    return updated_module;
  }

  /*!
   * \brief Compile a Relay IR module to runtime module.
   *
   * \param relay_module The Relay IR module.
   * \param params The parameters.
   */
  void BuildRelay(IRModule relay_module,
                  const std::unordered_map<std::string, tvm::runtime::NDArray>& params,
                  const String mod_name) {
    Target target_host = GetTargetHost();
    // If no target_host has been set, we choose a default one, which is
    // llvm if "codegen.LLVMModuleCreate" is accessible.
    const runtime::PackedFunc* pf = runtime::Registry::Get("codegen.LLVMModuleCreate");
    if (!target_host.defined()) target_host = (pf != nullptr) ? Target("llvm") : Target("stackvm");

    // Update all the targets in the targets_ TargetsMap
    CheckAndUpdateHostConsistency(&targets_, &target_host);

    // Relay IRModule -> IRModule optimizations.
    relay_module = Optimize(relay_module, targets_, params);

    // Get the updated function.
    auto func = Downcast<Function>(relay_module->Lookup("main"));

    // Generate code for the updated function.
    executor_codegen_ = MakeExecutorCodegen(executor_);
    executor_codegen_->Init(nullptr, targets_);
    executor_codegen_->Codegen(func, mod_name);
    executor_codegen_->UpdateOutput(&ret_);
    ret_.params = executor_codegen_->GetParams();

    auto lowered_funcs = executor_codegen_->GetIRModule();

    // Generate a placeholder function that attaches linked params as its arguments.
    if (target_host->GetAttr<Bool>("link-params").value_or(Bool(false))) {
      CHECK(pf != nullptr) << "Unable to link-params with no target_host and no llvm codegen.";
      auto param_ids = executor_codegen_->GetParamIds();
      auto link_params = Map<String, tir::LinkedParam>();
      for (auto param : ret_.params) {
        link_params.Set(param.first, tir::LinkedParam(param_ids[param.first], param.second));
      }

      Map<String, ObjectRef> dict;
      dict.Set(tvm::tir::attr::kLinkedParams, link_params);
      dict.Set(tvm::attr::kGlobalSymbol, String(::tvm::runtime::symbol::tvm_lookup_linked_param));
      DictAttrs attrs{dict};
      auto prim = tir::PrimFunc(Array<tir::Var>(), tir::SeqStmt(Array<tir::Stmt>()), VoidType(),
                                Map<tir::Var, tir::Buffer>(), attrs);
      if (lowered_funcs.find(target_host->str()) == lowered_funcs.end()) {
        lowered_funcs.Set(target_host->str(), IRModule(Map<GlobalVar, BaseFunc>({})));
      }
      lowered_funcs[target_host->str()]->Add(
          GlobalVar(::tvm::runtime::symbol::tvm_lookup_linked_param), prim);
    }

    // When there is no lowered_funcs due to reasons such as optimization.
    if (lowered_funcs.size() == 0) {
      if (target_host.defined() && target_host->kind->name == "llvm") {
        // If we can decide the target is LLVM, we then create an empty LLVM module.
        ret_.mod = (*pf)(target_host->str(), "empty_module");
      } else {
        // If we cannot decide the target is LLVM, we create an empty CSourceModule.
        // The code content is initialized with ";" to prevent complaining
        // from CSourceModuleNode::SaveToFile.
        ret_.mod = tvm::codegen::CSourceModuleCreate(";", "", Array<String>{});
      }
    } else {
      ret_.mod = tvm::build(lowered_funcs, target_host_);
    }

    auto ext_mods = executor_codegen_->GetExternalModules();
    ret_.mod = tvm::codegen::CreateMetadataModule(ret_.params, ret_.mod, ext_mods, GetTargetHost(),
                                                  executor_codegen_->GetMetadata());
    // Remove external params which were stored in metadata module.
    for (tvm::runtime::Module mod : ext_mods) {
      auto pf_var = mod.GetFunction("get_const_vars");
      if (pf_var != nullptr) {
        Array<String> variables = pf_var();
        for (size_t i = 0; i < variables.size(); i++) {
          auto it = ret_.params.find(variables[i].operator std::string());
          if (it != ret_.params.end()) {
            ret_.params.erase(it);
          }
        }
      }
    }
  }

 private:
  Target GetTargetHost() {
    Target target_host = target_host_;
    if (!target_host_.defined()) {
      for (const auto& it : targets_) {
        if (it.second->kind->device_type == kDLCPU) {
          target_host = it.second;
          break;
        }
      }
    }
    return target_host;
  }

 protected:
  std::unique_ptr<ExecutorCodegen> executor_codegen_;
  /*! \brief target device */
  TargetsMap targets_;
  /*! \brief target host device */
  tvm::Target target_host_;
  /*! \brief parameters */
  std::unordered_map<std::string, runtime::NDArray> params_;
  /*! \brief building output */
  BuildOutput ret_;
  /*!
   * \brief Executor used to execute the model:
   * - graph: use the json graph executor
   * - aot: use the aot executor
   */
  String executor_;
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
