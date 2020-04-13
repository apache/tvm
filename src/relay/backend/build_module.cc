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
 * \brief Code generation for TVM's graph runtime.
 */
#include <tvm/relay/analysis.h>
#include <tvm/driver/driver_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/vm.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/qnn/transform.h>
#include <tvm/tir/ir_pass.h>
#include <memory>

#include "../../target/source/codegen_source_base.h"
#include "utils.h"

namespace tvm {
namespace relay {
namespace backend {


using TargetsMap = Map<tvm::Integer, tvm::Target>;
using namespace tvm::relay::transform;

/*!
 * \brief Output of building module
 *
 */
struct BuildOutput {
  std::string graph_json;
  runtime::Module mod;
  std::unordered_map<std::string, tvm::runtime::NDArray> params;
};

/*!
 * \brief GraphCodegen module wrapper
 *
 */
struct GraphCodegen {
 public:
  GraphCodegen() {
    auto pf = GetPackedFunc("relay.build_module._GraphRuntimeCodegen");
    mod = (*pf)();
  }
  ~GraphCodegen() {}

  void Init(runtime::Module* m, TargetsMap targets) {
    CallFunc("init", m, targets);
  }

  void Codegen(const Function& func) {
    CallFunc("codegen", func);
  }

  std::string GetJSON() {
    return CallFunc<std::string>("get_graph_json", nullptr);
  }

  Array<tvm::runtime::Module> GetExternalModules() {
    return CallFunc<Array<tvm::runtime::Module>>("get_external_modules", nullptr);
  }

  Map<std::string, IRModule> GetIRModule() {
    return CallFunc<Map<std::string, IRModule>>("get_irmodule", nullptr);
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

 protected:
  tvm::runtime::Module mod;
  template<typename R, typename ...Args>
  R CallFunc(const std::string &name, Args... args) {
    auto pf = mod.GetFunction(name, false);
    return pf(std::forward<Args>(args)...);
  }
  template<typename ...Args>
  void CallFunc(const std::string &name, Args... args) {
    auto pf = mod.GetFunction(name, false);
    pf(std::forward<Args>(args)...);
    return;
  }
};

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
  PackedFunc GetFunction(const std::string& name,
                         const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "get_graph_json") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->GetGraphJSON();
      });
    } else if (name == "get_module") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->GetModule();
      });
    } else if (name == "build") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.num_args, 3);
        this->Build(args[0], args[1], args[2]);
      });
    } else if (name == "list_params") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->ListParamNames();
      });
    } else if (name == "get_params") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->GetParams();
      });
    } else if (name == "set_params") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        Map<std::string, Constant> params = args[0];
        for (const auto& kv : params) {
          this->SetParam(kv.first, kv.second->data);
        }
      });
    } else if (name == "get_irmodule") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
          *rv = this->graph_codegen_->GetIRModule();
      });
    } else if (name == "get_external_modules") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
          *rv = this->graph_codegen_->GetExternalModules();
      });
    } else if (name == "optimize") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.num_args, 2);
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
  const std::string& GetGraphJSON() {
    return ret_.graph_json;
  }

  /*!
   * \brief Get the Module object
   *
   * \return runtime::Module
   */
  runtime::Module GetModule() {
    return ret_.mod;
  }

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
   * \return Map<std::string, Constant> params dictionary
   */
  Map<std::string, Constant> GetParams() {
    Map<std::string, Constant> ret;
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
  void SetParam(const std::string& name, runtime::NDArray data_in) {
    params_[name] = data_in;
  }

  /*!
   * \brief type key
   *
   * \return const char*
   */
  const char* type_key() const final {
    return "RelayBuildModule";
  }

  /*!
   * \brief Build relay IRModule for graph runtime
   *
   * \param mod Relay IRModule
   * \param target Target device
   * \param target_host Host target device
   */
  void Build(IRModule mod,
             const TargetsMap& targets,
             const tvm::Target& target_host) {
    targets_ = targets;
    target_host_ = target_host;
    BuildRelay(mod, params_);
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
  IRModule Optimize(
      IRModule relay_module,
      const TargetsMap& targets,
      const std::unordered_map<std::string, runtime::NDArray>& params) {
    if (params.size()) {
      CHECK(relay_module->ContainGlobalVar("main"))
        << "Missing the main entry function";
      GlobalVar main_glb_var = relay_module->GetGlobalVar("main");
      Function main_func = Downcast<Function>(relay_module->Lookup(main_glb_var));
      auto new_main = BindParamsByName(main_func, params);
      relay_module->Update(main_glb_var, new_main);
    }

    Array<Pass> pass_seqs;
    Array<runtime::String> entry_functions{"main"};
    pass_seqs.push_back(transform::RemoveUnusedFunctions(entry_functions));

    // Run all dialect legalization passes.
    pass_seqs.push_back(relay::qnn::transform::Legalize());

    // Legalize pass is restricted to homogeneous execution for now.
    if (targets.size() == 1) {
      pass_seqs.push_back(transform::Legalize());
    }

    pass_seqs.push_back(transform::SimplifyInference());
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
    pass_seqs.push_back(transform::CombineParallelConv2D(3));
    pass_seqs.push_back(transform::CombineParallelDense(3));
    pass_seqs.push_back(transform::FoldConstant());
    pass_seqs.push_back(transform::FoldScaleAxis());
    pass_seqs.push_back(transform::CanonicalizeCast());
    pass_seqs.push_back(transform::CanonicalizeOps());

    // Alter layout transformation is only applied to homogeneous execution yet.
    if (targets.size() == 1) {
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
      relay_module =
          RunDeviceAnnotationPass(relay_module, pass_ctx->fallback_device);
    }

    // Fuse the operations if it is needed.
    relay_module = transform::FuseOps()(relay_module);
    relay_module = transform::InferType()(relay_module);
    // Inline the functions that have been lifted by the module scope.
    //
    // TODO(@zhiics) Note that we need to be careful about the subgraphs with
    // global function calls. We should make sure that these callees are also
    // inline functions. However, this should be very unlikely for accelerators
    // and vendor-provided libraries. So we don't handle for now.
    relay_module = transform::Inline()(relay_module);
    CHECK(relay_module.defined());

    return relay_module;
  }

  /*!
   * \brief Create a default type.
   * \param device_type The device type index.
   * \return the default target for the device.
   */
  Target CreateDefaultTarget(int device_type) {
    std::string name = runtime::DeviceName(device_type);
    if (name == "cpu") return Target::Create("llvm");
    if (name == "gpu") return Target::Create("cuda");
    return Target::Create(name);
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
  IRModule RunDeviceAnnotationPass(const IRModule& relay_module,
                                          int fallback_device) {
    UpdateHeterogeneousInputs(fallback_device);
    auto rewrite = transform::RewriteAnnotatedOps(fallback_device);
    auto updated_module = rewrite(relay_module);
    CHECK(updated_module.defined());

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
          CHECK_EQ(kv.second->value, dev_type)
            << "Expressions in the function are "
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
  void BuildRelay(
      IRModule relay_module,
      const std::unordered_map<std::string, tvm::runtime::NDArray>& params) {
    // Relay IRModule -> IRModule optimizations.
    relay_module = Optimize(relay_module, targets_, params);
    // Get the updated function.
    auto func = Downcast<Function>(relay_module->Lookup("main"));

    // Generate code for the updated function.
    graph_codegen_ = std::unique_ptr<GraphCodegen>(new GraphCodegen());
    graph_codegen_->Init(nullptr, targets_);
    graph_codegen_->Codegen(func);

    ret_.graph_json = graph_codegen_->GetJSON();
    ret_.params = graph_codegen_->GetParams();

    auto lowered_funcs = graph_codegen_->GetIRModule();

    // When there is no lowered_funcs due to reasons such as optimization.
    if (lowered_funcs.size() == 0) {
      Target target_host = GetTargetHost();

      // If no target_host has been set, we choose a default one, which is
      // llvm if "codegen.LLVMModuleCreate" is accessible.
      const runtime::PackedFunc* pf = runtime::Registry::Get("codegen.LLVMModuleCreate");
      if (!target_host.defined())
        target_host = (pf != nullptr) ? target::llvm() : target::stackvm();

      if (target_host.defined() && target_host->target_name == "llvm") {
        // If we can decide the target is LLVM, we then create an empty LLVM module.
        ret_.mod = (*pf)(target_host->str(), "empty_module");
      } else {
        // If we cannot decide the target is LLVM, we create an empty CSourceModule.
        // The code content is initialized with ";" to prevent complaining
        // from CSourceModuleNode::SaveToFile.
        ret_.mod = tvm::codegen::CSourceModuleCreate(";", "");
      }
    } else {
      ret_.mod = tvm::build(
        lowered_funcs,
        target_host_,
        BuildConfig::Current());
    }

    Array<tvm::runtime::Module> ext_mods = graph_codegen_->GetExternalModules();
    // Import all external runtime modules.
    for (const auto& it : ext_mods)
      ret_.mod.Import(it);
  }

 private:
  Target GetTargetHost() {
    Target target_host = target_host_;
    if (!target_host_.defined()) {
      for (const auto &it : targets_) {
        if (it.second->device_type == kDLCPU) {
          target_host = it.second;
          break;
        }
      }
    }
    return target_host;
  }

 protected:
  std::unique_ptr<GraphCodegen> graph_codegen_;
  /*! \brief target device */
  TargetsMap targets_;
  /*! \brief target host device */
  tvm::Target target_host_;
  /*! \brief parameters */
  std::unordered_map<std::string, runtime::NDArray> params_;
  /*! \brief building output */
  BuildOutput ret_;
};

runtime::Module RelayBuildCreate() {
  auto exec = make_object<RelayBuildModule>();
  return runtime::Module(exec);
}

TVM_REGISTER_GLOBAL("relay.build_module._BuildModule")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = RelayBuildCreate();
});

TVM_REGISTER_GLOBAL("relay.build_module.BindParamsByName")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  Map<std::string, Constant> params = args[1];
  std::unordered_map<std::string, runtime::NDArray> params_;
  for (const auto& kv : params) {
    params_[kv.first] = kv.second->data;
  }
  *rv = relay::backend::BindParamsByName(args[0], params_);
});

}  // namespace backend
}  // namespace relay
}  // namespace tvm
