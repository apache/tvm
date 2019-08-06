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
#include <tvm/build_module.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/vm.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/transform.h>
#include <memory>

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

  Map<std::string, Array<LoweredFunc> > GetLoweredFunc() {
    return CallFunc<Map<std::string, Array<LoweredFunc> > >("get_lowered_funcs", nullptr);
  }

  std::unordered_map<std::string, tvm::runtime::NDArray> GetParams() {
    std::unordered_map<std::string, tvm::runtime::NDArray> ret;
    auto names = CallFunc<Array<tvm::Expr> >("list_params_name", nullptr);
    for (auto expr : names) {
      auto key = expr.as<ir::StringImm>()->value;
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
                         const std::shared_ptr<ModuleNode>& sptr_to_self) final {
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
   * \return Array<StringImm> names of params
   */
  Array<tvm::Expr> ListParamNames() {
    Array<tvm::Expr> ret;
    for (const auto& kv : params_) {
      ret.push_back(ir::StringImm::make(kv.first));
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
      ret.Set(kv.first, ConstantNode::make(kv.second));
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
   * \brief Build relay function for graph runtime
   *
   * \param func Relay Function
   * \param target Target device
   * \param target_host Host target device
   */
  void Build(Function func,
             const TargetsMap& targets,
             const tvm::Target& target_host) {
    targets_ = targets;
    target_host_ = target_host;
    BuildRelay(func, params_);
  }

 protected:
  /*!
   * \brief Bind params to function by using name
   * \param func Relay function
   * \param params params dict
   * \return relay::Function
   */
  relay::Function BindParamsByName(
      relay::Function func,
      const std::unordered_map<std::string, runtime::NDArray>& params) {
    std::unordered_map<std::string, relay::Var> name_dict;
    std::unordered_set<relay::Var, NodeHash, NodeEqual> repeat_var;
    for (auto arg : func->params) {
      const auto &name = arg->name_hint();
      if (name_dict.count(name)) {
        repeat_var.insert(arg);
      } else {
        name_dict[name] = arg;
      }
    }

    std::unordered_map<relay::Var, Expr, NodeHash, NodeEqual> bind_dict;
    for (auto &kv : params) {
      if (name_dict.count(kv.first) == 0) {
        continue;
      }
      auto arg = name_dict.at(kv.first);
      if (repeat_var.count(arg)) {
        LOG(FATAL) << "Multiple args in the function have name " << kv.first;
      }
      bind_dict[arg] = ConstantNode::make(kv.second);
    }
    Expr bound_expr = relay::Bind(func, bind_dict);
    Function ret = Downcast<Function>(bound_expr);
    CHECK(ret.defined())
        << "The returning type is expected to be a Relay Function."
        << "\n";
    return ret;
  }

  /*!
   * \brief Optimize a Relay module.
   *
   * \param relay_module The input Relay module where optmization will be
   *        applied on.
   * \param targets The device type to `Target` mapping.
   * \param params The param name to value mapping.
   *
   * \return relay::Module The updated Relay module after optimization.
   */
  relay::Module Optimize(
      relay::Module relay_module,
      const TargetsMap& targets,
      const std::unordered_map<std::string, runtime::NDArray>& params) {
    Array<Pass> pass_seqs;
    pass_seqs.push_back(transform::SimplifyInference());
    PackedFunc fskip = PackedFunc([](TVMArgs args, TVMRetValue* rv) {
      Expr expr = args[0];
      if (expr.as<CallNode>()) {
        auto call_node = expr.as<CallNode>();
        auto op_node = call_node->op.as<OpNode>();
        if (op_node->name == "cast") {
          auto attrs = call_node->attrs.as<CastAttrs>();
          if (attrs->dtype == Int(32)) {
            *rv = true;
          }
        }
      }
      *rv = false;
    });
    pass_seqs.push_back(transform::EliminateCommonSubexpr(fskip));
    pass_seqs.push_back(transform::CombineParallelConv2D(3));
    pass_seqs.push_back(transform::FoldConstant());
    pass_seqs.push_back(transform::FoldScaleAxis());
    pass_seqs.push_back(transform::CanonicalizeCast());
    pass_seqs.push_back(transform::CanonicalizeOps());

    // Legalize pass is restricted to homogeneous execution for now.
    if (targets.size() == 1) {
      pass_seqs.push_back(transform::Legalize());
    }

    // Alter layout transformation is only applied to homogeneous execution yet.
    if (targets.size() == 1) {
      pass_seqs.push_back(transform::AlterOpLayout());
    }
    pass_seqs.push_back(transform::FoldConstant());

    // Create a sequential pass and perform optimizations.
    transform::Pass seq = transform::Sequential(pass_seqs);
    if (targets.size() == 1) {
      for (const auto& kv : targets) {
        With<Target> tctx(kv.second);
        relay_module = seq(relay_module);
      }
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
  relay::Module RunDeviceAnnotationPass(const relay::Module& relay_module,
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
   * \brief Compile a Relay function to runtime module.
   *
   * \param func The Relay function.
   * \param params The parameters.
   */
  void BuildRelay(
      Function func,
      const std::unordered_map<std::string, tvm::runtime::NDArray>& params) {
    if (params.size()) {
      func = BindParamsByName(func, params);
    }

    // Perform Module->Module optimizations.
    relay::Module relay_module = relay::ModuleNode::FromExpr(func);
    relay_module = Optimize(relay_module, targets_, params);
    CHECK(relay_module.defined());
    // Get the updated function.
    func = relay_module->Lookup("main");

    // Generate code for the updated function.
    graph_codegen_ = std::unique_ptr<GraphCodegen>(new GraphCodegen());
    graph_codegen_->Init(nullptr, targets_);
    graph_codegen_->Codegen(func);

    ret_.graph_json = graph_codegen_->GetJSON();
    ret_.params = graph_codegen_->GetParams();

    auto lowered_funcs = graph_codegen_->GetLoweredFunc();
    if (lowered_funcs.size() != 0) {
      ret_.mod = tvm::build(
        lowered_funcs,
        target_host_,
        BuildConfig::Current());
    }
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
  std::shared_ptr<RelayBuildModule> exec = std::make_shared<RelayBuildModule>();
  return runtime::Module(exec);
}

TVM_REGISTER_GLOBAL("relay.build_module._BuildModule")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = RelayBuildCreate();
});

}  // namespace backend
}  // namespace relay
}  // namespace tvm
