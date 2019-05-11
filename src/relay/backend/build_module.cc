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
 *  Copyright (c) 2019 by Contributors
 * \file relay/backend/build_module.cc
 * \brief Code generation for TVM's graph runtime.
 */

#include <tvm/build_module.h>
#include <tvm/relay/op.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <vector>
#include <string>
#include <memory>

#include "utils.h"

namespace tvm {
namespace relay {
namespace backend {

/*!
 * \brief Context name / index
 *        See: python/tvm/_ffi/runtime_ctypes.py
 */
struct ContextMap {
  static const std::unordered_map<int, std::string> mask2str;
  static const std::unordered_map<std::string, int> str2mask;
  static std::string Mask2Str(int mask) {
    CHECK_GT(mask2str.count(mask), 0) << "Unknown mask.";
    return mask2str.at(mask);
  }
  static int Str2Mask(const std::string& str) {
    CHECK_GT(str2mask.count(str), 0) << "Unknown context.";
    return str2mask.at(str);
  }
};

const std::unordered_map<int, std::string> ContextMap::mask2str = {
  {1, "cpu"},
  {2, "gpu"},
  {4, "opencl"},
  {5, "aocl"},
  {6, "sdaccel"},
  {7, "vulkan"},
  {8, "metal"},
  {9, "vpi"},
  {10, "rocm"},
  {11, "opengl"},
  {12, "ext_dev"}
};

const std::unordered_map<std::string, int> ContextMap::str2mask = {
  {"llvm", 1},
  {"cpu", 1},
  {"c", 1},
  {"gpu", 2},
  {"cuda", 2},
  {"nvptx", 2},
  {"cl", 4},
  {"opencl", 4},
  {"aocl", 5},
  {"aocl_sw_emu", 5},
  {"vulkan", 7},
  {"metal", 8},
  {"vpi", 9},
  {"rocm", 10},
  {"opengl", 11},
  {"ext_dev", 12}
};

/*!
 * \brief A data structure to map the names of specific optimizations to
 *        numeric optimization levels
 *
 */
struct OptPassLevel {
  static const std::unordered_map<std::string, int> _data;
  /*!
   * \brief Get level for an optimization pass
   *
   * \param key pass name
   * \return int level
   */
  int operator[](const std::string& key) const {
    auto it = _data.find(key);
    if (it == _data.end()) {
      return -1;
    }
    return it->second;
  }
};

const std::unordered_map<std::string, int> OptPassLevel::_data = {
  {"SimplifyInference", 0},
  {"OpFusion", 1},
  {"FoldConstant", 2},
  {"CombineParallelConv2D", 3},
  {"FoldScaleAxis", 3},
  {"AlterOpLayout", 3},
  {"CanonicalizeOps", 3},
  {"EliminateCommonSubexpr", 3}
};

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
 * \brief Relay building config
 *
 */
struct RelayBuildConfig {
  int opt_level{2};
  std::string fallback_device{"llvm"};
  std::unordered_set<std::string> enabled_pass;
  std::unordered_set<std::string> disabled_pass;
  OptPassLevel OPT_PASS_LEVEL;
  inline bool pass_enabled(const std::string& pass_name) const {
    if (disabled_pass.count(pass_name)) {
      return false;
    }
    if (enabled_pass.count(pass_name)) {
      return true;
    }
    return opt_level >= OPT_PASS_LEVEL[pass_name];
  }
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

  void Init(runtime::Module* m,
            Map<HalideIR::Expr, HalideIR::Expr> targets) {
    Array<HalideIR::Expr> tgts;
    for (auto kv : targets) {
      tgts.push_back(kv.first);
      tgts.push_back(kv.second);
    }
    CallFunc("init", m, tgts);
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
    auto names = CallFunc<Array<HalideIR::Expr> >("list_params_name", nullptr);
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

template<typename R, typename ...Args>
R CallPackedFunc(const std::string &name, Args... args) {
  auto pf = GetPackedFunc(name);
  return (*pf)(std::forward<Args>(args)...);
}

template<typename ...Args>
Function CallPackedFunc(const std::string &name, Args... args) {
  auto pf = GetPackedFunc(name);
  return (*pf)(std::forward<Args>(args)...);
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
        Array<HalideIR::Expr> tmp = args[1];
        std::unordered_map<std::string, std::string> targets;
        for (size_t i = 0; i < tmp.size(); i += 2) {
          auto k = tmp[i].as<ir::StringImm>()->value;
          auto v = tmp[i + 1].as<ir::StringImm>()->value;
          targets[k] = v;
        }
        this->Build(args[0], targets, args[2]);
      });
    } else if (name == "list_params") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->ListParamNames();
      });
    } else if (name == "get_params") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->GetParams();
      });
    } else if (name == "set_opt_level") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.num_args, 1);
        int level = args[0];
        this->SetOptLevel(level);
      });
    } else if (name == "set_fallback_device") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        std::string dev = args[0];
        this->SetFallBackDev(dev);
      });
    } else if (name == "add_pass") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        std::string pass_name = args[0];
        this->AddPass(pass_name);
      });
    } else if (name == "disable_pass") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        std::string pass_name = args[0];
        this->DisablePass(pass_name);
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
   * \brief Add extra pass into build cfg
   *
   * \param pass_name name of pass
   */
  void AddPass(const std::string& pass_name) {
    cfg_.enabled_pass.insert(pass_name);
  }
  /*!
   * \brief Disable a specific pass in cfg
   *
   * \param pass_name name of pass
   */
  void DisablePass(const std::string& pass_name) {
    cfg_.disabled_pass.insert(pass_name);
  }
  /*!
   * \brief Set the Fallback device
   *
   * \param device name
   */
  void SetFallBackDev(const std::string& dev) {
    cfg_.fallback_device = dev;
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
  Array<HalideIR::Expr> ListParamNames() {
    Array<HalideIR::Expr> ret;
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
   * \brief Set the optimization level
   *
   * \param level
   */
  void SetOptLevel(char level) {
    cfg_.opt_level = level;
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
             const std::unordered_map<std::string, std::string>& targets,
             const std::string& target_host) {
    targets_ = targets;
    target_host_ = target_host;
    BuildRelay(func, cfg_, params_);
  }

 protected:
  /*!
   * \brief Bind params to function by using name
   * \param func Relay function
   * \param params params dict
   * \return relay::Function
   */
  relay::Function BindParamsByName(relay::Function func,
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
      auto e = CallPackedFunc<Expr>("relay._make.Constant", kv.second);
      bind_dict[arg] = e;
    }
    return CallPackedFunc("relay._expr.Bind", func, tvm::Map<relay::Var, Expr>(bind_dict));
  }

  /*!
   * \brief Optimize Relay function
   *
   * \param func Input function
   * \param target target device
   * \param cfg Relay build config
   * \param params params dict
   * \return relay::Function
   */
  relay::Function Optimize(relay::Function func,
                           const std::unordered_map<std::string, std::string>& targets,
                           const RelayBuildConfig& cfg,
                           const std::unordered_map<std::string, runtime::NDArray>& params) {
    if (params.size()) {
      func = BindParamsByName(func, params);
    }
    if (cfg.pass_enabled("SimplifyInference")) {
      func = CallPackedFunc("relay._ir_pass.infer_type", func, nullptr);
      func = CallPackedFunc("relay._ir_pass.simplify_inference", func);
    }
    if (cfg.pass_enabled("EliminateCommonSubexpr")) {
      auto fskip = PackedFunc([](TVMArgs args, TVMRetValue* rv) {
        Expr expr = args[0];
        if (expr.as<CallNode>()) {
          auto call_node = expr.as<CallNode>();
          auto op_node = call_node->op.as<OpNode>();
          if (op_node->name == "cast") {
            auto attrs = call_node->attrs.as<CastAttrs>();
            if (attrs->dtype == HalideIR::Int(32)) {
              *rv = true;
            }
          }
        }
        *rv =  false;
      });
      func = CallPackedFunc("relay._ir_pass.infer_type", func, nullptr);
      func = CallPackedFunc("relay._ir_pass.eliminate_common_subexpr", func, fskip);
    }
    if (cfg.pass_enabled("CombineParallelConv2D")) {
      const int min_num_branches = 3;
      func = CallPackedFunc("relay._ir_pass.infer_type", func, nullptr);
      func = CallPackedFunc("relay._ir_pass.CombineParallelConv2D", func, min_num_branches);
    }
    if (cfg.pass_enabled("FoldConstant")) {
      func = CallPackedFunc("relay._ir_pass.FoldConstant", func);
    }
    if (cfg.pass_enabled("FoldScaleAxis")) {
      func = CallPackedFunc("relay._ir_pass.infer_type", func, nullptr);
      func = CallPackedFunc("relay._ir_pass.backward_fold_scale_axis", func);
      func = CallPackedFunc("relay._ir_pass.infer_type", func, nullptr);
      func = CallPackedFunc("relay._ir_pass.forward_fold_scale_axis", func);
      func = CallPackedFunc("relay._ir_pass.FoldConstant", func);
    }
    if (cfg.pass_enabled("CanonicalizeOps")) {
      func = CallPackedFunc("relay._ir_pass.infer_type", func, nullptr);
      func = CallPackedFunc("relay._ir_pass.canonicalize_ops", func);
    }
    if (cfg.pass_enabled("AlterOpLayout")) {
      if (targets.size() == 1) {
        func = CallPackedFunc("relay._ir_pass.infer_type", func, nullptr);
        auto enter_pf = GetPackedFunc("_EnterTargetScope");
        auto exit_pf = GetPackedFunc("_ExitTargetScope");
        for (const auto& kv : targets) {
          auto target = Target::create(kv.second);
          (*enter_pf)(target);
          func = CallPackedFunc("relay._ir_pass.AlterOpLayout", func);
          (*exit_pf)();
        }
      } else {
        LOG(WARNING) << "AlterOpLayout pass is not enabled for heterogeneous"
                  << " execution yet.";
      }
    }
    if (cfg.pass_enabled("FoldConstant")) {
      func = CallPackedFunc("relay._ir_pass.FoldConstant", func);
    }
    return func;
  }
  /*!
   * \brief Update the target and fallback device required for heterogeneous
   * compilation. CPU is used as the fallback device if it wasn't provided.
   * Meanwhile, a CPU device type and "llvm" pair will be added to the target
   * dictionary in this case.
   *
   * \param targets dictionary
   * \param cfg
   * \return Map<HalideIR::Expr, HalideIR::Expr>
   */
  Map<HalideIR::Expr, HalideIR::Expr> UpdateHeterogeneousInputs(
    const std::unordered_map<std::string, std::string>& targets,
    const RelayBuildConfig& cfg) {
    Map<HalideIR::Expr, HalideIR::Expr> device_target;
    std::unordered_map<int64_t, std::string> tmp_map;
    auto fallback_idx = ContextMap::Str2Mask(cfg.fallback_device);

    for (const auto& kv : targets) {
      tmp_map[ContextMap::Str2Mask(kv.first)] = kv.second;
    }
    if (tmp_map.count(fallback_idx) == 0) {
      tmp_map[fallback_idx] = cfg.fallback_device;
    }
    for (const auto& kv : tmp_map) {
      device_target.Set(
        ir::IntImm::make(HalideIR::Int(64), kv.first),
        ir::StringImm::make(kv.second));
    }
    return device_target;
  }
  /*!
   * \brief Execute the device annotation passes to update the input program and
   *        target information.
   *
   * \param func
   * \param cfg
   * \param targets_map_ptr
   * \return Function
   */
  Function RunDeviceAnnotationPass(
      Function func,
      const RelayBuildConfig& cfg,
      Map<HalideIR::Expr, HalideIR::Expr>* targets_map_ptr) {
    auto fallback_idx = ContextMap::Str2Mask(cfg.fallback_device);
    func = CallPackedFunc("relay._ir_pass.infer_type", func, nullptr);
    func = CallPackedFunc("relay._ir_pass.RewriteDeviceAnnotation", func, fallback_idx);
    auto device_map = CallPackedFunc<Map<Expr, Integer> >("relay._ir_pass.CollectDeviceInfo",
                                                       func,
                                                       nullptr);
    if (device_map.size() == 0) {
      auto annotation_map =
        CallPackedFunc<Map<Expr, Integer> >("relay._ir_pass.CollectDeviceAnnotationOps",
                                            func,
                                            nullptr);
      if (annotation_map.size() == 0) {
        targets_map_ptr->Set(
          ir::IntImm::make(HalideIR::Int(64), 0),
          ir::StringImm::make(cfg.fallback_device));
      } else {
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
        targets_map_ptr->Set(
          ir::IntImm::make(HalideIR::Int(64), 0),
          ir::StringImm::make(ContextMap::Mask2Str(dev_type)));
      }
    }
    return func;
  }

  /*!
   * \brief Build relay function to runtime module
   *
   * \param func Relay Function
   * \param cfg Relay build config
   * \param params parameters
   */
  void BuildRelay(Function func,
                  const RelayBuildConfig& cfg,
                  const std::unordered_map<std::string, tvm::runtime::NDArray> &params) {
    // convert
    tvm_cfg_ = build_config();
    Map<HalideIR::Expr, HalideIR::Expr> device_target;
    if (targets_.size() > 1) {
      device_target = UpdateHeterogeneousInputs(targets_, cfg);
    } else {
      for (auto &kv : targets_) {
        device_target.Set(
          ir::IntImm::make(HalideIR::Int(64), ContextMap::Str2Mask(kv.first)),
          ir::StringImm::make(kv.second));
      }
    }
    func = Optimize(func, targets_, cfg, params);
    if (device_target.size() > 1) {
      func = RunDeviceAnnotationPass(func, cfg, &device_target);
    }
    // TODO(@jroesch): use the passes directly.
    func = CallPackedFunc("relay._ir_pass.infer_type", func, nullptr);
    func = CallPackedFunc("relay._ir_pass.FuseOps", func, cfg.opt_level, nullptr);
    func = CallPackedFunc("relay._ir_pass.infer_type", func, nullptr);

    graph_codegen_ = std::unique_ptr<GraphCodegen>(new GraphCodegen());
    graph_codegen_->Init(nullptr, device_target);
    graph_codegen_->Codegen(func);

    ret_.graph_json = graph_codegen_->GetJSON();
    ret_.params = graph_codegen_->GetParams();

    auto target_host = Target::create(target_host_);
    ret_.mod = tvm::build(graph_codegen_->GetLoweredFunc(), target_host, tvm_cfg_);
  }

 protected:
  std::unique_ptr<GraphCodegen> graph_codegen_;
  /*! \brief target device */
  std::unordered_map<std::string, std::string> targets_;
  /*! \brief target host device */
  std::string target_host_;
  /*! \brief frontend optimization configure */
  RelayBuildConfig cfg_;
  /*! \brief parameters */
  std::unordered_map<std::string, runtime::NDArray> params_;
  /*! \brief building output */
  BuildOutput ret_;
  /*! \brief tvm building cfg */
  BuildConfig tvm_cfg_;
};

runtime::Module RelayBuildCreate() {
  std::shared_ptr<RelayBuildModule> exec = std::make_shared<RelayBuildModule>();
  return runtime::Module(exec);
}

TVM_REGISTER_GLOBAL("relay.build_module._BuildModule").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = RelayBuildCreate();
});

}  // namespace backend
}  // namespace relay
}  // namespace tvm
