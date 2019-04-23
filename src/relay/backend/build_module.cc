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
 *  Copyright (c) 2018 by Contributors
 * \file relay/backend/build_module.cc
 * \brief Graph runtime codegen
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
  static std::unordered_map<int, std::string> _declare_mask2str() {
    std::unordered_map<int, std::string> ret;
    ret[1] = "cpu";
    ret[2] = "gpu";
    ret[4] = "opencl";
    ret[5] = "aocl";
    ret[6] = "sdaccel";
    ret[7] = "valkan";
    ret[8] = "metal";
    ret[9] = "vpi";
    ret[10] = "rocm";
    ret[11] = "opengl";
    ret[12] = "ext_dev";
    return ret;
  }
  static std::unordered_map<std::string, int> _declare_str2mask() {
    std::unordered_map<std::string, int> ret;
    ret["llvm"] = 1;
    ret["stackvm"] = 1;
    ret["cpu"] = 1;
    ret["c"] = 1;
    ret["gpu"] = 2;
    ret["cuda"] = 2;
    ret["nvptx"] = 2;
    ret["cl"] = 4;
    ret["opencl"] = 4;
    ret["aocl"] = 5;
    ret["aocl_sw_emu"] = 5;
    ret["sdaccel"] = 6;
    ret["vulkan"] = 7;
    ret["metal"] = 8;
    ret["vpi"] = 9;
    ret["rocm"] = 10;
    ret["opengl"] = 11;
    ret["ext_dev"] = 12;
    return ret;
  }
};

const std::unordered_map<int, std::string> ContextMap::mask2str =
  ContextMap::_declare_mask2str();
const std::unordered_map<std::string, int> ContextMap::str2mask =
  ContextMap::_declare_str2mask();

/*! \brief Optimization pass level */
struct OptPassLevel {
  static const std::unordered_map<std::string, int> _data;
  static std::unordered_map<std::string, int> _declare_opt_level() {
    std::unordered_map<std::string, int> ret;
    ret["SimplifyInference"] = 0;
    ret["OpFusion"] = 1;
    ret["FoldConstant"] = 2;
    ret["CombineParallelConv2D"] = 3;
    ret["FoldScaleAxis"] = 3;
    ret["AlterOpLayout"] = 3;
    ret["CanonicalizeOps"] = 3;
    ret["EliminateCommonSubexpr"] = 3;
    return ret;
  }
  int operator[](const std::string& key) const {
    auto it = _data.find(key);
    if (it == _data.end()) {
      return -1;
    }
    return it->second;
  }
};

const std::unordered_map<std::string, int> OptPassLevel::_data =
  OptPassLevel::_declare_opt_level();

/*! \brief Output of function building */
struct BuildOutput {
  std::string graph_json;
  runtime::Module mod;
  std::unordered_map<std::string, tvm::runtime::NDArray> params;
};

/*! \brief Relay Building configuration */
struct RelayBuildConfig {
  int opt_level{2};
  std::string fall_back_device{"llvm"};
  std::unordered_set<std::string> add_pass;
  std::unordered_set<std::string> disabled_pass;
  OptPassLevel  OPT_PASS_LEVEL;
  inline bool pass_enabled(std::string pass_name) const {
    if (disabled_pass.count(pass_name)) {
      return false;
    }
    if (add_pass.count(pass_name)) {
      return true;
    }
    return opt_level >= OPT_PASS_LEVEL[pass_name];
  }
};

/*! \brief GraphCodegen wrapper */
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
    _CallFunc("init", m, tgts);
  }

  void Codegen(Function func) {
    _CallFunc("codegen", func);
  }

  std::string GetJSON() {
    return _CallFunc<std::string>("get_graph_json", nullptr);
  }

  Map<std::string, Array<LoweredFunc> > GetLoweredFunc() {
    return _CallFunc<Map<std::string, Array<LoweredFunc> > >("get_lowered_funcs", nullptr);
  }

  std::unordered_map<std::string, tvm::runtime::NDArray> GetParams() {
    std::unordered_map<std::string, tvm::runtime::NDArray> ret;
    auto names = _CallFunc<Array<HalideIR::Expr> >("list_params_name", nullptr);
    for (auto expr : names) {
      auto key = expr.as<ir::StringImm>()->value;
      ret[key] = _CallFunc<runtime::NDArray>("get_param_by_name", key);
    }
    return ret;
  }

 protected:
  tvm::runtime::Module mod;
  template<typename R, typename ...Args>
  R _CallFunc(const std::string &name, Args... args) {
    auto pf = mod.GetFunction(name, false);
    return pf(std::forward<Args>(args)...);
  }
  template<typename ...Args>
  void _CallFunc(const std::string &name, Args... args) {
    auto pf = mod.GetFunction(name, false);
    pf(std::forward<Args>(args)...);
    return;
  }
};

template<typename R, typename ...Args>
R _CallPacked(const std::string &name, Args... args) {
  auto pf = GetPackedFunc(name);
  return (*pf)(std::forward<Args>(args)...);
}

template<typename ...Args>
Function _CallPacked(const std::string &name, Args... args) {
  auto pf = GetPackedFunc(name);
  return (*pf)(std::forward<Args>(args)...);
}


class RelayBuildModule : public runtime::ModuleNode {
 public:
  /*!
   * \brief Get member function to front-end
   * \param name The name of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The corresponding member function.
   */
  virtual PackedFunc GetFunction(const std::string& name,
                                 const std::shared_ptr<ModuleNode>& sptr_to_self) {
    if (name == "get_graph_json") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->_GetGraphJSON();
      });
    } else if (name == "get_module") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->_GetModule();
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
        this->_Build(args[0], targets, args[2]);
      });
    } else if (name == "list_params") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->_ListParamNames();
      });
    } else if (name == "get_param_by_name") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.num_args, 1);
        *rv = this->_GetParam(args[0]);
      });
    } else if (name == "set_opt_level") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.num_args, 1);
        int level = args[0];
        this->_SetOptLevel(level);
      });
    } else if (name == "set_fallback_device") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        std::string dev = args[0];
        this->_SetFallBackDev(dev);
      });
    } else if (name == "add_pass") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        std::string pass_name = args[0];
        this->_AddPass(pass_name);
      });
    } else if (name == "disable_pass") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        std::string pass_name = args[0];
        this->_DisablePass(pass_name);
      });
    } else {
      return PackedFunc([sptr_to_self](TVMArgs args, TVMRetValue* rv) {});
    }
  }

  /*!
   * \brief Get the GraphJSON for runtime
   *
   * \return const std::string graph_json
   */
  const std::string _GetGraphJSON() {
    return ret_.graph_json;
  }
  /*!
   * \brief Add extra pass during build
   * 
   * \param pass_name 
   */
  void _AddPass(const std::string& pass_name) {
    cfg_.add_pass.insert(pass_name);
  }

  void _DisablePass(const std::string& pass_name) {
    cfg_.disabled_pass.insert(pass_name);
  }

  void _SetFallBackDev(const std::string& dev) {
    cfg_.fall_back_device = dev;
  }
  /*!
   * \brief Get the Module object
   *
   * \return runtime::Module
   */
  runtime::Module _GetModule() {
    return ret_.mod;
  }

  /*!
   * \brief List all paramter names
   * 
   * \return Array<StringImm> 
   */
  Array<HalideIR::Expr> _ListParamNames() {
    Array<HalideIR::Expr> ret;
    for (const auto& kv : params_) {
      ret.push_back(ir::StringImm::make(kv.first));
    }
    return ret;
  }

  /*!
   * \brief Get the Param of name
   * 
   * \param name 
   * \return runtime::NDArray 
   */
  runtime::NDArray _GetParam(const std::string& name) {
    CHECK_GT(params_.count(name), 0) << "Can not find param with name: " << name;
    return params_[name];
  }

  /*!
   * \brief Set the parameters
   *
   * \param name name of parameter
   * \param data_in input DLTensor
   */
  void _SetParams(const std::string& name, DLTensor* data_in) {
    if (!params_.count(name)) {
      std::vector<int64_t> shape(data_in->shape, data_in->shape + data_in->ndim);
      params_[name] = tvm::runtime::NDArray::Empty(shape, data_in->dtype, {kDLCPU, 0});
    }
    params_[name].CopyFrom(data_in);
  }

  /*!
   * \brief Set the optimization level
   *
   * \param level
   */
  void _SetOptLevel(char level) {
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
  void _Build(Function func,
             const std::unordered_map<std::string, std::string>& targets,
             const std::string& target_host) {
    targets_ = targets;
    target_host_ = target_host;
    _BuildRelay(func, cfg_, params_);
  }

 protected:
  /*!
   * \brief bind params to function
   *
   * \param func Relay function
   * \param params params dict
   * \return relay::Function
   */
  relay::Function _bind_params_by_name(relay::Function func,
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
      auto e = _CallPacked<Expr>("relay._make.Constant", kv.second);
      bind_dict[arg] = e;
    }
    return _CallPacked("relay._expr.Bind", func, tvm::Map<relay::Var, Expr>(bind_dict));
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
  relay::Function _Optimize(relay::Function func,
                            const std::unordered_map<std::string, std::string>& targets,
                            const RelayBuildConfig& cfg,
                            const std::unordered_map<std::string, runtime::NDArray>& params) {
    if (params.size()) {
      func = _bind_params_by_name(func, params);
    }
    if (cfg.pass_enabled("SimplifyInference")) {
      func = _CallPacked("relay._ir_pass.infer_type", func, nullptr);
      func = _CallPacked("relay._ir_pass.simplify_inference", func);
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
              return true;
            }
          }
        }
        return false;
      });
      func = _CallPacked("relay._ir_pass.infer_type", func, nullptr);
      func = _CallPacked("relay._ir_pass.eliminate_common_subexpr", func, fskip);
    }
    if (cfg.pass_enabled("CombineParallelConv2D")) {
      func = _CallPacked("relay._ir_pass.infer_type", func, nullptr);
      func = _CallPacked("relay._ir_pass.CombineParallelConv2D", func);
    }
    if (cfg.pass_enabled("FoldConstant")) {
      func = _CallPacked("relay._ir_pass.FoldConstant", func);
    }
    if (cfg.pass_enabled("FoldScaleAxis")) {
      func = _CallPacked("relay._ir_pass.infer_type", func, nullptr);
      func = _CallPacked("relay._ir_pass.backward_fold_scale_axis", func);
      func = _CallPacked("relay._ir_pass.infer_type", func, nullptr);
      func = _CallPacked("relay._ir_pass.forward_fold_scale_axis", func);
      func = _CallPacked("relay._ir_pass.FoldConstant", func);
    }
    if (cfg.pass_enabled("CanonicalizeOps")) {
      func = _CallPacked("relay._ir_pass.infer_type", func, nullptr);
      func = _CallPacked("relay._ir_pass.canonicalize_ops", func);
    }
    if (cfg.pass_enabled("AlterOpLayout")) {
      if (targets.size() == 1) {
        func = _CallPacked("relay._ir_pass.infer_type", func, nullptr);
        func = _CallPacked("relay._ir_pass.AlterOpLayout", func);
      } else {
        LOG(WARNING) << "AlterOpLayout pass is not enabled for heterogeneous"
                  << " execution yet.";
      }
    }
    if (cfg.pass_enabled("FoldConstant")) {
      func = _CallPacked("relay._ir_pass.FoldConstant", func);
    }
    return func;
  }

  Map<HalideIR::Expr, HalideIR::Expr> _UpdateHeterogeneousInputs(
    const std::unordered_map<std::string, std::string>& targets,
    const RelayBuildConfig& cfg) {
    Map<HalideIR::Expr, HalideIR::Expr> device_target;
    std::unordered_map<int64_t, std::string> tmp_map;
    auto fallback_idx = ContextMap::Str2Mask(cfg.fall_back_device);

    for (const auto& kv : targets) {
      tmp_map[ContextMap::Str2Mask(kv.first)] = kv.second;
    }
    if (tmp_map.count(fallback_idx) == 0) {
      tmp_map[fallback_idx] = cfg.fall_back_device;
    }
    for (const auto& kv : tmp_map) {
      device_target.Set(
        ir::IntImm::make(HalideIR::Int(64), kv.first),
        ir::StringImm::make(kv.second));
    }
    return device_target;
  }

  Function _RunDeviceAnnotationPass(
    Function func,
    const RelayBuildConfig& cfg,
    Map<HalideIR::Expr, HalideIR::Expr>* targets_map_ptr) {
    auto fallback_idx = ContextMap::Str2Mask(cfg.fall_back_device);
    func = _CallPacked("relay._ir_pass.infer_type", func, nullptr);
    func = _CallPacked("relay._ir_pass.RewriteDeviceAnnotation", func, fallback_idx);
    auto device_map = _CallPacked<Map<Expr, Integer> >("relay._ir_pass.CollectDeviceInfo",
                                                       func,
                                                       nullptr);
    if (device_map.size() == 0) {
      auto annotation_map = _CallPacked<Map<Expr, Integer> >("_ir_pass.CollectDeviceAnnotationOps",
                                                             func,
                                                             nullptr);
      if (annotation_map.size() == 0) {
        targets_map_ptr->Set(
          ir::IntImm::make(HalideIR::Int(64), 0),
          ir::StringImm::make(cfg.fall_back_device));
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
   * \brief Build module given lowered functions
   * 
   * \param lowered_funcs 
   * \param targets 
   * \param cfg 
   */
  void _BuildModule(Map<std::string, Array<LoweredFunc> > lowered_funcs,
                    Map<HalideIR::Expr, HalideIR::Expr> targets,
                    const BuildConfig& cfg) {
    auto target_host = Target::create(cfg_.fall_back_device);
    for (const auto& kv : lowered_funcs) {
      std::unordered_set<std::string> fname_set;
      for (auto f : kv.second) {
        if (fname_set.count(f->name)) {
          LOG(FATAL) << "Duplicate function name "
                     << f->name;
        }
        fname_set.insert(f->name);
      }
    }
    std::unordered_map<std::string, Target> target_map;
    for (const auto& kv : lowered_funcs) {
      target_map[kv.first] = Target::create(kv.first);
    }
    Array<LoweredFunc> fhost_all;
    std::vector<runtime::Module> device_module;
    for (const auto& kv : lowered_funcs) {
      auto target = target_map[kv.first];
      auto mdev = build(kv.second,
                        target,
                        target_host,
                        cfg,
                        &fhost_all,
                        &device_module);
    }

    auto mhost = build(fhost_all,
                       target_host,
                       target_host,
                       cfg);

    for (auto mdev : device_module) {
      mhost.Import(mdev);
    }
    ret_.mod = mhost;
  }

  /*!
   * \brief Build relay function to runtime module
   *
   * \param func Relay Function
   * \param target target device
   * \param target_host host device
   * \param cfg Relay build config
   * \param params params
   * \return BuildOutput
   */
  void _BuildRelay(relay::Function func,
                   const RelayBuildConfig& cfg,
                   const std::unordered_map<std::string, tvm::runtime::NDArray> &params) {
    // convert
    tvm_cfg_ = build_config();
    auto device_target = _UpdateHeterogeneousInputs(targets_, cfg);
    func = _Optimize(func, targets_, cfg, params);
    if (targets_.size() > 1) {
      func = _RunDeviceAnnotationPass(func, cfg, &device_target);
    }
    func = _CallPacked("relay._ir_pass.infer_type", func, nullptr);
    func = _CallPacked("relay._ir_pass.FuseOps", func, cfg.opt_level);
    func = _CallPacked("relay._ir_pass.infer_type", func, nullptr);

    graph_codegen_ = std::unique_ptr<GraphCodegen>(new GraphCodegen());
    graph_codegen_->Init(nullptr, device_target);
    graph_codegen_->Codegen(func);

    ret_.graph_json = graph_codegen_->GetJSON();
    ret_.params = graph_codegen_->GetParams();

    _BuildModule(graph_codegen_->GetLoweredFunc(),
                 device_target,
                 tvm_cfg_);
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
