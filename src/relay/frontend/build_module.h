/*!
 *  Copyright (c) 2018 by Contributors
 * \file relay/backend/compile_engine.h
 * \brief Internal compilation engine handle function cache.
 *  and interface to low level code generation.
 */
#ifndef TVM_RELAY_FRONTEND_BUILD_MODULE_H_
#define TVM_RELAY_FRONTEND_BUILD_MODULE_H_
#include <vector>
#include <string>

#include "utils.h"
#include "graph_runtime_codegen.h"
#include "tvm_build_module.h"

namespace tvm {
namespace relay {
namespace frontend {

/*! \brief Optimization pass level */
struct OPT_PASS_LEVEL_DECL {
  const std::unordered_map<std::string, char> _data {
    {"SimplifyInference", 0},
    {"OpFusion", 1},
    {"FoldConstant", 2},
    {"CombineParallelConv2D", 3},
    {"FoldScaleAxis", 3},
    {"AlterOpLayout", 3}
  };
  char operator[](const std::string& key) const {
    auto it = _data.find(key);
    if (it == _data.end()) {
      return -1;
    }
    return it->second;
  }
};

/*! \brief Output of function building */
struct BuildOutput {
  std::string graph_json;
  runtime::Module mod;
  std::unordered_map<std::string, tvm::runtime::NDArray> params;
};

/*! \brief Relay Building configuration */
struct RelayBuildConfig {
  char opt_level{2};
  std::string fall_back_device;
  std::unordered_set<std::string> add_pass;
  OPT_PASS_LEVEL_DECL  OPT_PASS_LEVEL;
  inline bool pass_enabled(std::string pass_name) {
    if (add_pass.count(pass_name)) {
      return true;
    }
    return opt_level >= OPT_PASS_LEVEL[pass_name];
  }
};


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
    } else if (name == "get_param") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.num_args, 1);
        *rv = this->GetParam(args[0]);
      });
    } else if (name == "set_opt_level") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.num_args, 1);
        int level = args[0];
        this->SetOptLevel(level);
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
  const std::string GetGraphJSON() {
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
   * \return Array<std::string> 
   */
  Map<std::string, Integer> ListParamNames() {
    Map<std::string, Integer> ret;
    for (const auto& kv : params_) {
      ret.Set(kv.first, 1);
    }
    return ret;
  }

  /*!
   * \brief Get the Param of name
   * 
   * \param name 
   * \return runtime::NDArray 
   */
  runtime::NDArray GetParam(const std::string& name) {
    CHECK_GT(params_.count(name), 0) << "Can not find param with name: " << name;
    return params_[name];
  }

  /*!
   * \brief Set the parameters
   *
   * \param name name of parameter
   * \param data_in input DLTensor
   */
  void SetParams(const std::string& name, DLTensor* data_in) {
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
             const std::string& target,
             const std::string& target_host) {
    target_ = tvm::Target::create(target);
    target_host_ = tvm::Target::create(target_host);
    ret_ = _build_relay(func, target_, target_host_, cfg_, params_);
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
      auto pf_make_constant = GetPakcedFunc("relay._make.Constant");
      Expr e = (*pf_make_constant)(kv.second);
      bind_dict[arg] = e;
    }
    const PackedFunc* pf_bind = GetPakcedFunc("relay._expr.Bind");
    return (*pf_bind)(func, tvm::Map<relay::Var, Expr>(bind_dict));
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
  relay::Function _optimize(relay::Function func,
                           tvm::Target target,
                           RelayBuildConfig cfg,
                           const std::unordered_map<std::string, runtime::NDArray>& params) {
    if (params.size()) {
      func = _bind_params_by_name(func, params);
    }
    if (cfg.pass_enabled("SimplifyInference")) {
      auto pf0 = GetPakcedFunc("relay._ir_pass.infer_type");
      func = (*pf0)(func, nullptr);
      auto pf1 = GetPakcedFunc("relay._ir_pass.simplify_inference");
      func = (*pf1)(func);
    }
    if (cfg.pass_enabled("CombineParallelConv2D")) {
      auto pf0 = GetPakcedFunc("relay._ir_pass.infer_type");
      func = (*pf0)(func, nullptr);
      auto pf1 = GetPakcedFunc("relay._ir_pass.CombineParallelConv2D");
      func = (*pf1)(func);
    }
    if (cfg.pass_enabled("FoldConstant")) {
      auto pf0 = GetPakcedFunc("relay._ir_pass.FoldConstant");
      func = (*pf0)(func);
    }
    if (cfg.pass_enabled("FoldScaleAxis")) {
      auto pf0 = GetPakcedFunc("relay._ir_pass.infer_type");
      func = (*pf0)(func, nullptr);
      auto pf1 = GetPakcedFunc("relay._ir_pass.backward_fold_scale_axis");
      func = (*pf1)(func);
      func = (*pf0)(func, nullptr);
      auto pf2 = GetPakcedFunc("relay._ir_pass.forward_fold_scale_axis");
      func = (*pf2)(func);
      auto pf3 = GetPakcedFunc("relay._ir_pass.FoldConstant");
      func = (*pf3)(func);
    }
    if (cfg.pass_enabled("AlterOpLayout")) {
      auto pf0 = GetPakcedFunc("relay._ir_pass.infer_type");
      func = (*pf0)(func, nullptr);
      auto pf1 = GetPakcedFunc("relay._ir_pass.canonicalize_ops");
      func = (*pf1)(func);
      func = (*pf0)(func);
      auto pf2 = GetPakcedFunc("relay._ir_pass.AlterOpLayout");
      func = (*pf2)(func);
    }
    if (cfg.pass_enabled("FoldConstant")) {
      auto pf0 = GetPakcedFunc("relay._ir_pass.FoldConstant");
      func = (*pf0)(func);
    }
    return func;
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
  BuildOutput _build_relay(relay::Function func,
                           tvm::Target target,
                           tvm::Target target_host,
                           RelayBuildConfig cfg,
                  const std::unordered_map<std::string, tvm::runtime::NDArray> &params) {
    func = _optimize(func, target, cfg, params);
    // TODO(xxx): support heterogeneous execution
    auto pf0 = GetPakcedFunc("relay._ir_pass.infer_type");
    func = (*pf0)(func, nullptr);

    // fuse op
    auto pf1 = GetPakcedFunc("relay._ir_pass.FuseOps");
    func = (*pf1)(func, cfg.opt_level);

    // Graph code
    func = (*pf0)(func, nullptr);
    auto graph_gen = GraphRuntimeCodegen(nullptr, target);
    LoweredOutput result = graph_gen.Codegen(func);
    BuildOutput ret;
    auto config = build_config();
    ret.graph_json = result.graph_json;
    ret.params = result.params;

    ret.mod = tvm_build(result.lowered_funcs[target->str()],
                      target,
                      target_host,
                      config);

    return ret;
  }

 protected:
  /*! \brief Relay function to be built */
  relay::Function func_;
  /*! \brief target device */
  tvm::Target target_;
  /*! \brief target host device */
  tvm::Target target_host_;
  /*! \brief frontend optimization configure */
  RelayBuildConfig cfg_;
  /*! \brief parameters */
  std::unordered_map<std::string, runtime::NDArray> params_;
  /*! \brief output for graph runtime */
  BuildOutput ret_;
};

runtime::Module RelayBuildCreate() {
  std::shared_ptr<RelayBuildModule> exec = std::make_shared<RelayBuildModule>();
  return runtime::Module(exec);
}

}  // namespace frontend
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_FRONTEND_BUILD_MODULE_H_
