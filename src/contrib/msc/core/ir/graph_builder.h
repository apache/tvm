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
 * \file src/contrib/msc/core/ir/graph_builder.h
 * \brief Builder of MSCGraph.
 */
#ifndef TVM_CONTRIB_MSC_CORE_IR_GRAPH_BUILDER_H_
#define TVM_CONTRIB_MSC_CORE_IR_GRAPH_BUILDER_H_

#include <dmlc/json.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/tir/data_layout.h>

#include <set>
#include <stack>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../utils.h"
#include "graph.h"
#include "plugin.h"

namespace tvm {
namespace contrib {
namespace msc {

using Expr = tvm::RelayExpr;
using RelaxExprVisitor = tvm::relax::ExprVisitor;
using RelayExprVisitor = tvm::relay::ExprVisitor;

using tvm::runtime::NDArray;

/*!
 * \brief Config for building MSCGraph.
 *  Define the configuration for building MSCGraph
 */
struct MSCRBuildConfig {
  bool prune_graph{false};
  bool use_var_name{false};
  int float_precision = 6;
  std::string byoc_entry;
  std::string sort_by;
  std::string target = "";
  std::string graph_name = "";
  std::vector<std::string> input_aliases;
  std::vector<std::string> output_aliases;
  std::unordered_map<std::string, std::vector<std::string>> input_types;

  void LoadInputTypes(dmlc::JSONReader* reader) {
    std::string key;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      std::vector<std::string> types;
      reader->Read(&types);
      input_types[key] = types;
    }
  }

  void Load(dmlc::JSONReader* reader) {
    std::string key;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      if (key == "prune_graph") {
        reader->Read(&prune_graph);
      } else if (key == "use_var_name") {
        reader->Read(&use_var_name);
      } else if (key == "float_precision") {
        reader->Read(&float_precision);
      } else if (key == "byoc_entry") {
        reader->Read(&byoc_entry);
      } else if (key == "sort_by") {
        reader->Read(&sort_by);
      } else if (key == "target") {
        reader->Read(&target);
      } else if (key == "graph_name") {
        reader->Read(&graph_name);
      } else if (key == "input_aliases") {
        reader->Read(&input_aliases);
      } else if (key == "output_aliases") {
        reader->Read(&output_aliases);
      } else if (key == "input_types") {
        this->LoadInputTypes(reader);
      }
    }
  }
};

class AttrGetter : public AttrVisitor {
 public:
  /*!
   * \brief Get the attributes as Map<String, String>
   * \param attrs the attributes.
   */
  explicit AttrGetter(Map<String, String>* attrs) : attrs_(attrs) {}

  void Visit(const char* key, double* value) final { attrs_->Set(key, std::to_string(*value)); }

  void Visit(const char* key, int64_t* value) final { attrs_->Set(key, std::to_string(*value)); }

  void Visit(const char* key, uint64_t* value) final { attrs_->Set(key, std::to_string(*value)); }

  void Visit(const char* key, int* value) final { attrs_->Set(key, std::to_string(*value)); }

  void Visit(const char* key, bool* value) final { attrs_->Set(key, std::to_string(*value)); }

  void Visit(const char* key, std::string* value) final { attrs_->Set(key, *value); }

  void Visit(const char* key, DataType* value) final {
    attrs_->Set(key, runtime::DLDataType2String(*value));
  }

  void Visit(const char* key, runtime::ObjectRef* value) final {
    attrs_->Set(key, StringUtils::ToString(*value));
  }

  void Visit(const char* key, void** value) final {
    LOG(FATAL) << "TypeError: void is not allowed in Attrs";
  }

  void Visit(const char* key, runtime::NDArray* value) final {
    LOG(FATAL) << "TypeError: NDArray is not allowed in Attrs";
  }

 private:
  Map<String, String>* attrs_;
};

class RelaxFuncAttrGetter : public RelaxExprVisitor {
 public:
  /*! \brief Get the attributes as Map<String, String>*/
  Map<String, String> GetAttrs(const Expr& expr) {
    VisitExpr(expr);
    return attrs_;
  }

  void VisitExpr_(const relax::CallNode* op) final;

  void VisitExpr_(const relax::TupleGetItemNode* op) final;

 private:
  Map<String, String> attrs_;
};

class RelaxFuncValueGetter : public RelaxExprVisitor {
 public:
  /*! \brief Get the attributes from prim value as Map<String, String>*/
  Array<String> GetValues(const Expr& expr) {
    VisitExpr(expr);
    return values_;
  }

  void VisitExpr_(const relax::CallNode* op) final;

 private:
  Array<String> values_;
};

class RelaxFuncParamsFinder : public RelaxExprVisitor {
 public:
  /*!
   * \brief The constructor of RelaxFuncParamsFinder
   * \param ref_module the reference module.
   */
  explicit RelaxFuncParamsFinder(const IRModule& ref_module) : RelaxExprVisitor() {
    ref_module_ = ref_module;
  }

  /*! \brief Find the func params and bind with arguments*/
  Map<Expr, Expr> FindParams(const Expr& expr) {
    VisitExpr(expr);
    return params_;
  }

  void VisitBinding_(const relax::VarBindingNode* binding, const relax::FunctionNode* val) final;

  void VisitExpr_(const relax::CallNode* op) final;

 private:
  IRModule ref_module_;
  Map<Expr, Expr> params_;
  Map<Expr, relax::Function> local_funcs_;
};

class RelaxLayoutsFinder : public RelaxExprVisitor {
 public:
  /*!
   * \brief The constructor of RelaxLayoutsFinder
   * \param ref_module the reference module.
   */
  explicit RelaxLayoutsFinder(const IRModule& ref_module) : RelaxExprVisitor() {
    ref_module_ = ref_module;
  }

  /*! \brief Find the layouts form attrs*/
  Map<String, String> FindLayouts(const Expr& expr) {
    VisitExpr(expr);
    return layouts_;
  }

  void VisitBinding_(const relax::VarBindingNode* binding, const relax::FunctionNode* val) final;

  void VisitExpr_(const relax::CallNode* op) final;

 private:
  IRModule ref_module_;
  Map<String, String> layouts_;
  Map<Expr, relax::Function> local_funcs_;
};

class RelaxGraphBuilder : public RelaxExprVisitor {
 public:
  /*!
   * \brief The constructor of RelaxGraphBuilder
   * \param ref_module the reference module.
   * \param name the name of the graph.
   * \param options the options of build the graph.
   */
  explicit RelaxGraphBuilder(const IRModule& ref_module, const String& name,
                             const std::string& options = "")
      : RelaxExprVisitor() {
    ref_module_ = ref_module;
    if (options.size() > 0) {
      std::istringstream is(options);
      dmlc::JSONReader reader(&is);
      reader.Read(&config_);
    }
    name_ = config_.graph_name.size() > 0 ? String(config_.graph_name) : name;
    if (config_.byoc_entry.size() > 0) {
      func_params_ = RelaxFuncParamsFinder(ref_module).FindParams(ref_module->Lookup(name));
    }
    layouts_ = RelaxLayoutsFinder(ref_module).FindLayouts(ref_module->Lookup(name));
  }

  /*! \brief Build MSCGraph from relax function*/
  const MSCGraph Build(const relax::Function& func);

  /*! \brief Get the config of builder */
  const MSCRBuildConfig config() { return config_; }

  /*! \brief Create and add MSCJoint from expr*/
  const MSCJoint AddNode(const Expr& expr, const Optional<Expr>& binding_var = NullOpt,
                         const String& name = "");

  /*! \brief Create and add MSCPrim from prim*/
  const MSCPrim AddPrim(const PrimExpr& prim);

  const MSCPrim MatchOrCreatePrim(const PrimExpr& prim, const String& op = "",
                                  const Array<BaseJoint>& parents = Array<BaseJoint>(),
                                  const Map<String, String>& attrs = Map<String, String>());

  void VisitBindingBlock(const relax::BindingBlock& block) final;

  void VisitExpr_(const relax::ConstantNode* op) final;

  void VisitBinding_(const relax::VarBindingNode* binding, const relax::ConstantNode* val) final;

  void VisitBinding_(const relax::VarBindingNode* binding, const relax::ShapeExprNode* val) final;

  void VisitBinding_(const relax::VarBindingNode* binding, const relax::CallNode* call_node) final;

  void VisitBinding_(const relax::VarBindingNode* binding, const relax::TupleNode* val) final;

  void VisitBinding_(const relax::VarBindingNode* binding,
                     const relax::TupleGetItemNode* val) final;

  void VisitBinding_(const relax::VarBindingNode* binding, const relax::VarNode* val) final;

  void VisitBinding_(const relax::VarBindingNode* binding, const relax::DataflowVarNode* val) final;

  void VisitBinding_(const relax::VarBindingNode* binding, const relax::FunctionNode* val) final;

  void VisitPrimExpr(const PrimExpr& prim) final;

 private:
  /*! \brief Get the node_name, optype, layout for func*/
  const std::tuple<String, String, String> ParseFunc(const relax::Function& func);

  /*! \brief Get the plugin inputs*/
  Array<Expr> GetPluginInputs(const relax::Expr& expr);

  String name_;
  IRModule ref_module_;
  MSCRBuildConfig config_;
  Map<String, String> layouts_;
  Array<MSCJoint> nodes_;
  Map<String, MSCTensor> weights_;
  Map<Expr, Array<String>> expr_tensor_map_;
  std::unordered_map<String, std::pair<BaseJoint, size_t>> tensor_input_map_;
  std::set<String> ignore_nodes_;
  // scope name
  String scope_name_;
  std::set<String> setted_blocks_;
  Array<String> block_stack_;
  // BYOC maps
  Map<Expr, relax::Function> target_funcs_;
  Map<Expr, Expr> func_params_;
  // prims
  Array<MSCPrim> prims_;
  Map<PrimExpr, MSCPrim> prim_map_;
};

class RelaxWeightsExtractor : public RelaxExprVisitor {
 public:
  /*!
   * \brief The constructor of RelaxGraphBuilder
   * \param ref_module the reference module.
   * \param name the name of the graph.
   * \param options the options of build the graph.
   */
  explicit RelaxWeightsExtractor(const IRModule& ref_module) : RelaxExprVisitor() {
    ref_module_ = ref_module;
  }

  /*! \brief Visit the constant and save weights */
  Map<MSCTensor, NDArray> GetWeights(const relax::Function& func);

  void VisitExpr_(const relax::ConstantNode* op) final;

  void VisitExpr_(const relax::CallNode* op) final;

 private:
  Map<MSCTensor, NDArray> weights_;
  Map<Expr, relax::Function> local_funcs_;
  IRModule ref_module_;
};

class RelayFuncAttrGetter : public RelayExprVisitor {
 public:
  /*! \brief Get the attributes as Map<String, String>*/
  Map<String, String> GetAttrs(const Expr& expr) {
    RelayFuncAttrGetter::VisitExpr(expr);
    return attrs_;
  }

  void VisitExpr_(const relay::CallNode* op) final;

 private:
  Map<String, String> attrs_;
};

/*!
 * \brief A Scope for recording func
 */
class RelayFuncScope {
 public:
  /*! \brief The constructor */
  explicit RelayFuncScope(const String& name) : name_(name) {}

  /*! \brief Add a weight */
  void AddFuncWeight(const String& weight) { func_weights_.push_back(weight); }

  /*! \brief Get weights */
  const Array<String> GetFuncWeights() { return func_weights_; }

 private:
  String name_;
  Array<String> func_weights_;
};

class RelayGraphBuilder : public RelayExprVisitor {
 public:
  /*!
   * \brief The constructor of RelayGraphBuilder
   * \param ref_module the reference module.
   * \param name the name of the graph.
   * \param options the options of build the graph.
   */
  explicit RelayGraphBuilder(const IRModule& ref_module, const String& name,
                             const std::string& options = "")
      : RelayExprVisitor() {
    ref_module_ = ref_module;
    if (options.size() > 0) {
      std::istringstream is(options);
      dmlc::JSONReader reader(&is);
      reader.Read(&config_);
    }
    while (!func_scopes_.empty()) {
      func_scopes_.pop();
    }
    name_ = config_.graph_name.size() > 0 ? String(config_.graph_name) : name;
  }

  /*! \brief Build MSCGraph from relax function*/
  MSCGraph Build(const relay::Function& func);

  /*! \brief Get the config of builder */
  const MSCRBuildConfig config() { return config_; }

  /*! \brief Create and add MSCJoint from expr*/
  MSCJoint AddNode(const Expr& expr, const String& name = "");

  void VisitExpr_(const relay::ConstantNode* op) final;

  void VisitExpr_(const relay::FunctionNode* op) final;

  void VisitExpr_(const relay::CallNode* op) final;

  void VisitExpr_(const relay::TupleNode* val) final;

  void VisitExpr_(const relay::TupleGetItemNode* val) final;

 protected:
  /*! \brief Start a func scope */
  void StartFuncScope(const String& scope);

  /*! \brief End a func scope */
  void EndFuncScope();

  /*! \brief Check if has func scopes left */
  bool HasFuncScope();

 private:
  String name_;
  MSCRBuildConfig config_;
  IRModule ref_module_;
  Array<MSCJoint> nodes_;
  Map<String, MSCTensor> weights_;
  Map<Expr, Array<String>> expr_tensor_map_;
  std::unordered_map<String, std::pair<BaseJoint, size_t>> tensor_input_map_;
  std::stack<RelayFuncScope> func_scopes_;
};

class RelayWeightsExtractor : public RelayExprVisitor {
 public:
  /*! \brief Visit the constant and save weights*/
  Map<MSCTensor, NDArray> GetWeights(const relay::Function& func);

  void VisitExpr_(const relay::ConstantNode* op) final;

 private:
  Map<MSCTensor, NDArray> weights_;
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_CORE_IR_GRAPH_BUILDER_H_
