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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/runtime/tensor.h>
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

using namespace tvm::relax;

using Expr = tvm::RelaxExpr;
using tvm::runtime::Tensor;

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

class AttrGetter {
 public:
  /*!
   * \brief Get the attributes as ffi::Map<ffi::String, ffi::String>
   * \param attrs the attributes.
   */
  explicit AttrGetter(ffi::Map<ffi::String, ffi::String>* attrs) : attrs_(attrs) {}

  void operator()(const Attrs& attrs) {
    if (const auto* dict_attrs = attrs.as<DictAttrsNode>()) {
      for (const auto& [key, value] : dict_attrs->dict) {
        this->VisitAny(key, value);
      }
    } else {
      const TVMFFITypeInfo* attrs_tinfo = TVMFFIGetTypeInfo(attrs->type_index());
      if (attrs_tinfo->metadata != nullptr) {
        tvm::ffi::reflection::ForEachFieldInfo(attrs_tinfo, [&](const TVMFFIFieldInfo* field_info) {
          Any field_value = tvm::ffi::reflection::FieldGetter(field_info)(attrs);
          this->VisitAny(ffi::String(field_info->name), field_value);
        });
      }
    }
  }

 private:
  void VisitAny(ffi::String key, Any value) {
    switch (value.type_index()) {
      case kTVMFFINone: {
        attrs_->Set(key, "");
        break;
      }
      case kTVMFFIBool: {
        attrs_->Set(key, std::to_string(value.cast<bool>()));
        break;
      }
      case kTVMFFIInt: {
        attrs_->Set(key, std::to_string(value.cast<int64_t>()));
        break;
      }
      case kTVMFFIFloat: {
        attrs_->Set(key, std::to_string(value.cast<double>()));
        break;
      }
      case kTVMFFIDataType: {
        attrs_->Set(key, runtime::DLDataTypeToString(value.cast<DLDataType>()));
        break;
      }
      case kTVMFFISmallStr:
      case kTVMFFIStr: {
        attrs_->Set(key, value.cast<ffi::String>());
        break;
      }
      default: {
        if (value.type_index() >= kTVMFFIStaticObjectBegin) {
          attrs_->Set(key, StringUtils::ToString(value.cast<ObjectRef>()));
        } else {
          LOG(FATAL) << "Unsupported type: " << value.type_index();
        }
        break;
      }
    }
  }

 private:
  ffi::Map<ffi::String, ffi::String>* attrs_;
};

class FuncAttrGetter : public ExprVisitor {
 public:
  /*! \brief Get the attributes as ffi::Map<ffi::String, ffi::String>*/
  ffi::Map<ffi::String, ffi::String> GetAttrs(const Expr& expr) {
    VisitExpr(expr);
    return attrs_;
  }

  void VisitExpr_(const CallNode* op) final;

  void VisitExpr_(const TupleGetItemNode* op) final;

 private:
  ffi::Map<ffi::String, ffi::String> attrs_;
};

class FuncValueGetter : public ExprVisitor {
 public:
  /*! \brief Get the attributes from prim value as ffi::Map<ffi::String, ffi::String>*/
  ffi::Array<ffi::String> GetValues(const Expr& expr) {
    VisitExpr(expr);
    return values_;
  }

  void VisitExpr_(const CallNode* op) final;

 private:
  ffi::Array<ffi::String> values_;
};

class FuncParamsFinder : public ExprVisitor {
 public:
  /*!
   * \brief The constructor of FuncParamsFinder
   * \param ref_module the reference module.
   */
  explicit FuncParamsFinder(const IRModule& ref_module) : ExprVisitor() {
    ref_module_ = ref_module;
  }

  /*! \brief Find the func params and bind with arguments*/
  ffi::Map<Expr, Expr> FindParams(const Expr& expr) {
    VisitExpr(expr);
    return params_;
  }

  void VisitBinding_(const VarBindingNode* binding, const FunctionNode* val) final;

  void VisitExpr_(const CallNode* op) final;

 private:
  IRModule ref_module_;
  ffi::Map<Expr, Expr> params_;
  ffi::Map<Expr, Function> local_funcs_;
};

class LayoutsFinder : public ExprVisitor {
 public:
  /*!
   * \brief The constructor of LayoutsFinder
   * \param ref_module the reference module.
   */
  explicit LayoutsFinder(const IRModule& ref_module) : ExprVisitor() { ref_module_ = ref_module; }

  /*! \brief Find the layouts form attrs*/
  ffi::Map<ffi::String, ffi::String> FindLayouts(const Expr& expr) {
    VisitExpr(expr);
    return layouts_;
  }

  void VisitBinding_(const VarBindingNode* binding, const FunctionNode* val) final;

  void VisitExpr_(const CallNode* op) final;

 private:
  IRModule ref_module_;
  ffi::Map<ffi::String, ffi::String> layouts_;
  ffi::Map<Expr, Function> local_funcs_;
};

class GraphBuilder : public ExprVisitor {
 public:
  /*!
   * \brief The constructor of GraphBuilder
   * \param ref_module the reference module.
   * \param name the name of the graph.
   * \param options the options of build the graph.
   */
  explicit GraphBuilder(const IRModule& ref_module, const ffi::String& name,
                        const std::string& options = "")
      : ExprVisitor() {
    ref_module_ = ref_module;
    if (options.size() > 0) {
      std::istringstream is(options);
      dmlc::JSONReader reader(&is);
      reader.Read(&config_);
    }
    name_ = config_.graph_name.size() > 0 ? ffi::String(config_.graph_name) : name;
    if (config_.byoc_entry.size() > 0) {
      func_params_ = FuncParamsFinder(ref_module).FindParams(ref_module->Lookup(name));
    }
    layouts_ = LayoutsFinder(ref_module).FindLayouts(ref_module->Lookup(name));
  }

  /*! \brief Build MSCGraph from relax function*/
  const MSCGraph Build(const Function& func);

  /*! \brief Get the config of builder */
  const MSCRBuildConfig config() { return config_; }

  /*! \brief Create and add MSCJoint from expr*/
  const MSCJoint AddNode(const Expr& expr, const ffi::Optional<Expr>& binding_var = std::nullopt,
                         const ffi::String& name = "");

  /*! \brief Create and add MSCPrim from prim*/
  const MSCPrim AddPrim(const PrimExpr& prim);

  const MSCPrim MatchOrCreatePrim(
      const PrimExpr& prim, const ffi::String& op = "",
      const ffi::Array<BaseJoint>& parents = ffi::Array<BaseJoint>(),
      const ffi::Map<ffi::String, ffi::String>& attrs = ffi::Map<ffi::String, ffi::String>());

  void VisitBindingBlock(const BindingBlock& block) final;

  void VisitExpr_(const ConstantNode* op) final;

  void VisitBinding_(const VarBindingNode* binding, const ConstantNode* val) final;

  void VisitBinding_(const VarBindingNode* binding, const ShapeExprNode* val) final;

  void VisitBinding_(const VarBindingNode* binding, const CallNode* call_node) final;

  void VisitBinding_(const VarBindingNode* binding, const TupleNode* val) final;

  void VisitBinding_(const VarBindingNode* binding, const TupleGetItemNode* val) final;

  void VisitBinding_(const VarBindingNode* binding, const VarNode* val) final;

  void VisitBinding_(const VarBindingNode* binding, const DataflowVarNode* val) final;

  void VisitBinding_(const VarBindingNode* binding, const FunctionNode* val) final;

  void VisitPrimExpr(const PrimExpr& prim) final;

 private:
  /*! \brief Get the node_name, optype, layout for func*/
  const std::tuple<ffi::String, ffi::String, ffi::String> ParseFunc(const Function& func);

  /*! \brief Get the plugin inputs*/
  ffi::Array<Expr> GetPluginInputs(const Expr& expr);

  ffi::String name_;
  IRModule ref_module_;
  MSCRBuildConfig config_;
  ffi::Map<ffi::String, ffi::String> layouts_;
  ffi::Array<MSCJoint> nodes_;
  ffi::Map<ffi::String, MSCTensor> weights_;
  ffi::Map<Expr, ffi::Array<ffi::String>> expr_tensor_map_;
  std::unordered_map<ffi::String, std::pair<BaseJoint, size_t>> tensor_input_map_;
  std::set<ffi::String> ignore_nodes_;
  // scope name
  ffi::String scope_name_;
  std::set<ffi::String> setted_blocks_;
  ffi::Array<ffi::String> block_stack_;
  // BYOC maps
  ffi::Map<Expr, Function> target_funcs_;
  ffi::Map<Expr, Expr> func_params_;
  // prims
  ffi::Array<MSCPrim> prims_;
  ffi::Map<PrimExpr, MSCPrim> prim_map_;
};

class WeightsExtractor : public ExprVisitor {
 public:
  /*!
   * \brief The constructor of GraphBuilder
   * \param ref_module the reference module.
   * \param name the name of the graph.
   * \param options the options of build the graph.
   */
  explicit WeightsExtractor(const IRModule& ref_module) : ExprVisitor() {
    ref_module_ = ref_module;
  }

  /*! \brief Visit the constant and save weights */
  ffi::Map<MSCTensor, Tensor> GetWeights(const Function& func);

  void VisitExpr_(const ConstantNode* op) final;

  void VisitExpr_(const CallNode* op) final;

 private:
  ffi::Map<MSCTensor, Tensor> weights_;
  ffi::Map<Expr, Function> local_funcs_;
  IRModule ref_module_;
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_CORE_IR_GRAPH_BUILDER_H_
