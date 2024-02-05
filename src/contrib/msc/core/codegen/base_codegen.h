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
 * \file src/contrib/msc/core/codegen/base_codegen.h
 * \brief Basic CodeGen for MSCGraph and MSCJoint.
 */
#ifndef TVM_CONTRIB_MSC_CORE_CODEGEN_BASE_CODEGEN_H_
#define TVM_CONTRIB_MSC_CORE_CODEGEN_BASE_CODEGEN_H_

#include <dmlc/json.h>
#include <tvm/script/printer/doc.h>

#include <memory>
#include <stack>
#include <string>
#include <vector>

#include "../ir/graph.h"
#include "../ir/plugin.h"
#include "code_stack.h"
#include "codegen_utils.h"

namespace tvm {
namespace contrib {
namespace msc {

using namespace tvm::script::printer;

/*!
 * \brief CodeGen for MSCJoint op
 */
template <typename ConfigType, typename HelperType>
class BaseOpCode {
 public:
  /*!
   * \brief The constructor of BaseOpCode
   * \param func_name the function name for the node.
   */
  explicit BaseOpCode(const String& func_name) : func_name_(func_name) {}

  virtual ~BaseOpCode() = default;

  /*! \brief Config the BaseOpCode*/
  void Config(const MSCJoint& node, const std::shared_ptr<ConfigType> config) {
    node_ = node;
    config_ = config;
  }

  /*! \brief Get docs for the node*/
  virtual const Array<Doc> GetDocs() = 0;

  /*! \brief Get return describe for default node*/
  virtual const String IdxNode() { return IdxNodeBase(node_); }

  /*! \brief Get describe for default node input*/
  const String IdxInput(int idx = 0, bool process = true) {
    return IdxInputBase(node_, idx, process);
  }

  /*! \brief Get describe for default node output*/
  const String IdxOutput(int idx = 0) { return IdxOutputBase(node_, idx); }

  /*! \brief Get describe for default node weight*/
  const String IdxWeight(const String& wtype, bool process = true) {
    return IdxWeightBase(node_, wtype, process);
  }

  /*! \brief Get the node attr as doc*/
  const ExprDoc GetAttrDoc(const String& key, const String& type) {
    if (StringUtils::StartsWith(type, "list")) {
      const String& ele_type =
          StringUtils::Replace(StringUtils::Replace(type, "list(", ""), ")", "");
      if (ele_type == "bool") {
        return DocUtils::ToList(node_->GetTypeArrayAttr<bool>(key));
      } else if (ele_type == "int" || ele_type == "int32") {
        return DocUtils::ToList(node_->GetTypeArrayAttr<int>(key));
      } else if (ele_type == "long" || ele_type == "int64") {
        return DocUtils::ToList(node_->GetTypeArrayAttr<int64_t>(key));
      } else if (ele_type == "float" || ele_type == "float32") {
        return DocUtils::ToList(node_->GetTypeArrayAttr<float>(key));
      } else if (ele_type == "string") {
        return DocUtils::ToStrList(node_->GetTypeArrayAttr<std::string>(key));
      }
    } else if (type == "bool") {
      return DocUtils::ToDoc(node_->GetTypeAttr<bool>(key));
    } else if (type == "int" || type == "int32") {
      return DocUtils::ToDoc(node_->GetTypeAttr<int>(key));
    } else if (type == "long" || type == "int64") {
      return DocUtils::ToDoc(node_->GetTypeAttr<int64_t>(key));
    } else if (type == "float" || type == "float32") {
      return DocUtils::ToDoc(node_->GetTypeAttr<float>(key));
    } else if (type == "string") {
      return DocUtils::ToStr(node_->GetTypeAttr<std::string>(key));
    }
    return DocUtils::ToDoc(node_->GetTypeAttr<std::string>(key));
  }

  /*! \brief Get comment for default node*/
  const String Comment() { return Comment(node_); }

  /*! \brief Get func_name for the default node*/
  const String func_name() { return func_name_; }

  /*! \brief Get valid func name for the default node*/
  virtual const String callee_name() { return func_name(); }

  /*! \brief Get valid return name for the default node*/
  virtual const String ret_name() { return IdxNode(); }

  /*! \brief Get the default node*/
  const MSCJoint node() { return node_; }

  CODEGEN_MEMBERS;

 private:
  String func_name_;
  MSCJoint node_;
};

/*!
 * \brief CodeGen for MSCGraph
 */
template <typename ConfigType, typename HelperType>
class BaseCodeGen {
 public:
  /*!
   * \brief The constructor of BaseCodeGen
   * \param graph the graph to be generated.
   * \param config the options for codegen.
   */
  explicit BaseCodeGen(const MSCGraph& graph, const std::string& config = "") {
    graph_ = graph;
    config_.reset(new ConfigType());
    if (config.size() > 0) {
      std::istringstream is(config);
      dmlc::JSONReader reader(&is);
      reader.Read(config_.get());
    }
    while (!scopes_.empty()) {
      scopes_.pop();
    }
  }

  virtual ~BaseCodeGen() = default;

  /*! \brief Get sources*/
  virtual const Map<String, String> GetSources(const std::string& print_options = "") = 0;

  CODEGEN_MEMBERS;

 protected:
  /*!
   * \brief Compare node scope with current scope
   * 0 for same scope, 1 for increase scope, -1 for decrease scope
   */
  int CompareScope(const MSCJoint& node) {
    if (node->scope.size() == 0) {
      return 0;
    }
    if (scopes_.size() == 0) {
      scopes_.push(node->scope);
      return 1;
    }
    if (node->scope.size() == scopes_.top().size()) {
      ICHECK(StringUtils::CompareArrays(node->scope, scopes_.top()))
          << "Scope mismatch, node " << node->scope << " compare to current " << scopes_.top();
      return 0;
    } else if (node->scope.size() == scopes_.top().size() + 1) {
      ICHECK(StringUtils::CompareArrays(node->scope, scopes_.top(), scopes_.top().size()))
          << "Scope increase mismatch, node " << node->scope << " compare to current "
          << scopes_.top();
      scopes_.push(node->scope);
      return 1;
    } else if (node->scope.size() == scopes_.top().size() - 1) {
      ICHECK(StringUtils::CompareArrays(node->scope, scopes_.top(), node->scope.size()))
          << "Scope decrease mismatch, node " << node->scope << " compare to current "
          << scopes_.top();
      scopes_.pop();
      return -1;
    } else {
      LOG(FATAL) << "Unexpected node scope " << node->scope << " with current scope "
                 << scopes_.top();
    }
  }

  /*! \brief Get the optype for op codegen*/
  const String GetOpType(const MSCJoint& node) {
    if (config_->use_plugin && IsPlugin(node->optype)) {
      return "plugin";
    }
    return node->optype;
  }

  /*! \brief Get the docs for the op*/
  virtual const Array<Doc> GetOpCodes(const MSCJoint& node) = 0;

  /*! \brief Get the graph*/
  const MSCGraph graph() const { return graph_; }

  /*! \brief Get the scopes*/
  const std::stack<Array<String>> scopes() const { return scopes_; }

  /*! \brief The stack of codes*/
  CodeStack stack_;

 private:
  MSCGraph graph_;
  std::stack<Array<String>> scopes_;
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_CORE_CODEGEN_BASE_CODEGEN_H_
