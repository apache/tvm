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
 *  Copyright (c) 2018 by Contributors.
 *
 * \file tvm/relay/pass/quantize.h
 * \brief Header of definitions for quantization
 */
#ifndef TVM_RELAY_PASS_QUANTIZE_H_
#define TVM_RELAY_PASS_QUANTIZE_H_

#include <tvm/relay/op.h>
#include <tvm/relay/expr.h>
#include <string>
#include "pattern_util.h"

namespace tvm {
namespace relay {
namespace quantize {

/*! \brief Kind of annotate field */
enum QAnnotateKind : int {
  kQInput = 1,
  kQWeight = 2,
  kQActivation = 3,
};

/*!
 * \brief TempExpr used during annotate forward rewrite.
 */
class QAnnotateExpr;
/*!
 * \brief TempExprNode used during annotate forward rewrite.
 */
class QAnnotateExprNode : public TempExprNode {
 public:
  /*! \brief The original expression */
  Expr expr;
  /*! \brief The kind of annotate field */
  QAnnotateKind kind;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("expr", &expr);
    v->Visit("kind", &kind);
  }

  TVM_DLL static QAnnotateExpr make(Expr expr, QAnnotateKind kind);

  Expr Realize() const final;

  static constexpr const char* _type_key = "relay.QAnnotateExpr";
  TVM_DECLARE_NODE_TYPE_INFO(QAnnotateExprNode, TempExprNode);
};

RELAY_DEFINE_NODE_REF(QAnnotateExpr, QAnnotateExprNode, TempExpr);


/*!
 * \brief TempExpr used to insert `force_cast` for VTA.
 */
class QVTAExpr;
/*!
 * \brief TempExprNode used to insert `force_cast` for VTA.
 */
class QVTAExprNode : public TempExprNode {
 public:
  /*! \brief The original expression */
  Expr expr;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("expr", &expr);
  }

  TVM_DLL static QVTAExpr make(Expr expr);

  Expr Realize() const final;

  static constexpr const char* _type_key = "relay.QVTAExpr";
  TVM_DECLARE_NODE_TYPE_INFO(QVTAExprNode, TempExprNode);
};

RELAY_DEFINE_NODE_REF(QVTAExpr, QVTAExprNode, TempExpr);


/*! \brief TempExpr used during realize forward rewrite. */
class QRealizeExpr;
/*! \brief TempExpr representing integer. */
class QRealizeIntExpr;

class QRealizeExprNode : public TempExprNode {
 public:
  /*! \brief The original expression */
  Expr data;
  static constexpr const char* _type_key = "relay.quantize.QRealizeExpr";
  TVM_DECLARE_BASE_NODE_INFO(QRealizeExprNode, TempExprNode);
};

RELAY_DEFINE_NODE_REF(QRealizeExpr, QRealizeExprNode, TempExpr);


class QRealizeIntExprNode : public QRealizeExprNode {
 public:
  Expr dom_scale;
  /*! \brief current data type */
  DataType dtype;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("data", &data);
    v->Visit("dom_scale", &dom_scale);
    v->Visit("dtype", &dtype);
  }

  Expr Realize() const final;

  TVM_DLL static QRealizeIntExpr make(Expr data, Expr dom_scale, DataType dtype);

  static constexpr const char * _type_key = "relay.quantize.QRealizeIntExpr";
  TVM_DECLARE_NODE_TYPE_INFO(QRealizeIntExprNode, QRealizeExprNode);
};

RELAY_DEFINE_NODE_REF(QRealizeIntExpr, QRealizeIntExprNode, QRealizeExpr);


class QConfig;

/*!
* \brief Container for build configuration options
*/
class QConfigNode : public Node {
 public:
  int nbit_input = 8;
  int nbit_weight = 8;
  int nbit_activation = 32;
  DataType dtype_input = Int(8);
  DataType dtype_weight = Int(8);
  DataType dtype_activation = Int(32);
  double global_scale = 8.0;
  Array<Expr> skip_conv_layers = Array<Expr>(NodePtr<Node>(nullptr));
  bool round_for_shift = true;
  bool store_lowbit_output = true;
  Array<Expr> debug_enabled_ops = Array<Expr>(NodePtr<Node>(nullptr));

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("nbit_input", &nbit_input);
    v->Visit("nbit_weight", &nbit_weight);
    v->Visit("nbit_activation", &nbit_activation);
    v->Visit("dtype_input", &dtype_input);
    v->Visit("dtype_weight", &dtype_weight);
    v->Visit("dtype_activation", &dtype_activation);
    v->Visit("global_scale", &global_scale);
    v->Visit("skip_conv_layers", &skip_conv_layers);
    v->Visit("round_for_shift", &round_for_shift);
    v->Visit("store_lowbit_output", &store_lowbit_output);
    v->Visit("debug_enabled_ops", &debug_enabled_ops);
  }

  static constexpr const char* _type_key = "relay.quantize.QConfig";
  TVM_DECLARE_NODE_TYPE_INFO(QConfigNode, Node);
};

/*!
* \brief Container for build configuration options
*/
class QConfig : public NodeRef {
 public:
  QConfig() {}
  explicit QConfig(NodePtr<Node> n) : NodeRef(n) {}

  const QConfigNode* operator->() const {
    return static_cast<const QConfigNode*>(node_.get());
  }

  QConfigNode* operator->() {
    return static_cast<QConfigNode*>(node_.get());
  }

  /*!
   * \brief Push a new BuildConfig context onto the thread local stack.
   * \param build_config The configuration to set as the current context.
   */
  static void EnterQConfigScope(const QConfig& qconfig);

  /*!
   * \brief Pop a build config off the thread local context stack, restoring the previous
   * configuration as the current context.
   */
  static void ExitQConfigScope();

  /*!
   * \brief Get the current BuildConfig context from thread local storage, or a default
   * configuration if a BuildConfig scope has not been entered.
   * \return The configuration that is the current context.
   */
  static QConfig& Current();

  using ContainerType = QConfigNode;
};

/*!
 * \brief RAII container to provide a scoped BuildConfig context. Pushes a configuration onto the
 * context stack when constructed, and pops it when destructed.
 */
struct QConfigContext {
  /*!
   * \brief Enter a new BuildConfig context. The given BuildConfig becomes the new current
   * context. When the BuildConfigContext is destructed, the previous context is restored.
   * \param build_config The BuildConfig to set as the new current context.
   */
  explicit QConfigContext(const QConfig& qconfig) {
    QConfig::EnterQConfigScope(qconfig);
  }

  /*! \brief Destructor. Pops the context off the thread local stack. */
  ~QConfigContext() {
    QConfig::ExitQConfigScope();
  }
};

/*!
* \brief Construct a BuildConfig containing a new BuildConfigNode
* \return The new BuildConfig
*/
TVM_DLL QConfig qconfig();

}  // namespace quantize
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_PASS_QUANTIZE_H_
