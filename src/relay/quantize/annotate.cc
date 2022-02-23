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
 *
 * \file annotate.cc
 *
 * \brief Annotating the graph with simulated quantize operators.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/transform.h>

#include "./quantize.h"

namespace tvm {
namespace relay {
namespace quantize {

using namespace relay::transform;

class QAnnotateExpr;
class QAnnotateExprNode : public TempExprNode {
 public:
  Expr expr;
  QAnnotateKind kind;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("expr", &expr);
    v->Visit("kind", &kind);
  }

  Expr Realize() const final;

  static constexpr const char* _type_key = "relay.QAnnotateExpr";
  TVM_DECLARE_FINAL_OBJECT_INFO(QAnnotateExprNode, TempExprNode);
};

class QAnnotateExpr : public TempExpr {
 public:
  /*!
   * \brief The constructor
   * \param expr The original relay expression.
   * \param kind The annotation kind.
   */
  TVM_DLL QAnnotateExpr(Expr expr, QAnnotateKind kind);

  TVM_DEFINE_OBJECT_REF_METHODS(QAnnotateExpr, TempExpr, QAnnotateExprNode);
};

Expr QAnnotateExprNode::Realize() const { return expr; }

QAnnotateExpr::QAnnotateExpr(Expr expr, QAnnotateKind kind) {
  auto rnode = make_object<QAnnotateExprNode>();
  rnode->expr = std::move(expr);
  rnode->kind = kind;
  data_ = std::move(rnode);
}

TVM_REGISTER_GLOBAL("relay._quantize.make_annotate_expr").set_body_typed([](Expr expr, int kind) {
  return QAnnotateExpr(expr, static_cast<QAnnotateKind>(kind));
});

Pass QuantizeAnnotate() {
  // TODO(tvm-teams): since partition has added cast_hint in different
  // branches, try to remove this in the future.
  std::function<Expr(const Expr&)> fmulti_ref = [](const Expr& e) {
    if (e->IsInstance<TempExprNode>()) {
      const auto* n = e.as<QAnnotateExprNode>();
      ICHECK(n);
      const PackedFunc* f = runtime::Registry::Get("relay.quantize.attach_simulated_quantize");
      Expr ret = (*f)(n->expr, static_cast<int>(kQInput));
      return static_cast<Expr>(QAnnotateExpr(ret, kQInput));
    }
    return e;
  };

  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        auto func = Downcast<Function>(ForwardRewrite(f, "FQAnnotateRewrite", nullptr, fmulti_ref));
        auto new_params = func->params;
        for (const auto& x : FreeVars(func)) {
          new_params.push_back(x);
        }
        return WithFields(func, new_params);
      };
  return CreateFunctionPass(pass_func, 1, "QuantizeAnnotate", {});
}

TVM_REGISTER_GLOBAL("relay._quantize.QuantizeAnnotate").set_body_typed(QuantizeAnnotate);

TVM_REGISTER_NODE_TYPE(QAnnotateExprNode);

}  // namespace quantize
}  // namespace relay
}  // namespace tvm
