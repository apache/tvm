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
 * \file partition.cc
 *
 * \brief Partition a graph into sections for quantization.
 */

#include <tvm/relay/transform.h>

#include "../op/annotation/annotation.h"
#include "./quantize.h"

namespace tvm {
namespace relay {
namespace quantize {

using namespace relay::transform;

class QPartitionExpr;
class QPartitionExprNode : public TempExprNode {
 public:
  /*! \brief The original expression */
  Expr expr;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("expr", &expr); }

  Expr Realize() const final;

  static constexpr const char* _type_key = "relay.QPartitionExpr";
  TVM_DECLARE_FINAL_OBJECT_INFO(QPartitionExprNode, TempExprNode);
};

class QPartitionExpr : public TempExpr {
 public:
  /*!
   * \brief  The constructor
   * \param expr The original relay expression.
   */
  TVM_DLL explicit QPartitionExpr(Expr expr);

  TVM_DEFINE_OBJECT_REF_METHODS(QPartitionExpr, TempExpr, QPartitionExprNode);
};

Expr QPartitionExprNode::Realize() const {
  // insert cast hint and stop fusion
  const QConfig& cfg = QConfig::Current();
  Expr ret = CastHint(this->expr, cfg->dtype_input);
  return StopFusion(ret);
}

QPartitionExpr::QPartitionExpr(Expr expr) {
  auto rnode = make_object<QPartitionExprNode>();
  rnode->expr = std::move(expr);
  data_ = std::move(rnode);
}

TVM_REGISTER_GLOBAL("relay._quantize.make_partition_expr").set_body_typed([](Expr expr) {
  return QPartitionExpr(expr);
});

Pass QuantizePartition() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        auto ret = Downcast<Function>(ForwardRewrite(f, "FQPartitionRewrite", nullptr, nullptr));
        return ret;
      };
  return CreateFunctionPass(pass_func, 1, "QuantizePartition", {});
}

TVM_REGISTER_GLOBAL("relay._quantize.QuantizePartition").set_body_typed(QuantizePartition);

TVM_REGISTER_NODE_TYPE(QPartitionExprNode);

}  // namespace quantize
}  // namespace relay
}  // namespace tvm
