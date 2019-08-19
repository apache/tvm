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
 * Copyright (c) 2018 by Contributors
 *
 * \file partition.cc
 *
 * \brief Partition a graph into sections for quantization.
 */

#include <tvm/relay/transform.h>
#include "../pattern_util.h"
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

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("expr", &expr);
  }

  TVM_DLL static QPartitionExpr make(Expr expr);

  Expr Realize() const final;

  static constexpr const char* _type_key = "relay.QPartitionExpr";
  TVM_DECLARE_NODE_TYPE_INFO(QPartitionExprNode, TempExprNode);
};

RELAY_DEFINE_NODE_REF(QPartitionExpr, QPartitionExprNode, TempExpr);


Expr QPartitionExprNode::Realize() const {
  // insert cast hint and stop fusion
  const QConfig& cfg = QConfig::Current();
  Expr ret = CastHint(this->expr, cfg->dtype_input);
  return StopFusion(ret);
}

QPartitionExpr QPartitionExprNode::make(Expr expr) {
  auto rnode = make_node<QPartitionExprNode>();
  rnode->expr = expr;
  return QPartitionExpr(rnode);
}

TVM_REGISTER_API("relay._quantize.make_partition_expr")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    *ret = QPartitionExprNode::make(args[0]);
  });

Pass QuantizePartition() {
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
    [=](Function f, Module m, PassContext pc) {
      auto ret = Downcast<Function>(
          ForwardRewrite(f, "FQPartitionRewrite", nullptr, nullptr));
      return ret;
  };
  return CreateFunctionPass(pass_func, 1, "QuantizePartition", {});
}

TVM_REGISTER_API("relay._quantize.QuantizePartition")
.set_body_typed(QuantizePartition);

}  // namespace quantize
}  // namespace relay
}  // namespace tvm
