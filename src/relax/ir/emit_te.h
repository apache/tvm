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
 * \file relax/src/ir/emit_te.h
 * \brief Tensor expression extension in Relax.
 */
#ifndef TVM_RELAX_IR_EMIT_TE_H_
#define TVM_RELAX_IR_EMIT_TE_H_

#include <tvm/relax/expr.h>
#include <tvm/te/operation.h>

#include <string>

namespace tvm {
namespace relax {

/*!
 * \brief A placeholder op that represents a relax expression.
 */
class RXPlaceholderOpNode : public te::PlaceholderOpNode {
 public:
  /*! \brief The relax expression. */
  Expr value;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("tag", &tag);
    v->Visit("attrs", &attrs);
    v->Visit("value", &value);
    v->Visit("shape", &shape);
    v->Visit("dtype", &dtype);
  }

  static constexpr const char* _type_key = "RXPlaceholderOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(RXPlaceholderOpNode, te::PlaceholderOpNode);
};

/*!
 * \brief Create a TE tensor from relax expression, with TIR variables in the
 * tensor shape substituted by the given mapping.
 * \param value The relax expression, which is required to have TensorStructInfo.
 * \param tir_var_map The mapping to substitute the TIR variables appeared in the
 * shape of the input Expr.
 * \param name The name of the created tensor.
 */
te::Tensor TETensor(Expr value, Map<tir::Var, PrimExpr> tir_var_map, std::string name);

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_IR_EMIT_TE_H_
