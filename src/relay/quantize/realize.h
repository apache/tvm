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
 * \file realize.h
 *
 * \brief Header of definitions for op realizations
 *
 */
#ifndef TVM_RELAY_QUANTIZE_REALIZE_H_
#define TVM_RELAY_QUANTIZE_REALIZE_H_

#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {
namespace quantize {

class QRealizeExprNode : public TempExprNode {
 public:
  Expr data;
  static constexpr const char* _type_key = "relay.quantize.QRealizeExpr";
  TVM_DECLARE_BASE_OBJECT_INFO(QRealizeExprNode, TempExprNode);
};

class QRealizeExpr : public TempExpr {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(QRealizeExpr, TempExpr, QRealizeExprNode);
};

class QRealizeIntExprNode : public QRealizeExprNode {
 public:
  Expr dom_scale;
  DataType dtype;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("data", &data);
    v->Visit("dom_scale", &dom_scale);
    v->Visit("dtype", &dtype);
  }

  Expr Realize() const final;

  static constexpr const char* _type_key = "relay.quantize.QRealizeIntExpr";
  TVM_DECLARE_FINAL_OBJECT_INFO(QRealizeIntExprNode, QRealizeExprNode);
};

class QRealizeIntExpr : public QRealizeExpr {
 public:
  TVM_DLL QRealizeIntExpr(Expr data, Expr dom_scale, DataType dtype);

  TVM_DEFINE_OBJECT_REF_METHODS(QRealizeIntExpr, QRealizeExpr, QRealizeIntExprNode);
};

}  // namespace quantize
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_QUANTIZE_REALIZE_H_
