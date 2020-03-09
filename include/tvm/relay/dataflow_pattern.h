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
 * \file tvm/relay/dataflow_pattern.h
 * \brief A pattern language for matching dataflow properties.
 */
#ifndef TVM_RELAY_DATAFLOW_PATTERN_H_
#define TVM_RELAY_DATAFLOW_PATTERN_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/type.h>

namespace tvm {
namespace relay {

/*!
 * \brief Base type of all dataflow patterns.
 * \sa DFPattern
 */
class DFPatternNode : public Object {
 public:
  static constexpr const char* _type_key = "DFPatternNode";
  TVM_DECLARE_BASE_OBJECT_INFO(DFPatternNode, Object);
};

/*!
 * \brief Managed reference to dataflow patterns.
 * \sa DFPatternNode
 */
class DFPattern : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(DFPattern, ObjectRef, DFPatternNode);
};

/*!
 * \brief A pattern which matches a literal expression.
 *
 * \note Uses structural equality on expressions to check equality.
 *
 */
class ExprPattern;
/*!
 * \brief Constant tensor type.
 */
class ExprPatternNode : public DFPatternNode {
 public:
  /*! \brief The expression to match. */
  Expr expr;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("expr", &expr);
  }

  static constexpr const char* _type_key = "relay.df_pattern.ExprPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(ExprPatternNode, DFPatternNode);
};

class ExprPattern : public DFPattern {
 public:
  TVM_DLL ExprPattern(Expr expr);
  TVM_DEFINE_OBJECT_REF_METHODS(ExprPattern, DFPattern, ExprPatternNode);
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_DATAFLOW_PATTERN_H_
