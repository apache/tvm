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
 * \file meta_ref.h
 * \brief A reference into the metadata section of the Relay text format.
 */

#ifndef TVM_PARSER_META_REF_H_
#define TVM_PARSER_META_REF_H_

#include <tvm/relay/expr.h>

#include <string>

namespace tvm {
namespace parser {

using namespace relay;

/*! \brief A reference to a "meta-expression".
 *
 * In the text format we allow referencing metadata which
 * uses a compact serialization that proceeds the main
 * program body.
 *
 * We can reference this table using an expression of
 * the form `meta[Type][index]`.
 *
 * We must later resolve these references to actual in-memory
 * AST nodes but this requires first parsing the full program
 * then expanding these temporary AST nodes into their corresponding
 * nodes.
 *
 * For example the nth large constant will be pretty-printed as meta[relay.Constant][n]
 * with its compact binary serialization residing in the metadata section at the end
 * of the program.
 */
class MetaRefExprNode : public TempExprNode {
 public:
  /*! \brief The type key of the meta expression. */
  std::string type_key;
  /*! \brief The index into the type key's table. */
  uint64_t node_index;

  void VisitAttrs(tvm::AttrVisitor* v) {}

  // TODO(@jroesch): we probably will need to manually
  // expand these with a pass.
  Expr Realize() const final { return Expr(); }

  static constexpr const char* _type_key = "relay.MetaRefExpr";
  TVM_DECLARE_FINAL_OBJECT_INFO(MetaRefExprNode, TempExprNode);
};

class MetaRefExpr : public TempExpr {
 public:
  /*!
   * \brief The constructor for MetaRefExpr
   * \param type_key The type key of the object in the meta section.
   * \param kind The index into that subfield.
   */
  TVM_DLL MetaRefExpr(std::string type_key, uint64_t node_index);

  TVM_DEFINE_OBJECT_REF_METHODS(MetaRefExpr, TempExpr, MetaRefExprNode);
};

MetaRefExpr::MetaRefExpr(std::string type_key, uint64_t node_index) {
  auto rnode = make_object<MetaRefExprNode>();
  rnode->type_key = type_key;
  rnode->node_index = node_index;
  data_ = std::move(rnode);
}

}  // namespace parser
}  // namespace tvm

#endif  // TVM_PARSER_META_REF_H_
