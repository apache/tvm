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

#ifndef TVM_RELAY_PARSER_META_REF_H_
#define TVM_RELAY_PARSER_META_REF_H_

#include <tvm/ir/attrs.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/relay/parser.h>

#include <string>

namespace tvm {
namespace relay {

/*!
 * \brief Options for allocating storage.
 */
struct MetaRefAttrs : public tvm::AttrsNode<MetaRefAttrs> {
  tvm::String node_type_key;
  uint64_t node_index;

  TVM_DECLARE_ATTRS(MetaRefAttrs, "relay.attrs.MetaRefAttrs") {
    TVM_ATTR_FIELD(node_type_key)
        .describe("The type_key representing the type of the node referenced.");
    TVM_ATTR_FIELD(node_index).describe("The index into the type specific node array.");
  }
};

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
 *
 * \param type_key The type key of the object in the meta section.
 * \param node_index The index into that subfield.
 * \returns The meta table reference.
 */
Expr MetaRef(std::string type_key, uint64_t node_index);

relay::Function ExpandMetaRefs(const MetaTable& meta_table, const relay::Function& func);
IRModule ExpandMetaRefs(const MetaTable& meta_table, const IRModule& mod);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_PARSER_META_REF_H_
