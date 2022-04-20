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
 * \file tvm/relay/attrs/annotation.h
 * \brief Attribute for annotation operators.
 */
#ifndef TVM_RELAY_ATTRS_ANNOTATION_H_
#define TVM_RELAY_ATTRS_ANNOTATION_H_

#include <tvm/ir/attrs.h>

#include <string>

namespace tvm {
namespace relay {

/*!
 * \brief Annotate an expression to be cast into specific data type.
 */
struct CastHintAttrs : public tvm::AttrsNode<CastHintAttrs> {
  DataType dtype;

  TVM_DECLARE_ATTRS(CastHintAttrs, "relay.attrs.CastHintAttrs") {
    TVM_ATTR_FIELD(dtype).describe("The data type denoted to be cast.");
  }
};

/*!
 * \brief Options for the operators used to annotate a compiler.
 */
struct CompilerAttrs : public tvm::AttrsNode<CompilerAttrs> {
  /*! \brief A 3rd party compiler for code generation. */
  std::string compiler;

  TVM_DECLARE_ATTRS(CompilerAttrs, "relay.attrs.CompilerAttrs") {
    TVM_ATTR_FIELD(compiler).describe("A 3rd party compiler used for code generation.");
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_ANNOTATION_H_
