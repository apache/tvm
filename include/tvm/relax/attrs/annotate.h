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
 * \file tvm/relax/attrs/annotate.h
 * \brief Attributes for annotate operators.
 */
#ifndef TVM_RELAX_ATTRS_ANNOTATE_H_
#define TVM_RELAX_ATTRS_ANNOTATE_H_

#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

enum SQAnnotateKind : int { kSQNone = 0, kSQActivation = 1, kSQWeight = 2 };

/*! \brief Attributes for annotate.smooth operator */
struct AnnotateSmoothAttrs : public tvm::AttrsNode<AnnotateSmoothAttrs> {
  int kind;
  String mode;

  TVM_DECLARE_ATTRS(AnnotateSmoothAttrs, "relax.attrs.AnnotateSmoothAttrs") {
    TVM_ATTR_FIELD(kind).set_default(kSQNone).describe("Kind of argument to be annotated.");
    TVM_ATTR_FIELD(mode)
        .set_default("identity")
        .describe("Execution mode for op. Can be: \"identity\", \"multiply\" or \"quantize\"");
  }
};  // struct AnnotateSmoothAttrs

/*! \brief Attributes for annotate.absmax operator */
struct AnnotateAbsMaxAttrs : public tvm::AttrsNode<AnnotateAbsMaxAttrs> {
  int kind;

  TVM_DECLARE_ATTRS(AnnotateAbsMaxAttrs, "relax.attrs.AnnotateAbsMaxAttrs") {
    TVM_ATTR_FIELD(kind).set_default(kSQNone).describe("Kind of argument to be annotated.");
  }
};  // struct AnnotateAbsMaxAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_ANNOTATE_H_
