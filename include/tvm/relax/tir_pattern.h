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
 * \file tir_pattern.h
 * \brief Data Structure of TIR Pattern used for matching.
 */

#ifndef TVM_RELAX_TIR_PATTERN_H_
#define TVM_RELAX_TIR_PATTERN_H_

#include <tvm/tir/function.h>

namespace tvm {
namespace relax {

using TIRPattern = tir::PrimFunc;

/*
 * \brief The match result of a TIR pattern.
 */
class MatchResultNode : public Object {
 public:
  /*! The matched tir pattern*/
  TIRPattern pattern;
  /*! \brief The evaluated values of symbolic vars. */
  Array<PrimExpr> symbol_values;
  /*! \brief The matched buffers of input and output. */
  Array<tir::Buffer> matched_buffers;
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("pattern", &pattern);
    v->Visit("symbol_values", &symbol_values);
    v->Visit("matched_buffers", &matched_buffers);
  }
  static constexpr const char* _type_key = "relax.MatchResult";
  TVM_DECLARE_FINAL_OBJECT_INFO(MatchResultNode, Object);
};

/*!
 * \brief Managed reference to MatchResultNode.
 */
class MatchResult : public ObjectRef {
 public:
  /*!
   * \brief Constructor
   * \param pattern The matched tir pattern.
   * \param symbol_values The evaluated values of symbolic vars.
   * \param matched_buffers The matched buffers of input and output.
   */
  TVM_DLL explicit MatchResult(TIRPattern pattern, Array<PrimExpr> symbol_values,
                               Array<tir::Buffer> matched_buffers);

  TVM_DEFINE_OBJECT_REF_METHODS(MatchResult, ObjectRef, MatchResultNode)
};

using FCodegen = runtime::TypedPackedFunc<Array<ObjectRef>(Array<MatchResult> match_results)>;
}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_TIR_PATTERN_H_
