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
 * \file tvm/ir/string_type.h
 * \brief string type.
 */
#ifndef TVM_IR_STRING_TYPE_H_
#define TVM_IR_STRING_TYPE_H_

#include <tvm/ir/expr.h>
#include <tvm/ir/type.h>

namespace tvm {

class StringTypeNode : public TypeNode {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("span", &span); }

  bool SEqualReduce(const StringTypeNode* other, SEqualReducer equal) const { return true; }

  void SHashReduce(SHashReducer hash_reduce) const {
    static String a = "relay.StringType";
    hash_reduce(a);
  }

  static constexpr const char* _type_key = "relay.StringType";
  TVM_DECLARE_FINAL_OBJECT_INFO(StringTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to StringTypeNode.
 * \sa StringTypeNode.
 */
class StringType : public Type {
 public:
  TVM_DLL StringType();

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(StringType, Type, StringTypeNode);
};

}  // namespace tvm
#endif  // TVM_IR_STRING_TYPE_H_
