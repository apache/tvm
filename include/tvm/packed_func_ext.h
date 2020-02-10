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
 * \file tvm/packed_func_ext.h
 * \brief Extension package to PackedFunc
 *   This enales pass ObjectRef types into/from PackedFunc.
 */
#ifndef TVM_PACKED_FUNC_EXT_H_
#define TVM_PACKED_FUNC_EXT_H_

#include <tvm/top/tensor.h>

#include <string>
#include <memory>
#include <limits>
#include <type_traits>

#include "expr.h"
#include "runtime/packed_func.h"

namespace tvm {

using runtime::TVMArgs;
using runtime::TVMRetValue;
using runtime::PackedFunc;

namespace runtime {


template<typename T>
struct ObjectTypeChecker<Array<T> > {
  static bool Check(const Object* ptr) {
    if (ptr == nullptr) return true;
    if (!ptr->IsInstance<ArrayNode>()) return false;
    const ArrayNode* n = static_cast<const ArrayNode*>(ptr);
    for (const auto& p : n->data) {
      if (!ObjectTypeChecker<T>::Check(p.get())) {
        return false;
      }
    }
    return true;
  }
  static std::string TypeName() {
    return "List[" + ObjectTypeChecker<T>::TypeName() + "]";
  }
};

template<typename V>
struct ObjectTypeChecker<Map<std::string, V> > {
  static bool Check(const Object* ptr) {
    if (ptr == nullptr) return true;
    if (!ptr->IsInstance<StrMapNode>()) return false;
    const StrMapNode* n = static_cast<const StrMapNode*>(ptr);
    for (const auto& kv : n->data) {
      if (!ObjectTypeChecker<V>::Check(kv.second.get())) return false;
    }
    return true;
  }
  static std::string TypeName() {
    return "Map[str, " +
        ObjectTypeChecker<V>::TypeName()+ ']';
  }
};

template<typename K, typename V>
struct ObjectTypeChecker<Map<K, V> > {
  static bool Check(const Object* ptr) {
    if (ptr == nullptr) return true;
    if (!ptr->IsInstance<MapNode>()) return false;
    const MapNode* n = static_cast<const MapNode*>(ptr);
    for (const auto& kv : n->data) {
      if (!ObjectTypeChecker<K>::Check(kv.first.get())) return false;
      if (!ObjectTypeChecker<V>::Check(kv.second.get())) return false;
    }
    return true;
  }
  static std::string TypeName() {
    return "Map[" +
        ObjectTypeChecker<K>::TypeName() +
        ", " +
        ObjectTypeChecker<V>::TypeName()+ ']';
  }
};

// extensions for tvm arg value
inline TVMPODValue_::operator tvm::PrimExpr() const {
  if (type_code_ == kTVMNullptr) return PrimExpr();
  if (type_code_ == kDLInt) {
    CHECK_LE(value_.v_int64, std::numeric_limits<int>::max());
    CHECK_GE(value_.v_int64, std::numeric_limits<int>::min());
    return PrimExpr(static_cast<int>(value_.v_int64));
  }
  if (type_code_ == kDLFloat) {
    return PrimExpr(static_cast<float>(value_.v_float64));
  }

  TVM_CHECK_TYPE_CODE(type_code_, kTVMObjectHandle);
  Object* ptr = static_cast<Object*>(value_.v_handle);

  if (ptr->IsInstance<IterVarNode>()) {
    return IterVar(ObjectPtr<Object>(ptr))->var;
  }
  if (ptr->IsInstance<top::TensorNode>()) {
    return top::Tensor(ObjectPtr<Object>(ptr))();
  }
  CHECK(ObjectTypeChecker<PrimExpr>::Check(ptr))
      << "Expect type " << ObjectTypeChecker<PrimExpr>::TypeName()
      << " but get " << ptr->GetTypeKey();
  return PrimExpr(ObjectPtr<Object>(ptr));
}

inline TVMPODValue_::operator tvm::Integer() const {
  if (type_code_ == kTVMNullptr) return Integer();
  if (type_code_ == kDLInt) {
    CHECK_LE(value_.v_int64, std::numeric_limits<int>::max());
    CHECK_GE(value_.v_int64, std::numeric_limits<int>::min());
    return Integer(static_cast<int>(value_.v_int64));
  }
  TVM_CHECK_TYPE_CODE(type_code_, kTVMObjectHandle);
  Object* ptr = static_cast<Object*>(value_.v_handle);
  CHECK(ObjectTypeChecker<Integer>::Check(ptr))
      << "Expect type " << ObjectTypeChecker<PrimExpr>::TypeName()
      << " but get " << ptr->GetTypeKey();
  return Integer(ObjectPtr<Object>(ptr));
}
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_PACKED_FUNC_EXT_H_
