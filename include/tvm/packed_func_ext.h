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

#include <sstream>
#include <string>
#include <memory>
#include <limits>
#include <type_traits>

#include "base.h"
#include "expr.h"
#include "tensor.h"
#include "runtime/packed_func.h"

namespace tvm {

using runtime::TVMArgs;
using runtime::TVMRetValue;
using runtime::PackedFunc;

namespace runtime {
/*!
 * \brief Runtime type checker for node type.
 * \tparam T the type to be checked.
 */
template<typename T>
struct ObjectTypeChecker {
  static bool Check(const Object* ptr) {
    using ContainerType = typename T::ContainerType;
    if (ptr == nullptr) return true;
    return ptr->IsInstance<ContainerType>();
  }
  static void PrintName(std::ostream& os) { // NOLINT(*)
    using ContainerType = typename T::ContainerType;
    os << ContainerType::_type_key;
  }
};

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
  static void PrintName(std::ostream& os) { // NOLINT(*)
    os << "List[";
    ObjectTypeChecker<T>::PrintName(os);
    os << "]";
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
  static void PrintName(std::ostream& os) { // NOLINT(*)
    os << "Map[str";
    os << ',';
    ObjectTypeChecker<V>::PrintName(os);
    os << ']';
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
  static void PrintName(std::ostringstream& os) { // NOLINT(*)
    os << "Map[";
    ObjectTypeChecker<K>::PrintName(os);
    os << ',';
    ObjectTypeChecker<V>::PrintName(os);
    os << ']';
  }
};

template<typename T>
inline std::string ObjectTypeName() {
  std::ostringstream os;
  ObjectTypeChecker<T>::PrintName(os);
  return os.str();
}

// extensions for tvm arg value

template<typename TObjectRef>
inline TObjectRef TVMArgValue::AsObjectRef() const {
  static_assert(
      std::is_base_of<ObjectRef, TObjectRef>::value,
      "Conversion only works for ObjectRef");
  if (type_code_ == kNull) return TObjectRef(NodePtr<Node>(nullptr));
  TVM_CHECK_TYPE_CODE(type_code_, kObjectHandle);
  Object* ptr = static_cast<Object*>(value_.v_handle);
  CHECK(ObjectTypeChecker<TObjectRef>::Check(ptr))
      << "Expected type " << ObjectTypeName<TObjectRef>()
      << " but get " << ptr->GetTypeKey();
  return TObjectRef(ObjectPtr<Node>(ptr));
}

inline TVMArgValue::operator tvm::Expr() const {
  if (type_code_ == kNull) return Expr();
  if (type_code_ == kDLInt) {
    CHECK_LE(value_.v_int64, std::numeric_limits<int>::max());
    CHECK_GE(value_.v_int64, std::numeric_limits<int>::min());
    return Expr(static_cast<int>(value_.v_int64));
  }
  if (type_code_ == kDLFloat) {
    return Expr(static_cast<float>(value_.v_float64));
  }

  TVM_CHECK_TYPE_CODE(type_code_, kObjectHandle);
  Object* ptr = static_cast<Object*>(value_.v_handle);

  if (ptr->IsInstance<IterVarNode>()) {
    return IterVar(ObjectPtr<Node>(ptr))->var;
  }
  if (ptr->IsInstance<TensorNode>()) {
    return Tensor(ObjectPtr<Node>(ptr))();
  }
  CHECK(ObjectTypeChecker<Expr>::Check(ptr))
      << "Expected type " << ObjectTypeName<Expr>()
      << " but get " << ptr->GetTypeKey();
  return Expr(ObjectPtr<Node>(ptr));
}

inline TVMArgValue::operator tvm::Integer() const {
  if (type_code_ == kNull) return Integer();
  if (type_code_ == kDLInt) {
    CHECK_LE(value_.v_int64, std::numeric_limits<int>::max());
    CHECK_GE(value_.v_int64, std::numeric_limits<int>::min());
    return Integer(static_cast<int>(value_.v_int64));
  }
  TVM_CHECK_TYPE_CODE(type_code_, kObjectHandle);
  Object* ptr = static_cast<Object*>(value_.v_handle);
  CHECK(ObjectTypeChecker<Integer>::Check(ptr))
      << "Expected type " << ObjectTypeName<Expr>()
      << " but get " << ptr->GetTypeKey();
  return Integer(ObjectPtr<Node>(ptr));
}

template<typename TObjectRef, typename>
inline bool TVMPODValue_::IsObjectRef() const {
  TVM_CHECK_TYPE_CODE(type_code_, kObjectHandle);
  Object* ptr = static_cast<Object*>(value_.v_handle);
  return ObjectTypeChecker<TObjectRef>::Check(ptr);
}

// extensions for TVMRetValue
template<typename TObjectRef>
inline TObjectRef TVMRetValue::AsObjectRef() const {
  static_assert(
      std::is_base_of<ObjectRef, TObjectRef>::value,
      "Conversion only works for ObjectRef");
  if (type_code_ == kNull) return TObjectRef();
  TVM_CHECK_TYPE_CODE(type_code_, kObjectHandle);

  Object* ptr = static_cast<Object*>(value_.v_handle);

  CHECK(ObjectTypeChecker<TObjectRef>::Check(ptr))
      << "Expected type " << ObjectTypeName<TObjectRef>()
      << " but get " << ptr->GetTypeKey();
  return TObjectRef(ObjectPtr<Object>(ptr));
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_PACKED_FUNC_EXT_H_
