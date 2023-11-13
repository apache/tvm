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
 * \file tvm/runtime/container/boxed_primitive.h
 * \brief Runtime container types for primitives stored as ObjectRef.
 */
#ifndef TVM_RUNTIME_CONTAINER_BOXED_PRIMITIVE_H_
#define TVM_RUNTIME_CONTAINER_BOXED_PRIMITIVE_H_

#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>

namespace tvm {
namespace runtime {

namespace detail {
/* \brief Provide the BoxNode<T> type key in templated contexts
 *
 * The Box<T> class is used in many templated contexts, and is easier
 * to have templated over the primitive type.  However, much of the
 * TVM type system depends on classes having a unique name.  For
 * example, the use of `Object::IsInstance` depends on
 * `Object::GetOrAllocRuntimeTypeIndex`.  Any duplicate names will
 * result in duplicate indices, and invalid downcasting.
 *
 * Furthermore, the name must be specified in the Python FFI using
 * `tvm._ffi.register_object`.  This prevents use of
 * `typeid(T)::name()` to build a unique name, as the name is not
 * required to be human-readable or consistent across compilers.
 *
 * This utility struct exists to bridge that gap, providing a unique
 * name where required.
 */
template <typename Prim>
struct BoxNodeTypeKey;

template <>
struct BoxNodeTypeKey<int64_t> {
  static constexpr const char* _type_key = "runtime.BoxInt";
};

template <>
struct BoxNodeTypeKey<double> {
  static constexpr const char* _type_key = "runtime.BoxFloat";
};

template <>
struct BoxNodeTypeKey<bool> {
  static constexpr const char* _type_key = "runtime.BoxBool";
};
}  // namespace detail

template <typename Prim>
class BoxNode : public Object {
 public:
  /*! \brief Constructor
   *
   * \param value The value to be boxed
   */
  BoxNode(Prim value) : value(value) {}

  /*! \brief The boxed value */
  Prim value;

  static constexpr const char* _type_key = detail::BoxNodeTypeKey<Prim>::_type_key;
  static constexpr bool _type_has_method_visit_attrs = false;
  TVM_DECLARE_FINAL_OBJECT_INFO(BoxNode, Object);
};

template <typename Prim>
class Box : public ObjectRef {
 public:
  /*! \brief Constructor
   *
   * \param value The value to be boxed
   */
  Box(Prim value) : ObjectRef(make_object<BoxNode<Prim>>(value)) {}

  operator Prim() const { return (*this)->value; }

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Box, ObjectRef, BoxNode<Prim>);
};

/*! \brief Runtime equivalent of IntImm */
using BoxInt = Box<int64_t>;

/*! \brief Runtime equivalent of FloatImm */
using BoxFloat = Box<double>;

/*! \brief Runtime equivalent of IntImm with DataType::Bool()
 *
 * When passing from Python to C++, TVM PackedFunc conversion follow
 * C++ conversion rules, and allow bool->int and int->bool
 * conversions.  When passing from C++ to Python, the types are
 * returned as bool or int.  If the C++ function uses ObjectRef to
 * hold the object, a Python to C++ to Python round trip will preserve
 * the distinction between bool and int.
 */
using BoxBool = Box<bool>;

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTAINER_BOXED_PRIMITIVE_H_
