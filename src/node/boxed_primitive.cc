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
 * \file node/boxed_primitive.cc
 *
 * \brief Reflection utilities for runtime-supported classes
 *
 * The fundamental support for boxing and unboxing of primitives
 * during FFI calls is implemented in runtime/boxed_primitive.cc.  In
 * addition, boxed primitives may be registered with compile-time
 * utilities (e.g. reflection, JSON import/export) that can provide
 * additional functionality and improved debugging ability.  However,
 * neither these compile-time utilities nor any registration of
 * `Box<Prim>` into the compile-time utilities should be included as
 * part of `libtvm_runtime.so`.
 *
 * This file contains the registration of the `libtvm_runtime.so`
 * class `Box<Prim>` for utilities that are contained in `libtvm.so`.
 */
#include <tvm/ir/attrs.h>
#include <tvm/node/node.h>
#include <tvm/node/reflection.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime_ext {

using runtime::Box;
using runtime::BoxNode;

/* \brief Compile-time extension trait for runtime types
 *
 * Extends the use of boxed primitive during TVM's compilation step.
 *
 * Most TVM classes define these functions as part of the class
 * definition.  However, the boxed primitives must be usable at
 * runtime, and so the class definition may only refer to types that
 * are present in `libtvm_runtime.so`.
 */
template <typename Prim>
struct BoxNodeCompileTimeTraits {
  static constexpr const std::nullptr_t VisitAttrs = nullptr;

  static void SHashReduce(const BoxNode<Prim>* node, SHashReducer hash_reduce) {
    hash_reduce(node->value);
  }

  static bool SEqualReduce(const BoxNode<Prim>* lhs, const BoxNode<Prim>* rhs,
                           SEqualReducer equal) {
    return equal(lhs->value, rhs->value);
  }
};

TVM_REGISTER_REFLECTION_VTABLE(BoxNode<int64_t>, BoxNodeCompileTimeTraits<int64_t>)
    .set_creator([](const std::string& blob) -> ObjectPtr<Object> {
      int64_t value = std::atoll(blob.c_str());
      return make_object<BoxNode<int64_t>>(value);
    })
    .set_repr_bytes([](const Object* n) -> std::string {
      int64_t value = GetRef<ObjectRef>(n).as<Box<int64_t>>().value()->value;
      std::stringstream ss;
      ss << value;
      return ss.str();
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<BoxNode<int64_t>>([](const ObjectRef& node, ReprPrinter* p) {
      auto box = Downcast<Box<int64_t>>(node);
      p->stream << box->GetTypeKey() << "(" << box->value << ")";
    });

TVM_REGISTER_REFLECTION_VTABLE(BoxNode<bool>, BoxNodeCompileTimeTraits<bool>)
    .set_creator([](const std::string& blob) -> ObjectPtr<Object> {
      if (blob == "true") {
        return make_object<BoxNode<bool>>(true);
      } else if (blob == "false") {
        return make_object<BoxNode<bool>>(false);
      } else {
        LOG(FATAL) << "Invalid string '" << blob << "' for boolean";
      }
    })
    .set_repr_bytes([](const Object* n) -> std::string {
      bool value = GetRef<ObjectRef>(n).as<Box<bool>>().value()->value;
      if (value) {
        return "true";
      } else {
        return "false";
      }
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<BoxNode<bool>>([](const ObjectRef& node, ReprPrinter* p) {
      auto box = Downcast<Box<bool>>(node);
      p->stream << box->GetTypeKey() << "(" << (box->value ? "true" : "false") << ")";
    });

TVM_REGISTER_REFLECTION_VTABLE(BoxNode<double>, BoxNodeCompileTimeTraits<double>)
    .set_creator([](const std::string& blob) -> ObjectPtr<Object> {
      double value = std::atof(blob.c_str());
      return make_object<BoxNode<double>>(value);
    })
    .set_repr_bytes([](const Object* n) -> std::string {
      double value = GetRef<ObjectRef>(n).as<Box<double>>().value()->value;
      std::stringstream ss;
      ss << value;
      return ss.str();
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<BoxNode<double>>([](const ObjectRef& node, ReprPrinter* p) {
      auto box = Downcast<Box<double>>(node);
      p->stream << box->GetTypeKey() << "(" << box->value << ")";
    });

}  // namespace runtime_ext

}  // namespace tvm
