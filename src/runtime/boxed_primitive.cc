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
 * \file src/runtime/boxed_primitive.cc
 * \brief Implementations of ObjectRef wrapper.
 */

#include <tvm/runtime/container/boxed_primitive.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {

TVM_REGISTER_OBJECT_TYPE(BoxNode<int64_t>);
TVM_REGISTER_OBJECT_TYPE(BoxNode<double>);
TVM_REGISTER_OBJECT_TYPE(BoxNode<bool>);

/* \brief Allow explicit construction of Box<bool>
 *
 * Convert a `bool` to `Box<bool>`.  For use in FFI handling, to
 * provide an umambiguous representation between `bool(true)` and
 * `int(1)`.  Will be automatically unboxed in the case where a
 * `Box<bool>` is provided to a PackedFunc that requires `int` input,
 * mimicking C++'s default conversions.
 *
 * This is only needed for Box<bool>, as Box<double> and Box<int64_t>
 * can be converted in C++ as part of `TVMArgValue::operator
 * ObjectRef()` without ambiguity, postponing conversions until
 * required.
 */
TVM_REGISTER_GLOBAL("runtime.BoxBool").set_body_typed([](bool value) { return Box(value); });

/* \brief Return the underlying boolean object.
 *
 * Used while unboxing a boolean return value during FFI handling.
 * The return type is intentionally `int` and not `bool`, to avoid
 * recursive unwrapping of boolean values.
 *
 * This is only needed for Box<bool>, as Box<double> and Box<int64_t>
 * can be unambiguously unboxed as part of
 * `TVMRetValue::operator=(ObjectRef)`.
 */
TVM_REGISTER_GLOBAL("runtime.UnBoxBool").set_body_typed([](Box<bool> obj) -> int {
  return obj->value;
});

}  // namespace runtime
}  // namespace tvm
