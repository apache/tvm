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
 * \file tvm/ir/repr.h
 * \brief ostream operator<< for ObjectRef, Any, and Variant, delegating to
 *        ffi::ReprPrint.  Also re-exports the Dump() debug helpers.
 *
 * Include this header wherever you need `os << some_objectref` and you are
 * no longer pulling in the legacy repr_printer.h.
 */
#ifndef TVM_IR_REPR_H_
#define TVM_IR_REPR_H_

#include <tvm/ffi/extra/dataclass.h>
#include <tvm/runtime/object.h>

#include <iostream>

namespace tvm {

/*!
 * \brief Dump the node to stderr, used for debug purposes.
 * \param node The input node
 */
TVM_DLL void Dump(const runtime::ObjectRef& node);

/*!
 * \brief Dump the node to stderr, used for debug purposes.
 * \param node The input node
 */
TVM_DLL void Dump(const runtime::Object* node);

}  // namespace tvm

namespace tvm {
namespace ffi {

// ostream << ObjectRef  — delegates to ffi::ReprPrint
inline std::ostream& operator<<(std::ostream& os, const ObjectRef& n) {  // NOLINT(*)
  return os << ffi::ReprPrint(Any(n));
}

// ostream << Any — delegates to ffi::ReprPrint
inline std::ostream& operator<<(std::ostream& os, const Any& n) {  // NOLINT(*)
  return os << ffi::ReprPrint(n);
}

// ostream << Variant<...> — delegates to ffi::ReprPrint
template <typename... V>
inline std::ostream& operator<<(std::ostream& os, const ffi::Variant<V...>& n) {  // NOLINT(*)
  return os << ffi::ReprPrint(Any(n));
}

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_IR_REPR_H_
