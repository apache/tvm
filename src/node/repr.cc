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
 * \file node/repr.cc
 * \brief Implements Dump helpers and FFI registration for ffi-repr-based printing.
 *
 * The legacy ReprPrinter has been replaced by ffi::ReprPrint.  This file:
 *  - Implements the Dump() debug helpers (they call ffi::ReprPrint).
 *  - Registers node.AsRepr (for backward Python compatibility) via ffi::ReprPrint.
 *  - Registers __ffi_repr__ hooks for AccessPath and AccessStep.
 */
#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/access_path.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/node/repr.h>
#include <tvm/runtime/device_api.h>

#include <sstream>
#include <vector>

#include "../support/str_escape.h"

namespace tvm {

void Dump(const runtime::ObjectRef& n) {
  std::cerr << ffi::ReprPrint(ffi::Any(n)) << "\n";
}

void Dump(const runtime::Object* n) { Dump(runtime::GetRef<runtime::ObjectRef>(n)); }

namespace {
/*!
 * \brief Format an AccessStep as a concise string fragment.
 */
void FormatAccessStep(std::ostringstream& os, const ffi::reflection::AccessStep& step) {
  using ffi::reflection::AccessKind;
  static const ffi::Function repr_fn = ffi::Function::GetGlobal("ffi.ReprPrint").value();
  switch (step->kind) {
    case AccessKind::kAttr:
      os << "." << step->key.cast<ffi::String>();
      break;
    case AccessKind::kArrayItem:
      os << "[" << step->key.cast<int64_t>() << "]";
      break;
    case AccessKind::kMapItem:
      os << "[" << repr_fn(step->key).cast<ffi::String>() << "]";
      break;
    case AccessKind::kAttrMissing:
      os << "." << step->key.cast<ffi::String>() << "?";
      break;
    case AccessKind::kArrayItemMissing:
      os << "[" << step->key.cast<int64_t>() << "]?";
      break;
    case AccessKind::kMapItemMissing:
      os << "[" << repr_fn(step->key).cast<ffi::String>() << "]?";
      break;
  }
}

/*!
 * \brief Format an AccessPath as "<root>.field[idx]".
 */
ffi::String FormatAccessPath(const ffi::reflection::AccessPath& path) {
  std::vector<ffi::reflection::AccessStep> steps;
  const ffi::reflection::AccessPathObj* cur = path.get();
  while (cur->step.defined()) {
    steps.push_back(cur->step.value());
    cur = static_cast<const ffi::reflection::AccessPathObj*>(cur->parent.get());
  }
  std::ostringstream os;
  os << "<root>";
  for (auto it = steps.rbegin(); it != steps.rend(); ++it) {
    FormatAccessStep(os, *it);
  }
  return os.str();
}
}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // node.AsRepr: backward-compatible Python entry point.
  // Python's tvm.runtime._ffi_node_api sets __object_repr__ = AsRepr via init_ffi_api.
  refl::GlobalDef().def("node.AsRepr", [](ffi::Any obj) -> ffi::String {
    return ffi::ReprPrint(obj);
  });
  // Register __ffi_repr__ for AccessPath/AccessStep so that ffi.ReprPrint
  // uses the concise "<root>.field[idx]" format.
  refl::TypeAttrDef<ffi::reflection::AccessPathObj>().def(
      refl::type_attr::kRepr,
      [](ffi::reflection::AccessPath path, ffi::Function) -> ffi::String {
        return FormatAccessPath(path);
      });
  refl::TypeAttrDef<ffi::reflection::AccessStepObj>().def(
      refl::type_attr::kRepr,
      [](ffi::reflection::AccessStep step, ffi::Function) -> ffi::String {
        std::ostringstream os;
        FormatAccessStep(os, step);
        return os.str();
      });
}
}  // namespace tvm
