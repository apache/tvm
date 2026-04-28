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
 * \file ir/repr.cc
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
#include <tvm/ir/repr.h>
#include <tvm/runtime/device_api.h>

#include <sstream>

namespace tvm {

void Dump(const runtime::ObjectRef& n) { std::cerr << ffi::ReprPrint(ffi::Any(n)) << "\n"; }

void Dump(const runtime::Object* n) { Dump(runtime::GetRef<runtime::ObjectRef>(n)); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // node.AsRepr: backward-compatible Python entry point.
  // Python's tvm.runtime._ffi_node_api sets __object_repr__ = AsRepr via init_ffi_api.
  refl::GlobalDef().def("node.AsRepr",
                        [](ffi::Any obj) -> ffi::String { return ffi::ReprPrint(obj); });
  // Register __ffi_repr__ for AccessPath/AccessStep so that ffi.ReprPrint
  // uses the concise "<root>.field[idx]" format.
  //
  // AccessStep: format one step fragment (e.g. ".field", "[0]", "[key]?").
  refl::TypeAttrDef<ffi::reflection::AccessStepObj>().def(
      refl::type_attr::kRepr,
      [](ffi::reflection::AccessStep step, ffi::Function fn_repr) -> ffi::String {
        using ffi::reflection::AccessKind;
        std::ostringstream os;
        switch (step->kind) {
          case AccessKind::kAttr:
            os << "." << step->key.cast<ffi::String>();
            break;
          case AccessKind::kArrayItem:
            os << "[" << step->key.cast<int64_t>() << "]";
            break;
          case AccessKind::kMapItem:
            os << "[" << fn_repr(step->key).cast<ffi::String>() << "]";
            break;
          case AccessKind::kAttrMissing:
            os << "." << step->key.cast<ffi::String>() << "?";
            break;
          case AccessKind::kArrayItemMissing:
            os << "[" << step->key.cast<int64_t>() << "]?";
            break;
          case AccessKind::kMapItemMissing:
            os << "[" << fn_repr(step->key).cast<ffi::String>() << "]?";
            break;
        }
        return os.str();
      });
  // AccessPath: recurse through parent via fn_repr rather than walking the
  // linked list manually.  Root (no step) emits "<root>"; each non-root node
  // prepends its parent's repr and appends the current step's repr.
  refl::TypeAttrDef<ffi::reflection::AccessPathObj>().def(
      refl::type_attr::kRepr,
      [](ffi::reflection::AccessPath path, ffi::Function fn_repr) -> ffi::String {
        if (!path->step.has_value()) {
          // Root node: no parent, no step.
          return "<root>";
        }
        std::ostringstream os;
        os << fn_repr(path->parent.value()).cast<ffi::String>();
        os << fn_repr(path->step.value()).cast<ffi::String>();
        return os.str();
      });
}
}  // namespace tvm
