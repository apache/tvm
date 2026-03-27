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
 * Printer utilities
 * \file node/repr_printer.cc
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/access_path.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/node/cast.h>
#include <tvm/node/repr_printer.h>
#include <tvm/runtime/device_api.h>

#include <sstream>
#include <vector>

#include "../support/str_escape.h"

namespace tvm {

void ReprPrinter::Print(const ObjectRef& node) {
  static const FType& f = vtable();
  if (!node.defined()) {
    stream << "(nullptr)";
  } else {
    if (f.can_dispatch(node)) {
      f(node, this);
    } else {
      // default value, output type key and addr.
      stream << node->GetTypeKey() << "(" << node.get() << ")";
    }
  }
}

void ReprPrinter::Print(const ffi::Any& node) {
  switch (node.type_index()) {
    case ffi::TypeIndex::kTVMFFINone: {
      stream << "(nullptr)";
      break;
    }
    case ffi::TypeIndex::kTVMFFIInt: {
      stream << node.cast<int64_t>();
      break;
    }
    case ffi::TypeIndex::kTVMFFIBool: {
      stream << node.cast<bool>();
      break;
    }
    case ffi::TypeIndex::kTVMFFIFloat: {
      stream << node.cast<double>();
      break;
    }
    case ffi::TypeIndex::kTVMFFIOpaquePtr: {
      stream << node.cast<void*>();
      break;
    }
    case ffi::TypeIndex::kTVMFFIDataType: {
      stream << node.cast<DataType>();
      break;
    }
    case ffi::TypeIndex::kTVMFFIDevice: {
      runtime::operator<<(stream, node.cast<Device>());
      break;
    }
    case ffi::TypeIndex::kTVMFFIObject: {
      Print(node.cast<ObjectRef>());
      break;
    }
    case ffi::TypeIndex::kTVMFFISmallStr:
    case ffi::TypeIndex::kTVMFFIStr: {
      ffi::String str = node.cast<ffi::String>();
      stream << '"' << support::StrEscape(str.data(), str.size()) << '"';
      break;
    }
    case ffi::TypeIndex::kTVMFFISmallBytes:
    case ffi::TypeIndex::kTVMFFIBytes: {
      ffi::Bytes bytes = node.cast<ffi::Bytes>();
      stream << "b\"" << support::StrEscape(bytes.data(), bytes.size()) << '"';
      break;
    }
    default: {
      if (auto opt_obj = node.as<ObjectRef>()) {
        Print(opt_obj.value());
      } else {
        stream << "Any(type_key=`" << node.GetTypeKey() << "`)";
      }
      break;
    }
  }
}

void ReprPrinter::PrintIndent() {
  for (int i = 0; i < indent; ++i) {
    stream << ' ';
  }
}

ReprPrinter::FType& ReprPrinter::vtable() {
  static FType inst;
  return inst;
}

void Dump(const runtime::ObjectRef& n) { std::cerr << n << "\n"; }

void Dump(const runtime::Object* n) { Dump(runtime::GetRef<runtime::ObjectRef>(n)); }

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ffi::reflection::AccessPathObj>([](const ObjectRef& node, ReprPrinter* p) {
      p->stream << Downcast<ffi::reflection::AccessPath>(node);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ffi::reflection::AccessStepObj>([](const ObjectRef& node, ReprPrinter* p) {
      p->stream << Downcast<ffi::reflection::AccessStep>(node);
    });

namespace {
/*!
 * \brief Format an AccessStep as a concise string fragment.
 */
void FormatAccessStep(std::ostringstream& os, const ffi::reflection::AccessStep& step) {
  using ffi::reflection::AccessKind;
  switch (step->kind) {
    case AccessKind::kAttr:
      os << "." << step->key.cast<ffi::String>();
      break;
    case AccessKind::kArrayItem:
      os << "[" << step->key.cast<int64_t>() << "]";
      break;
    case AccessKind::kMapItem:
      os << "{" << step->key.cast<ffi::String>() << "}";
      break;
    case AccessKind::kAttrMissing:
      os << "." << step->key.cast<ffi::String>() << "?";
      break;
    case AccessKind::kArrayItemMissing:
      os << "[" << step->key.cast<int64_t>() << "]?";
      break;
    case AccessKind::kMapItemMissing:
      os << "{" << step->key.cast<ffi::String>() << "}?";
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
  refl::GlobalDef().def("node.AsRepr", [](ffi::Any obj) {
    std::ostringstream os;
    os << obj;
    return os.str();
  });
  // Register __ffi_repr__ for AccessPath/AccessStep so that ffi.ReprPrint
  // uses the concise "<root>.field[idx]" format instead of the dataclass repr.
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
