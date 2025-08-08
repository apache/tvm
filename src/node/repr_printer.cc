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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/node/cast.h>
#include <tvm/node/repr_printer.h>
#include <tvm/runtime/device_api.h>

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

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("node.AsRepr", [](ffi::Any obj) {
    std::ostringstream os;
    os << obj;
    return os.str();
  });
});
}  // namespace tvm
