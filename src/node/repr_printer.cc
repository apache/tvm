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
#include <tvm/node/repr_printer.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/device_api.h>

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
      stream << node.operator int64_t();
      break;
    }
    case ffi::TypeIndex::kTVMFFIBool: {
      stream << node.operator bool();
      break;
    }
    case ffi::TypeIndex::kTVMFFIFloat: {
      stream << node.operator double();
      break;
    }
    case ffi::TypeIndex::kTVMFFIOpaquePtr: {
      stream << node.operator void*();
      break;
    case ffi::TypeIndex::kTVMFFIDataType:
      stream << node.operator DataType();
      break;
    }
    case ffi::TypeIndex::kTVMFFIDevice: {
      runtime::operator<<(stream, node.operator Device());
      break;
    }
    case ffi::TypeIndex::kTVMFFIObject: {
      Print(node.operator ObjectRef());
      break;
    }
    default: {
      if (auto opt_obj = node.TryAs<ObjectRef>()) {
        Print(opt_obj.value());
      } else {
        stream << "Any(type_key=`" << ffi::TypeIndex2TypeKey(node.type_index()) << "`)";
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

void ReprLegacyPrinter::Print(const ObjectRef& node) {
  static const FType& f = vtable();
  if (!node.defined()) {
    stream << "(nullptr)";
  } else if (f.can_dispatch(node)) {
    f(node, this);
  } else {
    try {
      stream << node;  // Use ReprPrinter
    } catch (const tvm::Error& e) {
      LOG(WARNING) << "ReprPrinter fails";
      stream << node->GetTypeKey() << '(' << node.get() << ')';
    }
  }
}

bool ReprLegacyPrinter::CanDispatch(const ObjectRef& node) {
  static const FType& f = vtable();
  return !node.defined() || f.can_dispatch(node);
}

void ReprLegacyPrinter::PrintIndent() {
  for (int i = 0; i < indent; ++i) {
    stream << ' ';
  }
}

ReprLegacyPrinter::FType& ReprLegacyPrinter::vtable() {
  static FType inst;
  return inst;
}

void Dump(const runtime::ObjectRef& n) { std::cerr << n << "\n"; }

void Dump(const runtime::Object* n) { Dump(runtime::GetRef<runtime::ObjectRef>(n)); }

TVM_REGISTER_GLOBAL("node.AsRepr").set_body_typed([](runtime::ObjectRef obj) {
  std::ostringstream os;
  os << obj;
  return os.str();
});

TVM_REGISTER_GLOBAL("node.AsLegacyRepr").set_body_typed(ffi::AsLegacyRepr);

}  // namespace tvm
