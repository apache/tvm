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
 * \file src/node/structural_hash.cc
 */
#include <dmlc/memory_io.h>
#include <tvm/ffi/extra/base64.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/access_path.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/node/functor.h>
#include <tvm/node/node.h>
#include <tvm/node/structural_hash.h>
#include <tvm/runtime/profiling.h>
#include <tvm/target/codegen.h>

#include <algorithm>
#include <unordered_map>

#include "../support/base64.h"
#include "../support/str_escape.h"
#include "../support/utils.h"

namespace tvm {

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("node.StructuralHash",
                        [](const Any& object, bool map_free_vars) -> int64_t {
                          return ffi::StructuralHash::Hash(object, map_free_vars);
                        });
  refl::TypeAttrDef<ffi::ModuleObj>()
      .def("__data_to_json__",
           [](const ffi::ModuleObj* node) {
             std::string bytes = codegen::SerializeModuleToBytes(ffi::GetRef<ffi::Module>(node),
                                                                 /*export_dso*/ false);
             return ffi::Base64Encode(ffi::Bytes(bytes));
           })
      .def("__data_from_json__", [](const ffi::String& base64_bytes) {
        ffi::Bytes bytes = ffi::Base64Decode(base64_bytes);
        ffi::Module rtmod = codegen::DeserializeModuleFromBytes(bytes.operator std::string());
        return rtmod;
      });

  refl::TypeAttrDef<runtime::Tensor::Container>()
      .def("__data_to_json__",
           [](const runtime::Tensor::Container* node) {
             std::string blob;
             dmlc::MemoryStringStream mstrm(&blob);
             support::Base64OutStream b64strm(&mstrm);
             runtime::SaveDLTensor(&b64strm, node);
             b64strm.Finish();
             return ffi::String(blob);
           })
      .def("__data_from_json__", [](const std::string& blob) {
        dmlc::MemoryStringStream mstrm(const_cast<std::string*>(&blob));
        support::Base64InStream b64strm(&mstrm);
        b64strm.InitPosition();
        runtime::Tensor temp;
        ICHECK(temp.Load(&b64strm));
        return temp;
      });
}

uint64_t StructuralHash::operator()(const ffi::Any& object) const {
  return ffi::StructuralHash::Hash(object, false);
}

struct RefToObjectPtr : public ObjectRef {
  static ObjectPtr<Object> Get(const ObjectRef& ref) {
    return ffi::details::ObjectUnsafe::ObjectPtrFromObjectRef<Object>(ref);
  }
};

struct ReportNodeTrait {
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<runtime::profiling::ReportNode>()
        .def_ro("calls", &runtime::profiling::ReportNode::calls)
        .def_ro("device_metrics", &runtime::profiling::ReportNode::device_metrics)
        .def_ro("configuration", &runtime::profiling::ReportNode::configuration);
  }
};

TVM_FFI_STATIC_INIT_BLOCK() { ReportNodeTrait::RegisterReflection(); }

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<runtime::profiling::ReportNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const runtime::profiling::ReportNode*>(node.get());
      p->stream << op->AsTable();
    });

struct CountNodeTrait {
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<runtime::profiling::CountNode>().def_ro("value",
                                                            &runtime::profiling::CountNode::value);
  }
};

TVM_FFI_STATIC_INIT_BLOCK() { CountNodeTrait::RegisterReflection(); }

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<runtime::profiling::CountNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const runtime::profiling::CountNode*>(node.get());
      p->stream << op->GetTypeKey() << "(" << op->value << ")";
    });

struct DurationNodeTrait {
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<runtime::profiling::DurationNode>().def_ro(
        "microseconds", &runtime::profiling::DurationNode::microseconds);
  }
};

TVM_FFI_STATIC_INIT_BLOCK() { DurationNodeTrait::RegisterReflection(); }

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<runtime::profiling::DurationNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const runtime::profiling::DurationNode*>(node.get());
      p->stream << op->GetTypeKey() << "(" << op->microseconds << ")";
    });

struct PercentNodeTrait {
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<runtime::profiling::PercentNode>().def_ro(
        "percent", &runtime::profiling::PercentNode::percent);
  }
};

TVM_FFI_STATIC_INIT_BLOCK() { PercentNodeTrait::RegisterReflection(); }

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<runtime::profiling::PercentNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const runtime::profiling::PercentNode*>(node.get());
      p->stream << op->GetTypeKey() << "(" << op->percent << ")";
    });

struct RatioNodeTrait {
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<runtime::profiling::RatioNode>().def_ro("ratio",
                                                            &runtime::profiling::RatioNode::ratio);
  }
};

TVM_FFI_STATIC_INIT_BLOCK() { RatioNodeTrait::RegisterReflection(); }

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<runtime::profiling::RatioNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const runtime::profiling::RatioNode*>(node.get());
      p->stream << op->GetTypeKey() << "(" << op->ratio << ")";
    });

}  // namespace tvm
