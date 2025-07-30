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
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/reflection/structural_hash.h>
#include <tvm/node/functor.h>
#include <tvm/node/node.h>
#include <tvm/node/object_path.h>
#include <tvm/node/reflection.h>
#include <tvm/node/structural_hash.h>
#include <tvm/runtime/profiling.h>
#include <tvm/target/codegen.h>

#include <algorithm>
#include <unordered_map>

#include "../support/base64.h"
#include "../support/str_escape.h"
#include "../support/utils.h"

namespace tvm {

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("node.StructuralHash",
                        [](const Any& object, bool map_free_vars) -> int64_t {
                          return ffi::reflection::StructuralHash::Hash(object, map_free_vars);
                        });
});

uint64_t StructuralHash::operator()(const ffi::Any& object) const {
  return ffi::reflection::StructuralHash::Hash(object, false);
}

struct RefToObjectPtr : public ObjectRef {
  static ObjectPtr<Object> Get(const ObjectRef& ref) {
    return ffi::details::ObjectUnsafe::ObjectPtrFromObjectRef<Object>(ref);
  }
};

TVM_REGISTER_REFLECTION_VTABLE(ffi::StringObj)
    .set_creator([](const std::string& bytes) { return RefToObjectPtr::Get(String(bytes)); })
    .set_repr_bytes([](const Object* n) -> std::string {
      return GetRef<String>(static_cast<const ffi::StringObj*>(n)).operator std::string();
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ffi::StringObj>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ffi::StringObj*>(node.get());
      p->stream << '"' << support::StrEscape(op->data, op->size) << '"';
    });

TVM_REGISTER_REFLECTION_VTABLE(ffi::BytesObj)
    .set_creator([](const std::string& bytes) { return RefToObjectPtr::Get(String(bytes)); })
    .set_repr_bytes([](const Object* n) -> std::string {
      return GetRef<ffi::Bytes>(static_cast<const ffi::BytesObj*>(n)).operator std::string();
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ffi::BytesObj>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ffi::BytesObj*>(node.get());
      p->stream << "b\"" << support::StrEscape(op->data, op->size) << '"';
    });

TVM_REGISTER_REFLECTION_VTABLE(runtime::ModuleNode)
    .set_creator([](const std::string& blob) {
      runtime::Module rtmod = codegen::DeserializeModuleFromBytes(blob);
      return RefToObjectPtr::Get(rtmod);
    })
    .set_repr_bytes([](const Object* n) -> std::string {
      const auto* rtmod = static_cast<const runtime::ModuleNode*>(n);
      return codegen::SerializeModuleToBytes(GetRef<runtime::Module>(rtmod), /*export_dso*/ false);
    });

TVM_REGISTER_REFLECTION_VTABLE(runtime::NDArray::Container)
    .set_creator([](const std::string& blob) {
      dmlc::MemoryStringStream mstrm(const_cast<std::string*>(&blob));
      support::Base64InStream b64strm(&mstrm);
      b64strm.InitPosition();
      runtime::NDArray temp;
      ICHECK(temp.Load(&b64strm));
      return RefToObjectPtr::Get(temp);
    })
    .set_repr_bytes([](const Object* n) -> std::string {
      std::string blob;
      dmlc::MemoryStringStream mstrm(&blob);
      support::Base64OutStream b64strm(&mstrm);
      const auto* ndarray = static_cast<const runtime::NDArray::Container*>(n);
      runtime::SaveDLTensor(&b64strm, ndarray);
      b64strm.Finish();
      return blob;
    });

TVM_REGISTER_REFLECTION_VTABLE(ffi::ArrayObj)
    .set_creator([](const std::string&) -> ObjectPtr<Object> {
      return ffi::make_object<ffi::ArrayObj>();
    });

TVM_REGISTER_REFLECTION_VTABLE(ffi::ShapeObj)
    .set_creator([](const std::string& blob) {
      // Store shape tuple in blob to avoid large integer overflow in JSON.
      dmlc::MemoryStringStream mstrm(const_cast<std::string*>(&blob));
      support::Base64InStream b64strm(&mstrm);
      b64strm.InitPosition();
      uint64_t size;
      b64strm.Read<uint64_t>(&size);
      std::vector<int64_t> data(size);
      b64strm.ReadArray(data.data(), size);
      ffi::Shape shape(data);
      return RefToObjectPtr::Get(shape);
    })
    .set_repr_bytes([](const Object* n) -> std::string {
      std::string blob;
      dmlc::MemoryStringStream mstrm(&blob);
      support::Base64OutStream b64strm(&mstrm);
      const auto* shape = static_cast<const ffi::ShapeObj*>(n);
      b64strm.Write<uint64_t>(shape->size);
      b64strm.WriteArray(shape->data, shape->size);
      b64strm.Finish();
      return blob;
    });

TVM_REGISTER_REFLECTION_VTABLE(ffi::MapObj)
    .set_creator([](const std::string&) -> ObjectPtr<Object> { return ffi::MapObj::Empty(); });

struct ReportNodeTrait {
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<runtime::profiling::ReportNode>()
        .def_ro("calls", &runtime::profiling::ReportNode::calls)
        .def_ro("device_metrics", &runtime::profiling::ReportNode::device_metrics)
        .def_ro("configuration", &runtime::profiling::ReportNode::configuration);
  }
};

TVM_FFI_STATIC_INIT_BLOCK({ ReportNodeTrait::RegisterReflection(); });
TVM_REGISTER_REFLECTION_VTABLE(runtime::profiling::ReportNode);

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

TVM_FFI_STATIC_INIT_BLOCK({ CountNodeTrait::RegisterReflection(); });

TVM_REGISTER_REFLECTION_VTABLE(runtime::profiling::CountNode);
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

TVM_FFI_STATIC_INIT_BLOCK({ DurationNodeTrait::RegisterReflection(); });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<runtime::profiling::DurationNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const runtime::profiling::DurationNode*>(node.get());
      p->stream << op->GetTypeKey() << "(" << op->microseconds << ")";
    });
TVM_REGISTER_REFLECTION_VTABLE(runtime::profiling::DurationNode);

struct PercentNodeTrait {
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<runtime::profiling::PercentNode>().def_ro(
        "percent", &runtime::profiling::PercentNode::percent);
  }
};

TVM_FFI_STATIC_INIT_BLOCK({ PercentNodeTrait::RegisterReflection(); });

TVM_REGISTER_REFLECTION_VTABLE(runtime::profiling::PercentNode);
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

TVM_FFI_STATIC_INIT_BLOCK({ RatioNodeTrait::RegisterReflection(); });

TVM_REGISTER_REFLECTION_VTABLE(runtime::profiling::RatioNode);
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<runtime::profiling::RatioNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const runtime::profiling::RatioNode*>(node.get());
      p->stream << op->GetTypeKey() << "(" << op->ratio << ")";
    });

}  // namespace tvm
