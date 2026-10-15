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
 * \file src/target/target_kind.cc
 * \brief Target kind registry
 */
#include <dlpack/dlpack.h>
#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/extra/serialization.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

#include <algorithm>

namespace tvm {

namespace refl = ffi::reflection;

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TargetKindObj::RegisterReflection();
  refl::TypeAttrDef<TargetKindObj>()
      .def(tvm::ffi::type_attr::kDataToJson.name,
           [](const TargetKindObj* node) {
             // simply save as the string
             return node->name;
           })
      .def(tvm::ffi::type_attr::kDataFromJson.name, [](const ffi::String& name) {
        auto kind = TargetKind::Get(name);
        TVM_FFI_ICHECK(kind.has_value()) << "Cannot find target kind \'" << name << '\'';
        return kind.value();
      });
  refl::TypeAttrDef<TargetKindObj>().def(
      ffi::type_attr::kRepr.name,
      [](TargetKind kind, ffi::Function) -> ffi::String { return kind->name; });
}
/**********  Registry-related code  **********/

TargetKindRegistry* TargetKindRegistry::Global() {
  static TargetKindRegistry registry;
  return &registry;
}

TargetKind TargetKindRegistry::RegisterOrGet(const ffi::String& name) {
  if (auto it = kinds_.find(name); it != kinds_.end()) {
    return (*it).second;
  }
  TargetKind kind(ffi::make_object<TargetKindObj>());
  kind->name = name;
  kind->default_device_type = kDLCPU;
  kind->schema_.def_option<ffi::String>("kind")
      .def_option<ffi::Array<ffi::String>>("keys")
      .def_option<ffi::String>("tag")
      .def_option<ffi::String>("device")
      .def_option<ffi::String>("model")
      .def_option<ffi::Array<ffi::String>>("libs")
      .def_option<Target>("host")
      .def_option<int64_t>("from_device")
      .def_option<int64_t>("target_device_type");
  kinds_.Set(name, kind);
  return kind;
}

ffi::Optional<TargetKind> TargetKindRegistry::Get(const ffi::String& name) {
  if (auto it = kinds_.find(name); it != kinds_.end()) {
    return (*it).second;
  }
  return std::nullopt;
}

ffi::Array<ffi::String> TargetKindRegistry::ListTargetKinds() {
  ffi::Array<ffi::String> names;
  for (const auto& entry : kinds_) {
    names.push_back(entry.first);
  }
  return names;
}

ffi::Map<ffi::String, ffi::String> TargetKindRegistry::ListTargetKindOptions(
    const TargetKind& target_kind) {
  ffi::Map<ffi::String, ffi::String> options;
  for (const auto& e : target_kind->schema_.ListOptions()) {
    options.Set(e.key, e.type_str);
  }
  return options;
}

ffi::Optional<TargetKind> TargetKind::Get(const ffi::String& target_kind_name) {
  return TargetKindRegistry::Global()->Get(target_kind_name);
}

/*!
 * \brief Test Target Parser
 * \param target The Target to update
 * \return The updated attributes
 */
ffi::Map<ffi::String, ffi::Any> TestTargetParser(ffi::Map<ffi::String, ffi::Any> target) {
  target.Set("feature.is_test", true);
  return target;
}

/**********  Register Target kinds and attributes  **********/

TVM_FFI_STATIC_INIT_BLOCK() {
  TargetKindDef("ext_dev")  // line break
      .set_default_device_type(kDLExtDev);

  TargetKindDef("composite")
      .set_default_device_type(kDLCPU)  // line break
      .def_option<ffi::Array<Target>>(
          "devices",
          ConfigSchema::AttrValidator(ffi::TypedFunction<ffi::Any(ffi::Any)>(  //
              [](ffi::Any val) -> ffi::Any {
                // Allow elements to be strings or dicts, converting them to Target objects.
                if (val.try_cast<ffi::Array<Target>>().has_value()) return val;
                auto arr = val.cast<ffi::Array<ffi::Any>>();
                ffi::Array<Target> result;
                for (const auto& elem : arr) {
                  if (auto t = elem.try_cast<Target>()) {
                    result.push_back(t.value());
                  } else if (auto s = elem.try_cast<ffi::String>()) {
                    result.push_back(Target(s.value()));
                  } else if (auto m = elem.try_cast<ffi::Map<ffi::String, ffi::Any>>()) {
                    result.push_back(Target(m.value()));
                  } else {
                    TVM_FFI_THROW(TypeError)
                        << "Expected Target, string, or dict in 'devices' array, got '"
                        << elem.GetTypeKey() << "'";
                  }
                }
                return ffi::Any(result);
              })));

  TargetKindDef("test")
      .set_default_device_type(kDLCPU)  // line break
      .set_target_canonicalizer(TestTargetParser);
}

/**********  Registry  **********/

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.ListTargetKinds", [] { return TargetKindRegistry::Global()->ListTargetKinds(); })
      .def(
          "target.ListTargetKindOptions",
          [](TargetKind kind) { return TargetKindRegistry::Global()->ListTargetKindOptions(kind); })
      .def("target.ListTargetKindOptionsFromName", [](ffi::String target_kind_name) {
        TargetKind kind = TargetKind::Get(target_kind_name).value();
        return TargetKindRegistry::Global()->ListTargetKindOptions(kind);
      });
}

}  // namespace tvm
