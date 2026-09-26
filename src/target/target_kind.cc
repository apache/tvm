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
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

#include <algorithm>

#include "../support/utils.h"
#include "./canonicalizer/llvm/canonicalize.h"

namespace tvm {

namespace refl = ffi::reflection;

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TargetKindNode::RegisterReflection();
  refl::TypeAttrDef<TargetKindNode>()
      .def("__data_to_json__",
           [](const TargetKindNode* node) {
             // simply save as the string
             return node->name;
           })
      .def("__data_from_json__", [](const ffi::String& name) {
        auto kind = TargetKind::Get(name);
        TVM_FFI_ICHECK(kind.has_value()) << "Cannot find target kind \'" << name << '\'';
        return kind.value();
      });
  refl::TypeAttrDef<TargetKindNode>().def(
      refl::type_attr::kRepr,
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
  TargetKind kind(ffi::make_object<TargetKindNode>());
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
  TargetKindDef("llvm")
      .set_default_device_type(kDLCPU)
      .def_option<ffi::Array<ffi::String>>("mattr")
      .def_option<ffi::String>("mcpu")
      .def_option<ffi::String>("mtriple")
      .def_option<ffi::String>("mfloat-abi")
      .def_option<ffi::String>("mabi")
      .def_option<int64_t>("num-cores")
      // Fast math flags, see https://llvm.org/docs/LangRef.html#fast-math-flags
      .def_option<bool>("fast-math")  // implies all the below
      .def_option<bool>("fast-math-nnan")
      .def_option<bool>("fast-math-ninf")
      .def_option<bool>("fast-math-nsz")
      .def_option<bool>("fast-math-arcp")
      .def_option<bool>("fast-math-contract")
      .def_option<bool>("fast-math-reassoc")
      .def_option<int64_t>("opt-level")
      // LLVM command line flags, see below
      .def_option<ffi::Array<ffi::String>>("cl-opt")
      // LLVM JIT engine mcjit/orcjit
      .def_option<ffi::String>("jit")
      // TVM & LLVM custom vector bit width
      .def_option<int64_t>("vector-width")
      .set_default_keys({"cpu"})
      // Force the external codegen kind attribute to be registered, even if no external
      // codegen targets are enabled by the TVM build.
      .set_target_canonicalizer(tvm::target::canonicalizer::llvm::Canonicalize);

  // Note regarding the "cl-opt" attribute:
  // Each string in the array has the format
  //   -optionname[[:type]=value]
  // where
  //   * optionname is the actual LLVM option (e.g. "unroll-threshold")
  //   * type is one of "bool", "int", "uint", or "string"
  //   * value is the corresponding option value (for "bool" type is can be 0 or "false"
  //     for false value, or 1 or "true" for true value)
  // If type is omitted, it is assumed to be "bool". If value is omitted, it is assumed
  // to be "true".
  //
  // The type must match the option type in LLVM. To find the type, search the LLVM
  // repository (https://github.com/llvm/llvm-project) for optionname, and look for
  // its definition: it will be a declaration of a variable of type cl::opt<T> with
  // optionname being an argument to the constructor. The T in the declaration is
  // the type.
  // For example, for unroll-threshold, we get the following declaration:
  // static cl::opt<unsigned>
  //     UnrollThreshold("unroll-threshold", cl::Hidden,
  //                     cl::desc("The cost threshold for loop unrolling"));
  // Hence the type is "uint".

  TargetKindDef("c")
      .set_default_device_type(kDLCPU)
      .def_option<ffi::String>("mcpu")
      .def_option<ffi::String>("march")
      .def_option<int64_t>("workspace-byte-alignment")
      .def_option<int64_t>("constants-byte-alignment")
      .set_default_keys({"cpu"})
      .set_target_canonicalizer(tvm::target::canonicalizer::llvm::Canonicalize);

  TargetKindDef("ext_dev")  // line break
      .set_default_device_type(kDLExtDev);

  TargetKindDef("composite")
      .set_default_device_type(kDLCPU)  // line break
      .def_option<ffi::Array<Target>>(
          "devices",
          ir::ConfigSchema::AttrValidator(ffi::TypedFunction<ffi::Any(ffi::Any)>(  //
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
