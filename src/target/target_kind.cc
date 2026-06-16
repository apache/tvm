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

#include "../ir/attr_registry.h"
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
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::TypeAttrDef<TargetKindNode>().def(
      refl::type_attr::kRepr,
      [](TargetKind kind, ffi::Function) -> ffi::String { return kind->name; });
}

/**********  Registry-related code  **********/

using TargetKindRegistry = AttrRegistry<TargetKindRegEntry, TargetKind>;

ffi::Array<ffi::String> TargetKindRegEntry::ListTargetKinds() {
  return TargetKindRegistry::Global()->ListAllNames();
}

ffi::Map<ffi::String, ffi::String> TargetKindRegEntry::ListTargetKindOptions(
    const TargetKind& target_kind) {
  ffi::Map<ffi::String, ffi::String> options;
  for (const auto& e : target_kind->schema_.ListOptions()) {
    options.Set(e.key, e.type_str);
  }
  return options;
}

TargetKindRegEntry& TargetKindRegEntry::RegisterOrGet(const ffi::String& target_kind_name) {
  return TargetKindRegistry::Global()->RegisterOrGet(target_kind_name);
}

void TargetKindRegEntry::UpdateAttr(const ffi::String& key, ffi::Any value, int plevel) {
  TargetKindRegistry::Global()->UpdateAttr(key, kind_, value, plevel);
}

const AttrRegistryMapContainerMap<TargetKind>& TargetKind::GetAttrMapContainer(
    const ffi::String& attr_name) {
  return TargetKindRegistry::Global()->GetAttrMap(attr_name);
}

ffi::Optional<TargetKind> TargetKind::Get(const ffi::String& target_kind_name) {
  const TargetKindRegEntry* reg = TargetKindRegistry::Global()->Get(target_kind_name);
  if (reg == nullptr) {
    return std::nullopt;
  }
  return reg->kind_;
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

TVM_REGISTER_TARGET_KIND("llvm", kDLCPU)
    .add_attr_option<ffi::Array<ffi::String>>("mattr")
    .add_attr_option<ffi::String>("mcpu")
    .add_attr_option<ffi::String>("mtriple")
    .add_attr_option<ffi::String>("mfloat-abi")
    .add_attr_option<ffi::String>("mabi")
    .add_attr_option<int64_t>("num-cores")
    // Fast math flags, see https://llvm.org/docs/LangRef.html#fast-math-flags
    .add_attr_option<bool>("fast-math")  // implies all the below
    .add_attr_option<bool>("fast-math-nnan")
    .add_attr_option<bool>("fast-math-ninf")
    .add_attr_option<bool>("fast-math-nsz")
    .add_attr_option<bool>("fast-math-arcp")
    .add_attr_option<bool>("fast-math-contract")
    .add_attr_option<bool>("fast-math-reassoc")
    .add_attr_option<int64_t>("opt-level")
    // LLVM command line flags, see below
    .add_attr_option<ffi::Array<ffi::String>>("cl-opt")
    // LLVM JIT engine mcjit/orcjit
    .add_attr_option<ffi::String>("jit")
    // TVM & LLVM custom vector bit width
    .add_attr_option<int64_t>("vector-width")
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

TVM_REGISTER_TARGET_KIND("c", kDLCPU)
    .add_attr_option<ffi::String>("mcpu")
    .add_attr_option<ffi::String>("march")
    .add_attr_option<int64_t>("workspace-byte-alignment")
    .add_attr_option<int64_t>("constants-byte-alignment")
    .set_default_keys({"cpu"})
    .set_target_canonicalizer(tvm::target::canonicalizer::llvm::Canonicalize);

TVM_REGISTER_TARGET_KIND("ext_dev", kDLExtDev);

TVM_REGISTER_TARGET_KIND("composite", kDLCPU)  // line break
    .add_attr_option<ffi::Array<Target>>(
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

TVM_REGISTER_TARGET_KIND("test", kDLCPU)  // line break
    .set_target_canonicalizer(TestTargetParser);

/**********  Registry  **********/

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.TargetKindGetAttr",
           [](TargetKind kind, ffi::String attr_name) -> ffi::Any {
             auto target_attr_map = TargetKind::GetAttrMap<ffi::Any>(attr_name);
             ffi::Any rv;
             if (target_attr_map.count(kind)) {
               rv = target_attr_map[kind];
             }
             return rv;
           })
      .def("target.ListTargetKinds", TargetKindRegEntry::ListTargetKinds)
      .def("target.ListTargetKindOptions", TargetKindRegEntry::ListTargetKindOptions)
      .def("target.ListTargetKindOptionsFromName", [](ffi::String target_kind_name) {
        TargetKind kind = TargetKind::Get(target_kind_name).value();
        return TargetKindRegEntry::ListTargetKindOptions(kind);
      });
}

}  // namespace tvm
