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
#include <tvm/ir/expr.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

#include <algorithm>

#include "../node/attr_registry.h"
#include "../support/utils.h"
#include "./parsers/cpu.h"

namespace tvm {

// helper to get internal dev function in objectref.
struct TargetKind2ObjectPtr : public ObjectRef {
  static ObjectPtr<Object> Get(const TargetKind& kind) { return GetDataPtr<Object>(kind); }
};

TVM_REGISTER_NODE_TYPE(TargetKindNode)
    .set_creator([](const std::string& name) {
      auto kind = TargetKind::Get(name);
      ICHECK(kind.defined()) << "Cannot find target kind \'" << name << '\'';
      return TargetKind2ObjectPtr::Get(kind.value());
    })
    .set_repr_bytes([](const Object* n) -> std::string {
      return static_cast<const TargetKindNode*>(n)->name;
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TargetKindNode>([](const ObjectRef& obj, ReprPrinter* p) {
      const TargetKind& kind = Downcast<TargetKind>(obj);
      p->stream << kind->name;
    });

/**********  Registry-related code  **********/

using TargetKindRegistry = AttrRegistry<TargetKindRegEntry, TargetKind>;

Array<String> TargetKindRegEntry::ListTargetKinds() {
  return TargetKindRegistry::Global()->ListAllNames();
}

Map<String, String> TargetKindRegEntry::ListTargetKindOptions(const TargetKind& target_kind) {
  Map<String, String> options;
  for (const auto& kv : target_kind->key2vtype_) {
    options.Set(kv.first, kv.second.type_key);
  }
  return options;
}

TargetKindRegEntry& TargetKindRegEntry::RegisterOrGet(const String& target_kind_name) {
  return TargetKindRegistry::Global()->RegisterOrGet(target_kind_name);
}

void TargetKindRegEntry::UpdateAttr(const String& key, TVMRetValue value, int plevel) {
  TargetKindRegistry::Global()->UpdateAttr(key, kind_, value, plevel);
}

const AttrRegistryMapContainerMap<TargetKind>& TargetKind::GetAttrMapContainer(
    const String& attr_name) {
  return TargetKindRegistry::Global()->GetAttrMap(attr_name);
}

Optional<TargetKind> TargetKind::Get(const String& target_kind_name) {
  const TargetKindRegEntry* reg = TargetKindRegistry::Global()->Get(target_kind_name);
  if (reg == nullptr) {
    return NullOpt;
  }
  return reg->kind_;
}

/**********  Utility functions  **********/

/*!
 * \brief Extract a string from the string with the given prefix.
 * For example, when `str` is "sm_20" and `prefix` is "sm_".
 * This function first checks if `str` starts with `prefix`,
 * then return the integer 20 after the `prefix`
 * \param str The string to be extracted
 * \param prefix The prefix to be checked
 * \return A string, the extracted string. "" if the check fails
 */
std::string ExtractStringWithPrefix(const std::string& str, const std::string& prefix) {
  if (str.find(prefix) != 0) return "";
  std::size_t pos = prefix.length();
  while (pos < str.length() && (std::isdigit(str[pos]) || std::isalpha(str[pos]))) {
    ++pos;
  }
  return str.substr(prefix.length(), pos - prefix.length());
}

/*!
 * \brief Using TVM DeviceAPI to detect the device flag
 * \param device The device to be detected
 * \param flag The device flag to be detected
 * \param val The detected value
 * \return A boolean indicating if detection succeeds
 */
static bool DetectDeviceFlag(Device device, runtime::DeviceAttrKind flag, TVMRetValue* val) {
  using runtime::DeviceAPI;
  DeviceAPI* api = DeviceAPI::Get(device, true);
  // Check if compiled with the corresponding device api
  if (api == nullptr) {
    return false;
  }
  // Check if the device exists
  api->GetAttr(device, runtime::kExist, val);
  int exists = *val;
  if (!exists) {
    return false;
  }
  // Get the arch of the device
  DeviceAPI::Get(device)->GetAttr(device, flag, val);
  return true;
}

void CheckOrSetAttr(Map<String, ObjectRef>* attrs, const String& name, const String& value) {
  auto iter = attrs->find(name);
  if (iter == attrs->end()) {
    attrs->Set(name, value);
  } else {
    auto str = (*iter).second.as<String>();
    ICHECK(str && str.value() == value) << "ValueError: Expects \"" << name << "\" to be \""
                                        << value << "\", but gets: " << (*iter).second;
  }
}

/**********  Target kind attribute updaters  **********/

/*!
 * \brief Update the attributes in the CUDA target.
 * \param target The Target to update
 * \return The updated attributes
 */
TargetJSON UpdateCUDAAttrs(TargetJSON target) {
  // Update -arch=sm_xx
  if (target.count("arch")) {
    // If -arch has been specified, validate the correctness
    String archStr = Downcast<String>(target.at("arch"));
    ICHECK(support::StartsWith(archStr, "sm_"))
        << "ValueError: CUDA target gets an invalid CUDA arch: -arch=" << archStr;
  } else {
    // Use the compute version of the first CUDA GPU instead
    int archInt;
    TVMRetValue version;
    if (!DetectDeviceFlag({kDLCUDA, 0}, runtime::kComputeVersion, &version)) {
      LOG(WARNING) << "Unable to detect CUDA version, default to \"-arch=sm_50\" instead";
      archInt = 50;
    } else {
      archInt = std::stod(version.operator std::string()) * 10 + 0.1;
    }
    target.Set("arch", String("sm_") + std::to_string(archInt));
  }
  return target;
}

/*!
 * \brief Update the attributes in the LLVM NVPTX target.
 * \param target The Target to update
 * \return The updated attributes
 */
TargetJSON UpdateNVPTXAttrs(TargetJSON target) {
  CheckOrSetAttr(&target, "mtriple", "nvptx64-nvidia-cuda");
  // Update -mcpu=sm_xx
  if (target.count("mcpu")) {
    // If -mcpu has been specified, validate the correctness
    String mcpu = Downcast<String>(target.at("mcpu"));
    ICHECK(support::StartsWith(mcpu, "sm_"))
        << "ValueError: NVPTX target gets an invalid CUDA arch: -mcpu=" << mcpu;
  } else {
    // Use the compute version of the first CUDA GPU instead
    int arch;
    TVMRetValue version;
    if (!DetectDeviceFlag({kDLCUDA, 0}, runtime::kComputeVersion, &version)) {
      LOG(WARNING) << "Unable to detect CUDA version, default to \"-mcpu=sm_50\" instead";
      arch = 50;
    } else {
      arch = std::stod(version.operator std::string()) * 10 + 0.1;
    }
    target.Set("mcpu", String("sm_") + std::to_string(arch));
  }
  return target;
}

/*!
 * \brief Update the attributes in the LLVM ROCm target.
 * \param target The Target to update
 * \return The updated attributes
 */
TargetJSON UpdateROCmAttrs(TargetJSON target) {
  using tvm::runtime::Registry;
  CheckOrSetAttr(&target, "mtriple", "amdgcn-amd-amdhsa-hcc");
  // Update -mcpu=gfx
  std::string arch = "gfx900";
  if (target.count("mcpu")) {
    String mcpu = Downcast<String>(target.at("mcpu"));
    arch = ExtractStringWithPrefix(mcpu, "gfx");
    ICHECK(!arch.empty()) << "ValueError: ROCm target gets an invalid GFX version: -mcpu=" << mcpu;
  } else {
    TVMRetValue val;
    if (const auto* f_get_rocm_arch = Registry::Get("tvm_callback_rocm_get_arch")) {
      arch = (*f_get_rocm_arch)().operator std::string();
    }
    target.Set("mcpu", String(arch));
  }
  // Update -mattr before ROCm 3.5:
  //   Before ROCm 3.5 we needed code object v2, starting
  //   with 3.5 we need v3 (this argument disables v3)

  TVMRetValue val;
  int version;
  if (!DetectDeviceFlag({kDLROCM, 0}, runtime::kApiVersion, &val)) {
    LOG(WARNING) << "Unable to detect ROCm version, assuming >= 3.5";
    version = 305;
  } else {
    version = val.operator int();
  }
  if (version < 305) {
    Array<String> mattr;
    if (target.count("mattr")) {
      mattr = Downcast<Array<String>>(target.at("mattr"));
    }
    mattr.push_back("-code-object-v3");
    target.Set("mattr", mattr);
  }
  return target;
}

/*!
 * \brief Test Target Parser
 * \param target The Target to update
 * \return The updated attributes
 */
TargetJSON TestTargetParser(TargetJSON target) {
  Map<String, ObjectRef> features = {{"is_test", runtime::Bool(true)}};
  target.Set("features", features);
  return target;
}

/**********  Register Target kinds and attributes  **********/

TVM_REGISTER_TARGET_KIND("llvm", kDLCPU)
    .add_attr_option<Array<String>>("mattr")
    .add_attr_option<String>("mcpu")
    .add_attr_option<String>("mtriple")
    .add_attr_option<String>("mfloat-abi")
    .add_attr_option<String>("mabi")
    .add_attr_option<runtime::Int>("num-cores")
    // Fast math flags, see https://llvm.org/docs/LangRef.html#fast-math-flags
    .add_attr_option<runtime::Bool>("fast-math")  // implies all the below
    .add_attr_option<runtime::Bool>("fast-math-nnan")
    .add_attr_option<runtime::Bool>("fast-math-ninf")
    .add_attr_option<runtime::Bool>("fast-math-nsz")
    .add_attr_option<runtime::Bool>("fast-math-arcp")
    .add_attr_option<runtime::Bool>("fast-math-contract")
    .add_attr_option<runtime::Bool>("fast-math-reassoc")
    .add_attr_option<runtime::Int>("opt-level")
    // LLVM command line flags, see below
    .add_attr_option<Array<String>>("cl-opt")
    // LLVM JIT engine mcjit/orcjit
    .add_attr_option<String>("jit")
    .set_default_keys({"cpu"})
    // Force the external codegen kind attribute to be registered, even if no external
    // codegen targets are enabled by the TVM build.
    .set_attr<runtime::Bool>(tvm::attr::kIsExternalCodegen, runtime::Bool(false))
    .set_target_parser(tvm::target::parsers::cpu::ParseTarget);

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
    .add_attr_option<String>("mcpu")
    .add_attr_option<String>("march")
    .add_attr_option<runtime::Int>("workspace-byte-alignment")
    .add_attr_option<runtime::Int>("constants-byte-alignment")
    .set_default_keys({"cpu"})
    .set_target_parser(tvm::target::parsers::cpu::ParseTarget);

TVM_REGISTER_TARGET_KIND("cuda", kDLCUDA)
    .add_attr_option<String>("mcpu")
    .add_attr_option<String>("arch")
    .add_attr_option<runtime::Int>("max_shared_memory_per_block")
    .add_attr_option<runtime::Int>("max_threads_per_block")
    .add_attr_option<runtime::Int>("thread_warp_size", runtime::Int(32))
    .add_attr_option<runtime::Int>("registers_per_block")
    .add_attr_option<runtime::Int>("l2_cache_size_bytes")
    .add_attr_option<runtime::Int>("max_num_threads",
                                   runtime::Int(1024))  // TODO(@zxybazh): deprecate it
    .set_default_keys({"cuda", "gpu"})
    .set_target_parser(UpdateCUDAAttrs);

TVM_REGISTER_TARGET_KIND("nvptx", kDLCUDA)
    .add_attr_option<String>("mcpu")
    .add_attr_option<String>("mtriple")
    .add_attr_option<runtime::Int>("max_num_threads", runtime::Int(1024))
    .add_attr_option<runtime::Int>("thread_warp_size", runtime::Int(32))
    .set_default_keys({"cuda", "gpu"})
    .set_target_parser(UpdateNVPTXAttrs);

TVM_REGISTER_TARGET_KIND("rocm", kDLROCM)
    .add_attr_option<String>("mcpu")
    .add_attr_option<String>("mtriple")
    .add_attr_option<Array<String>>("mattr")
    // TODO(masahi): Support querying from a target device
    // On RDNA cards, thread_warp_size should be 32
    .add_attr_option<runtime::Int>("max_num_threads", runtime::Int(256))
    .add_attr_option<runtime::Int>("max_threads_per_block", runtime::Int(256))
    .add_attr_option<runtime::Int>("max_shared_memory_per_block", runtime::Int(65536))
    .add_attr_option<runtime::Int>("thread_warp_size", runtime::Int(64))
    .set_default_keys({"rocm", "gpu"})
    .set_target_parser(UpdateROCmAttrs);

TVM_REGISTER_TARGET_KIND("opencl", kDLOpenCL)
    .add_attr_option<runtime::Int>("max_threads_per_block", runtime::Int(256))
    .add_attr_option<runtime::Int>("max_shared_memory_per_block", runtime::Int(16384))
    .add_attr_option<runtime::Int>("max_num_threads", runtime::Int(256))
    .add_attr_option<runtime::Int>("thread_warp_size", runtime::Int(1))
    .add_attr_option<runtime::Int>("texture_spatial_limit", runtime::Int(16384))
    // Faced that Qualcomm OpenCL runtime crashed without any error message in
    // the case when the number of kernel arguments was pretty big. OpenCL doesn't
    // specify any limitations on the number of kernel arguments. max_function_args
    // equals to 128 looks like a reasonable number of kernel arguments.
    .add_attr_option<runtime::Int>("max_function_args", runtime::Int(128))
    .set_default_keys({"opencl", "gpu"});

// The metal has some limitations on the number of input parameters. This is why attribute
// `max_function_args` was introduced. It specifies the maximum number of kernel argumetns. More
// information about this limitation can be found here:
// https://developer.apple.com/documentation/metal/buffers/about_argument_buffers?language=objc
// See also https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
TVM_REGISTER_TARGET_KIND("metal", kDLMetal)
    .add_attr_option<runtime::Int>("max_num_threads", runtime::Int(256))
    .add_attr_option<runtime::Int>("max_threads_per_block", runtime::Int(256))
    .add_attr_option<runtime::Int>("max_shared_memory_per_block", runtime::Int(32768))
    .add_attr_option<runtime::Int>("thread_warp_size", runtime::Int(16))
    .add_attr_option<runtime::Int>("max_function_args", runtime::Int(31))
    .set_default_keys({"metal", "gpu"});

TVM_REGISTER_TARGET_KIND("vulkan", kDLVulkan)
    .add_attr_option<Array<String>>("mattr")
    // Feature support
    .add_attr_option<runtime::Bool>("supports_float16")
    .add_attr_option<runtime::Bool>("supports_float32", runtime::Bool(true))
    .add_attr_option<runtime::Bool>("supports_float64")
    .add_attr_option<runtime::Bool>("supports_int8")
    .add_attr_option<runtime::Bool>("supports_int16")
    .add_attr_option<runtime::Bool>("supports_int32", runtime::Bool(true))
    .add_attr_option<runtime::Bool>("supports_int64")
    .add_attr_option<runtime::Bool>("supports_8bit_buffer")
    .add_attr_option<runtime::Bool>("supports_16bit_buffer")
    .add_attr_option<runtime::Bool>("supports_storage_buffer_storage_class")
    .add_attr_option<runtime::Bool>("supports_push_descriptor")
    .add_attr_option<runtime::Bool>("supports_dedicated_allocation")
    .add_attr_option<runtime::Bool>("supports_integer_dot_product")
    .add_attr_option<runtime::Bool>("supports_cooperative_matrix")
    .add_attr_option<runtime::Int>("supported_subgroup_operations")
    // Physical device limits
    .add_attr_option<runtime::Int>("max_num_threads", runtime::Int(256))
    .add_attr_option<runtime::Int>("max_threads_per_block", runtime::Int(256))
    .add_attr_option<runtime::Int>("thread_warp_size", runtime::Int(1))
    .add_attr_option<runtime::Int>("max_block_size_x")
    .add_attr_option<runtime::Int>("max_block_size_y")
    .add_attr_option<runtime::Int>("max_block_size_z")
    .add_attr_option<runtime::Int>("max_push_constants_size")
    .add_attr_option<runtime::Int>("max_uniform_buffer_range")
    .add_attr_option<runtime::Int>("max_storage_buffer_range")
    .add_attr_option<runtime::Int>("max_per_stage_descriptor_storage_buffer")
    .add_attr_option<runtime::Int>("max_shared_memory_per_block")
    // Other device properties
    .add_attr_option<String>("device_type")
    .add_attr_option<String>("device_name")
    .add_attr_option<String>("driver_name")
    .add_attr_option<runtime::Int>("driver_version")
    .add_attr_option<runtime::Int>("vulkan_api_version")
    .add_attr_option<runtime::Int>("max_spirv_version")
    // Tags
    .set_default_keys({"vulkan", "gpu"});

TVM_REGISTER_TARGET_KIND("webgpu", kDLWebGPU)
    .add_attr_option<runtime::Int>("max_num_threads", runtime::Int(256))
    .set_default_keys({"webgpu", "gpu"});

TVM_REGISTER_TARGET_KIND("sdaccel", kDLOpenCL)  // line break
    .set_default_keys({"sdaccel", "hls"});

TVM_REGISTER_TARGET_KIND("aocl", kDLAOCL)  // line break
    .set_default_keys({"aocl", "hls"});

TVM_REGISTER_TARGET_KIND("aocl_sw_emu", kDLAOCL)  // line break
    .set_default_keys({"aocl", "hls"});

TVM_REGISTER_TARGET_KIND("hexagon", kDLHexagon)
    .add_attr_option<Array<String>>("mattr")
    .add_attr_option<String>("mcpu")
    .add_attr_option<String>("mtriple")
    .add_attr_option<Array<String>>("llvm-options")
    .add_attr_option<runtime::Int>("num-cores")
    .add_attr_option<runtime::Int>("vtcm-capacity")
    .set_default_keys({"hexagon", "cpu"});

TVM_REGISTER_TARGET_KIND("stackvm", kDLCPU)  // line break
    .set_default_keys({"cpu"});

TVM_REGISTER_TARGET_KIND("ext_dev", kDLExtDev);

TVM_REGISTER_TARGET_KIND("hybrid", kDLCPU);

TVM_REGISTER_TARGET_KIND("composite", kDLCPU)  // line break
    .add_attr_option<Array<Target>>("devices");

TVM_REGISTER_TARGET_KIND("test", kDLCPU)  // line break
    .set_target_parser(TestTargetParser);

/**********  Registry  **********/

TVM_REGISTER_GLOBAL("target.TargetKindGetAttr")
    .set_body_typed([](TargetKind kind, String attr_name) -> TVMRetValue {
      auto target_attr_map = TargetKind::GetAttrMap<TVMRetValue>(attr_name);
      TVMRetValue rv;
      if (target_attr_map.count(kind)) {
        rv = target_attr_map[kind];
      }
      return rv;
    });
TVM_REGISTER_GLOBAL("target.ListTargetKinds").set_body_typed(TargetKindRegEntry::ListTargetKinds);
TVM_REGISTER_GLOBAL("target.ListTargetKindOptions")
    .set_body_typed(TargetKindRegEntry::ListTargetKindOptions);
TVM_REGISTER_GLOBAL("target.ListTargetKindOptionsFromName")
    .set_body_typed([](String target_kind_name) {
      TargetKind kind = TargetKind::Get(target_kind_name).value();
      return TargetKindRegEntry::ListTargetKindOptions(kind);
    });

}  // namespace tvm
