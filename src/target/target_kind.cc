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

namespace tvm {

TVM_REGISTER_NODE_TYPE(TargetKindNode);

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
 * \brief Extract a number from the string with the given prefix.
 * For example, when `str` is "sm_20" and `prefix` is "sm_".
 * This function first checks if `str` starts with `prefix`,
 * then return the integer 20 after the `prefix`
 * \param str The string to be extracted
 * \param prefix The prefix to be checked
 * \return An integer, the extracted number. -1 if the check fails
 */
static int ExtractIntWithPrefix(const std::string& str, const std::string& prefix) {
  if (str.substr(0, prefix.size()) != prefix) {
    return -1;
  }
  int result = 0;
  for (size_t i = prefix.size(); i < str.size(); ++i) {
    char c = str[i];
    if (!isdigit(c)) {
      return -1;
    }
    result = result * 10 + c - '0';
  }
  return result;
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
    const auto* str = (*iter).second.as<StringObj>();
    ICHECK(str != nullptr && GetRef<String>(str) == value)
        << "ValueError: Expects \"" << name << "\" to be \"" << value
        << "\", but gets: " << (*iter).second;
  }
}

/**********  Target kind attribute updaters  **********/

/*!
 * \brief Update the attributes in the LLVM NVPTX target.
 * \param attrs The original attributes
 * \return The updated attributes
 */
Map<String, ObjectRef> UpdateNVPTXAttrs(Map<String, ObjectRef> attrs) {
  CheckOrSetAttr(&attrs, "mtriple", "nvptx64-nvidia-cuda");
  // Update -mcpu=sm_xx
  int arch;
  if (attrs.count("mcpu")) {
    // If -mcpu has been specified, validate the correctness
    String mcpu = Downcast<String>(attrs.at("mcpu"));
    arch = ExtractIntWithPrefix(mcpu, "sm_");
    ICHECK(arch != -1) << "ValueError: NVPTX target gets an invalid CUDA arch: -mcpu=" << mcpu;
  } else {
    // Use the compute version of the first CUDA GPU instead
    TVMRetValue version;
    if (!DetectDeviceFlag({kDLCUDA, 0}, runtime::kComputeVersion, &version)) {
      LOG(WARNING) << "Unable to detect CUDA version, default to \"-mcpu=sm_20\" instead";
      arch = 20;
    } else {
      arch = std::stod(version.operator std::string()) * 10 + 0.1;
    }
    attrs.Set("mcpu", String("sm_") + std::to_string(arch));
  }
  return attrs;
}

/*!
 * \brief Update the attributes in the LLVM ROCm target.
 * \param attrs The original attributes
 * \return The updated attributes
 */
Map<String, ObjectRef> UpdateROCmAttrs(Map<String, ObjectRef> attrs) {
  CheckOrSetAttr(&attrs, "mtriple", "amdgcn-amd-amdhsa-hcc");
  // Update -mcpu=gfx
  int arch;
  if (attrs.count("mcpu")) {
    String mcpu = Downcast<String>(attrs.at("mcpu"));
    arch = ExtractIntWithPrefix(mcpu, "gfx");
    ICHECK(arch != -1) << "ValueError: ROCm target gets an invalid GFX version: -mcpu=" << mcpu;
  } else {
    TVMRetValue val;
    if (!DetectDeviceFlag({kDLROCM, 0}, runtime::kGcnArch, &val)) {
      LOG(WARNING) << "Unable to detect ROCm compute arch, default to \"-mcpu=gfx900\" instead";
      arch = 900;
    } else {
      arch = val.operator int();
    }
    attrs.Set("mcpu", String("gfx") + std::to_string(arch));
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
    if (attrs.count("mattr")) {
      mattr = Downcast<Array<String>>(attrs.at("mattr"));
    }
    mattr.push_back("-code-object-v3");
    attrs.Set("mattr", mattr);
  }
  return attrs;
}

/**********  Register Target kinds and attributes  **********/

TVM_REGISTER_TARGET_KIND("llvm", kDLCPU)
    .add_attr_option<Array<String>>("mattr")
    .add_attr_option<String>("mcpu")
    .add_attr_option<String>("mtriple")
    .add_attr_option<String>("mfloat-abi")
    .add_attr_option<String>("mabi")
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<String>("runtime")
    .add_attr_option<Bool>("link-params", Bool(false))
    .add_attr_option<Bool>("unpacked-api")
    .add_attr_option<String>("interface-api")
    .set_default_keys({"cpu"});

TVM_REGISTER_TARGET_KIND("c", kDLCPU)
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<Bool>("link-params", Bool(false))
    .add_attr_option<String>("runtime")
    .add_attr_option<String>("mcpu")
    .add_attr_option<String>("march")
    .add_attr_option<String>("executor")
    .add_attr_option<Integer>("workspace-byte-alignment")
    .add_attr_option<Bool>("unpacked-api")
    .add_attr_option<String>("interface-api")
    .set_default_keys({"cpu"});

TVM_REGISTER_TARGET_KIND("cuda", kDLCUDA)
    .add_attr_option<String>("mcpu")
    .add_attr_option<String>("arch")
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<Integer>("max_num_threads", Integer(1024))
    .add_attr_option<Integer>("thread_warp_size", Integer(32))
    .add_attr_option<Integer>("shared_memory_per_block")
    .add_attr_option<Integer>("registers_per_block")
    .add_attr_option<Integer>("max_threads_per_block")
    .set_default_keys({"cuda", "gpu"});

TVM_REGISTER_TARGET_KIND("nvptx", kDLCUDA)
    .add_attr_option<String>("mcpu")
    .add_attr_option<String>("mtriple")
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<Integer>("max_num_threads", Integer(1024))
    .add_attr_option<Integer>("thread_warp_size", Integer(32))
    .set_default_keys({"cuda", "gpu"})
    .set_attrs_preprocessor(UpdateNVPTXAttrs);

TVM_REGISTER_TARGET_KIND("rocm", kDLROCM)
    .add_attr_option<String>("mcpu")
    .add_attr_option<String>("mtriple")
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<Integer>("max_num_threads", Integer(256))
    .add_attr_option<Integer>("thread_warp_size", Integer(64))
    .set_default_keys({"rocm", "gpu"})
    .set_attrs_preprocessor(UpdateROCmAttrs);

TVM_REGISTER_TARGET_KIND("opencl", kDLOpenCL)
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<Integer>("max_num_threads", Integer(256))
    .add_attr_option<Integer>("thread_warp_size", Integer(1))
    .set_default_keys({"opencl", "gpu"});

// The metal has some limitations on the number of input parameters. This is why attribute
// `max_function_args` was introduced. It specifies the maximum number of kernel argumetns. More
// information about this limitation can be found here:
// https://developer.apple.com/documentation/metal/buffers/about_argument_buffers?language=objc
TVM_REGISTER_TARGET_KIND("metal", kDLMetal)
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<Integer>("max_num_threads", Integer(256))
    .add_attr_option<Integer>("thread_warp_size", Integer(16))
    .add_attr_option<Integer>("max_function_args", Integer(31))
    .set_default_keys({"metal", "gpu"});

TVM_REGISTER_TARGET_KIND("vulkan", kDLVulkan)
    .add_attr_option<Bool>("system-lib")
    // Feature support
    .add_attr_option<Bool>("supports_float16")
    .add_attr_option<Bool>("supports_float32", Bool(true))
    .add_attr_option<Bool>("supports_float64")
    .add_attr_option<Bool>("supports_int8")
    .add_attr_option<Bool>("supports_int16")
    .add_attr_option<Bool>("supports_int32", Bool(true))
    .add_attr_option<Bool>("supports_int64")
    .add_attr_option<Bool>("supports_8bit_buffer")
    .add_attr_option<Bool>("supports_16bit_buffer")
    .add_attr_option<Bool>("supports_storage_buffer_storage_class")
    .add_attr_option<Bool>("supports_push_descriptor")
    .add_attr_option<Bool>("supports_dedicated_allocation")
    .add_attr_option<Integer>("supported_subgroup_operations")
    // Physical device limits
    .add_attr_option<Integer>("max_num_threads", Integer(256))
    .add_attr_option<Integer>("thread_warp_size", Integer(1))
    .add_attr_option<Integer>("max_block_size_x")
    .add_attr_option<Integer>("max_block_size_y")
    .add_attr_option<Integer>("max_block_size_z")
    .add_attr_option<Integer>("max_push_constants_size")
    .add_attr_option<Integer>("max_uniform_buffer_range")
    .add_attr_option<Integer>("max_storage_buffer_range")
    .add_attr_option<Integer>("max_per_stage_descriptor_storage_buffer")
    .add_attr_option<Integer>("max_shared_memory_per_block")
    // Other device properties
    .add_attr_option<String>("device_type")
    .add_attr_option<String>("device_name")
    .add_attr_option<String>("driver_name")
    .add_attr_option<Integer>("driver_version")
    .add_attr_option<Integer>("vulkan_api_version")
    .add_attr_option<Integer>("max_spirv_version")
    // Tags
    .set_default_keys({"vulkan", "gpu"});

TVM_REGISTER_TARGET_KIND("webgpu", kDLWebGPU)
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<Integer>("max_num_threads", Integer(256))
    .set_default_keys({"webgpu", "gpu"});

TVM_REGISTER_TARGET_KIND("sdaccel", kDLOpenCL)
    .add_attr_option<Bool>("system-lib")
    .set_default_keys({"sdaccel", "hls"});

TVM_REGISTER_TARGET_KIND("aocl", kDLAOCL)
    .add_attr_option<Bool>("system-lib")
    .set_default_keys({"aocl", "hls"});

TVM_REGISTER_TARGET_KIND("aocl_sw_emu", kDLAOCL)
    .add_attr_option<Bool>("system-lib")
    .set_default_keys({"aocl", "hls"});

TVM_REGISTER_TARGET_KIND("hexagon", kDLHexagon)
    .add_attr_option<Array<String>>("mattr")
    .add_attr_option<String>("mcpu")
    .add_attr_option<String>("mtriple")
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<Array<String>>("llvm-options")
    .set_default_keys({"hexagon"});

TVM_REGISTER_TARGET_KIND("stackvm", kDLCPU)  // line break
    .add_attr_option<Bool>("system-lib");

TVM_REGISTER_TARGET_KIND("ext_dev", kDLExtDev)  // line break
    .add_attr_option<Bool>("system-lib");

TVM_REGISTER_TARGET_KIND("hybrid", kDLCPU)  // line break
    .add_attr_option<Bool>("system-lib");

TVM_REGISTER_TARGET_KIND("composite", kDLCPU).add_attr_option<Array<Target>>("devices");

/**********  Registry  **********/

TVM_REGISTER_GLOBAL("target.ListTargetKinds").set_body_typed(TargetKindRegEntry::ListTargetKinds);

}  // namespace tvm
