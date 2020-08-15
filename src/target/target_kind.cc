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
#include <tvm/target/target_kind.h>

#include <algorithm>

#include "../node/attr_registry.h"

namespace tvm {

TVM_REGISTER_NODE_TYPE(TargetKindNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TargetKindNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const TargetKindNode*>(node.get());
      p->stream << op->name;
    });

using TargetKindRegistry = AttrRegistry<TargetKindRegEntry, TargetKind>;

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

const TargetKind& TargetKind::Get(const String& target_kind_name) {
  const TargetKindRegEntry* reg = TargetKindRegistry::Global()->Get(target_kind_name);
  CHECK(reg != nullptr) << "ValueError: TargetKind \"" << target_kind_name
                        << "\" is not registered";
  return reg->kind_;
}

// TODO(@junrushao1994): remove some redundant attributes

TVM_REGISTER_TARGET_KIND("llvm")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<String>("runtime")
    .add_attr_option<String>("mcpu")
    .add_attr_option<Array<String>>("mattr")
    .add_attr_option<String>("mtriple")
    .add_attr_option<String>("mfloat-abi")
    .set_default_keys({"cpu"})
    .set_device_type(kDLCPU);

TVM_REGISTER_TARGET_KIND("c")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<String>("runtime")
    .set_default_keys({"cpu"})
    .set_device_type(kDLCPU);

TVM_REGISTER_TARGET_KIND("cuda")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<Integer>("max_num_threads", Integer(1024))
    .add_attr_option<Integer>("thread_warp_size", Integer(32))
    .add_attr_option<String>("mcpu")
    .set_default_keys({"cuda", "gpu"})
    .set_device_type(kDLGPU);

TVM_REGISTER_TARGET_KIND("nvptx")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<Integer>("max_num_threads", Integer(1024))
    .add_attr_option<Integer>("thread_warp_size", Integer(32))
    .add_attr_option<String>("mcpu")
    .set_default_keys({"cuda", "gpu"})
    .set_device_type(kDLGPU);

TVM_REGISTER_TARGET_KIND("rocm")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<Integer>("max_num_threads", Integer(256))
    .add_attr_option<Integer>("thread_warp_size", Integer(64))
    .set_default_keys({"rocm", "gpu"})
    .set_device_type(kDLROCM);

TVM_REGISTER_TARGET_KIND("opencl")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<Integer>("max_num_threads", Integer(256))
    .add_attr_option<Integer>("thread_warp_size")
    .set_default_keys({"opencl", "gpu"})
    .set_device_type(kDLOpenCL);

TVM_REGISTER_TARGET_KIND("metal")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<Integer>("max_num_threads", Integer(256))
    .set_default_keys({"metal", "gpu"})
    .set_device_type(kDLMetal);

TVM_REGISTER_TARGET_KIND("vulkan")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<Integer>("max_num_threads", Integer(256))
    .set_default_keys({"vulkan", "gpu"})
    .set_device_type(kDLVulkan);

TVM_REGISTER_TARGET_KIND("webgpu")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<Integer>("max_num_threads", Integer(256))
    .set_default_keys({"webgpu", "gpu"})
    .set_device_type(kDLWebGPU);

TVM_REGISTER_TARGET_KIND("sdaccel")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .set_default_keys({"sdaccel", "hls"})
    .set_device_type(kDLOpenCL);

TVM_REGISTER_TARGET_KIND("aocl")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .set_default_keys({"aocl", "hls"})
    .set_device_type(kDLAOCL);

TVM_REGISTER_TARGET_KIND("aocl_sw_emu")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .set_default_keys({"aocl", "hls"})
    .set_device_type(kDLAOCL);

TVM_REGISTER_TARGET_KIND("hexagon")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .set_default_keys({"hexagon"})
    .set_device_type(kDLHexagon);

TVM_REGISTER_TARGET_KIND("stackvm")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .set_device_type(kDLCPU);

TVM_REGISTER_TARGET_KIND("ext_dev")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .set_device_type(kDLExtDev);

TVM_REGISTER_TARGET_KIND("hybrid")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .set_device_type(kDLCPU);

}  // namespace tvm
