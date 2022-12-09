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
 * \file src/target/target_tag.cc
 * \brief Target tag registry
 */

#include <tvm/ir/expr.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/tag.h>
#include <tvm/target/target.h>

#include "../node/attr_registry.h"

namespace tvm {

TVM_REGISTER_NODE_TYPE(TargetTagNode);

TVM_REGISTER_GLOBAL("target.TargetTagListTags").set_body_typed(TargetTag::ListTags);
TVM_REGISTER_GLOBAL("target.TargetTagAddTag").set_body_typed(TargetTag::AddTag);

/**********  Registry-related code  **********/

using TargetTagRegistry = AttrRegistry<TargetTagRegEntry, TargetTag>;

TargetTagRegEntry& TargetTagRegEntry::RegisterOrGet(const String& target_tag_name) {
  return TargetTagRegistry::Global()->RegisterOrGet(target_tag_name);
}

Optional<Target> TargetTag::Get(const String& target_tag_name) {
  const TargetTagRegEntry* reg = TargetTagRegistry::Global()->Get(target_tag_name);
  if (reg == nullptr) {
    return NullOpt;
  }
  return Target(reg->tag_->config);
}

Map<String, Target> TargetTag::ListTags() {
  Map<String, Target> result;
  for (const String& tag : TargetTagRegistry::Global()->ListAllNames()) {
    result.Set(tag, TargetTag::Get(tag).value());
  }
  return result;
}

Target TargetTag::AddTag(String name, Map<String, ObjectRef> config, bool override) {
  TargetTagRegEntry& tag = TargetTagRegEntry::RegisterOrGet(name).set_name();
  ICHECK(override || tag.tag_->config.empty())
      << "Tag \"" << name << "\" has been previously defined as: " << tag.tag_->config;
  tag.set_config(config);
  return Target(config);
}

/**********  Register Target tags  **********/

TVM_REGISTER_TARGET_TAG("raspberry-pi/4b-aarch64")
    .set_config({{"kind", String("llvm")},
                 {"mtriple", String("aarch64-linux-gnu")},
                 {"mcpu", String("cortex-a72")},
                 {"mattr", Array<String>{"+neon"}},
                 {"num-cores", Integer(4)},
                 {"host", Map<String, ObjectRef>{{"kind", String("llvm")},
                                                 {"mtriple", String("aarch64-linux-gnu")},
                                                 {"mcpu", String("cortex-a72")},
                                                 {"mattr", Array<String>{"+neon"}},
                                                 {"num-cores", Integer(4)}}}});

TVM_REGISTER_TARGET_TAG("nvidia/jetson-agx-xavier")
    .set_config({{"kind", String("cuda")},
                 {"arch", String("sm_72")},
                 {"max_shared_memory_per_block", Integer(49152)},
                 {"max_threads_per_block", Integer(1024)},
                 {"thread_warp_size", Integer(32)},
                 {"registers_per_block", Integer(65536)},
                 {"host", Map<String, ObjectRef>{{"kind", String("llvm")},
                                                 {"mtriple", String("aarch64-linux-gnu")},
                                                 {"mcpu", String("carmel")},
                                                 {"num-cores", Integer(4)}}}});

#define TVM_REGISTER_CUDA_TAG(Name, Arch, SharedMem, RegPerBlock) \
  TVM_REGISTER_TARGET_TAG(Name).set_config({                      \
      {"kind", String("cuda")},                                   \
      {"keys", Array<String>{"cuda", "gpu"}},                     \
      {"arch", String(Arch)},                                     \
      {"max_shared_memory_per_block", Integer(SharedMem)},        \
      {"max_threads_per_block", Integer(1024)},                   \
      {"thread_warp_size", Integer(32)},                          \
      {"registers_per_block", Integer(RegPerBlock)},              \
  });

TVM_REGISTER_CUDA_TAG("nvidia/tesla-k80", "sm_37", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/tesla-k40", "sm_35", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/tesla-k20", "sm_35", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/tesla-c2075", "sm_20", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/tesla-c2050", "sm_20", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/tesla-c2070", "sm_20", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/nvidia-a100", "sm_80", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/nvidia-a40", "sm_86", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/nvidia-a30", "sm_80", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/nvidia-a10", "sm_86", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/nvidia-a16", "sm_86", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/nvidia-a2", "sm_86", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/nvidia-t4", "sm_75", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/nvidia-v100", "sm_70", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/tesla-p100", "sm_60", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/tesla-p40", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/tesla-p4", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/tesla-m60", "sm_52", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/tesla-m40", "sm_52", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/tesla-k80", "sm_37", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/tesla-k40", "sm_35", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/tesla-k20", "sm_35", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/tesla-k10", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-rtx-8000", "sm_75", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-rtx-6000", "sm_75", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-rtx-5000", "sm_75", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-rtx-4000", "sm_75", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-gv100", "sm_70", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-gp100", "sm_60", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-p6000", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-p5000", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-p4000", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-p2200", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-p2000", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-p1000", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-p620", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-p600", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-p400", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-m6000-24gb", "sm_52", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-m6000", "sm_52", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-k6000", "sm_35", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-m5000", "sm_52", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-k5200", "sm_35", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-k5000", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-m4000", "sm_52", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-k4200", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-k4000", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-m2000", "sm_52", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-k2200", "sm_50", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-k2000", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-k2000d", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-k1200", "sm_50", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-k620", "sm_50", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-k600", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-k420", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-410", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-plex-7000", "sm_20", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/rtx-5000", "sm_75", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/rtx-4000", "sm_75", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/rtx-3000", "sm_75", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/t2000", "sm_75", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/t1000", "sm_75", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/p620", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/p520", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-p5200", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-p4200", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-p3200", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-p5000", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-p4000", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-p3000", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-p2000", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-p1000", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-p600", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-p500", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-m5500m", "sm_52", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-m2200", "sm_52", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-m1200", "sm_50", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-m620", "sm_52", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-m520", "sm_50", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-k6000m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-k5200m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-k5100m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-m5000m", "sm_50", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-k500m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-k4200m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-k4100m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-m4000m", "sm_50", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-k3100m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-m3000m", "sm_50", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-k2200m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-k2100m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-m2000m", "sm_50", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-k1100m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-m1000m", "sm_50", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-k620m", "sm_50", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-k610m", "sm_35", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-m600m", "sm_50", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-k510m", "sm_35", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/quadro-m500m", "sm_50", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/nvidia-nvs-810", "sm_50", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/nvidia-nvs-510", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/nvidia-nvs-315", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/nvidia-nvs-310", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/nvs-5400m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/nvs-5200m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/nvs-4200m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-rtx-3090-ti", "sm_86", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-rtx-3090", "sm_86", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-rtx-3080-ti", "sm_86", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-rtx-3080", "sm_86", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-rtx-3070-ti", "sm_86", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-rtx-3070", "sm_86", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-rtx-3060", "sm_86", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/nvidia-titan-rtx", "sm_75", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-rtx-2080-ti", "sm_75", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-rtx-2080", "sm_75", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-rtx-2070", "sm_75", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-rtx-2060", "sm_75", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/nvidia-titan-v", "sm_70", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/nvidia-titan-xp", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/nvidia-titan-x", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-1080-ti", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-1080", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-1070-ti", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-1070", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-1060", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-1050", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-titan-x", "sm_52", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-titan-z", "sm_35", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-titan-black", "sm_35", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-titan", "sm_35", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-980-ti", "sm_52", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-980", "sm_52", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-970", "sm_52", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-960", "sm_52", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-950", "sm_52", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-780-ti", "sm_35", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-780", "sm_35", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-770", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-760", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-750-ti", "sm_50", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-750", "sm_50", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-690", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-680", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-670", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-660-ti", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-660", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-650-ti-boost", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-650-ti", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-650", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-560-ti", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-550-ti", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-460", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gts-450", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-590", "sm_20", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-580", "sm_20", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-570", "sm_20", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-480", "sm_20", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-470", "sm_20", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-465", "sm_20", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-740", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-730", "sm_35", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-730-ddr3,128bit", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-720", "sm_35", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-705", "sm_35", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-640-gddr5", "sm_35", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-640-gddr3", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-630", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-620", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-610", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-520", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-440", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-430", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-rtx-2080", "sm_75", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-rtx-2070", "sm_75", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-rtx-2060", "sm_75", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-1080", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-1070", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-1060", "sm_61", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-980", "sm_52", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-980m", "sm_52", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-970m", "sm_52", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-965m", "sm_52", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-960m", "sm_50", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-950m", "sm_50", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-940m", "sm_50", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-930m", "sm_50", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-920m", "sm_35", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-910m", "sm_52", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-880m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-870m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-860m-sm-30", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-860m-sm-50", "sm_50", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-850m", "sm_50", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-840m", "sm_50", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-830m", "sm_50", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-820m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-800m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-780m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-770m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-765m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-760m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-680mx", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-680m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-675mx", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-675m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-670mx", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-670m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-660m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-755m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-750m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-650m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-745m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-645m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-740m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-730m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-640m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-640m-le", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-735m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-635m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-730m", "sm_30", 49152, 65536);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-630m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-625m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-720m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-620m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-710m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-705m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-610m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-580m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-570m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-560m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-555m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-550m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-540m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-525m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-520mx", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-520m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-485m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-470m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-460m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-445m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-435m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-420m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gt-415m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-gtx-480m", "sm_20", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-710m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/geforce-410m", "sm_21", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/jetson-nano", "sm_53", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/jetson-tx2", "sm_62", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/jetson-tx1", "sm_53", 49152, 32768);
TVM_REGISTER_CUDA_TAG("nvidia/tegra-x1", "sm_53", 49152, 32768);

#undef TVM_REGISTER_CUDA_TAG

#define TVM_REGISTER_TAG_AWS_C5(Name, Cores, Arch)                                 \
  TVM_REGISTER_TARGET_TAG(Name).set_config({{"kind", String("llvm")},              \
                                            {"keys", Array<String>{"x86", "cpu"}}, \
                                            {"mcpu", String(Arch)},                \
                                            {"num-cores", Integer(Cores)}});

TVM_REGISTER_TAG_AWS_C5("aws/cpu/c5.large", 1, "skylake-avx512");
TVM_REGISTER_TAG_AWS_C5("aws/cpu/c5.xlarge", 2, "skylake-avx512");
TVM_REGISTER_TAG_AWS_C5("aws/cpu/c5.2xlarge", 4, "skylake-avx512");
TVM_REGISTER_TAG_AWS_C5("aws/cpu/c5.4xlarge", 8, "skylake-avx512");
TVM_REGISTER_TAG_AWS_C5("aws/cpu/c5.9xlarge", 18, "skylake-avx512");
TVM_REGISTER_TAG_AWS_C5("aws/cpu/c5.12xlarge", 24, "cascadelake");
TVM_REGISTER_TAG_AWS_C5("aws/cpu/c5.18xlarge", 36, "skylake-avx512");
TVM_REGISTER_TAG_AWS_C5("aws/cpu/c5.24xlarge", 48, "cascadelake");

#undef TVM_REGISTER_TAG_AWS_C5

}  // namespace tvm
