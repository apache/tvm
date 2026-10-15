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
#ifndef TVM_BACKEND_CUDA_LAUNCH_LAUNCH_PLAN_H_
#define TVM_BACKEND_CUDA_LAUNCH_LAUNCH_PLAN_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/function.h>

#include <unordered_set>
#include <vector>

#include "launch_fields.h"

namespace tvm_cuda_launch {

struct Operand {
  Field field;
  int index;
  int64_t constant{0};
};

// Shared by runtime decoding and host source generation. Only this boundary
// understands historical launch tags; native configurations have one typed
// field per operand and no flag-only or last-operand conventions.
class PackedLaunchPlan {
 public:
  PackedLaunchPlan() = default;
  PackedLaunchPlan(const tvm::ffi::Array<tvm::ffi::String>& fields,
                   const tvm::ffi::Array<tvm::ffi::String>& legacy_tags,
                   const tvm::ffi::Map<tvm::ffi::String, int64_t>& kernel_attrs) {
    auto attribute = [&](const std::string& key) { return kernel_attrs.Get(key).value_or(0); };
    requirements.required_block_size = attribute("required_block_size") != 0;
    for (int axis = 0; axis != 3; ++axis) {
      requirements.block[axis] = attribute("required_block_" + std::string(1, 'x' + axis));
      requirements.cluster[axis] = attribute("required_cluster_" + std::string(1, 'x' + axis));
    }
    std::unordered_set<std::string> seen;
    if (!fields.empty()) {
      if (!legacy_tags.empty())
        throw std::invalid_argument("Cannot mix CUDA launch fields and legacy tags");
      for (const auto& field : fields) {
        if (!seen.insert(field).second)
          throw std::invalid_argument("Duplicate CUDA launch field: " + std::string(field));
        operands.push_back({FindField(field), static_cast<int>(num_values++)});
      }
      return;
    }
    legacy = true;
    for (size_t i = 0; i < legacy_tags.size(); ++i) {
      std::string tag = legacy_tags[i];
      if (!seen.insert(tag).second)
        throw std::invalid_argument("Duplicate CUDA launch tag: " + tag);
      if (tag == "tirx.use_programtic_dependent_launch") {
        operands.push_back({Field::kProgrammaticStreamSerialization, -1, 1});
      } else if (tag == "tirx.use_cooperative_launch") {
        operands.push_back({Field::kCooperative, -1, 1});
      } else if (tag == "tirx.use_required_block_dimension") {
        requirements.required_block_size = true;
      } else if (tag == "tirx.use_dyn_shared_memory") {
        if (i + 1 != legacy_tags.size())
          throw std::invalid_argument(
              "Legacy dynamic shared memory must be the last launch parameter");
        operands.push_back({Field::kDynamicSmemBytes, static_cast<int>(num_values++)});
      } else {
        static const char* old_prefix[] = {"blockIdx.", "threadIdx.", "clusterCtaIdx.",
                                           "preferredClusterCtaIdx."};
        static const char* new_prefix[] = {"grid.", "block.", "cluster.", "preferred_cluster."};
        bool matched = false;
        for (int rank = 0; rank != 4; ++rank) {
          for (char axis : {'x', 'y', 'z'}) {
            if (tag == std::string(old_prefix[rank]) + axis) {
              operands.push_back({FindField(std::string(new_prefix[rank]) + axis),
                                  static_cast<int>(num_values++)});
              matched = true;
            }
          }
        }
        if (!matched) throw std::invalid_argument("Unknown CUDA launch parameter: " + tag);
      }
    }
  }

  LaunchValues Extract(tvm::ffi::PackedArgs args, size_t kernel_args) const {
    if (static_cast<size_t>(args.size()) != kernel_args + num_values)
      throw std::invalid_argument(
          "CUDA kernel launch argument count does not match its configuration");
    LaunchValues values;
    for (const auto& operand : operands) {
      if (operand.index < 0) {
        values.SetInt(operand.field, operand.constant);
        continue;
      }
      auto arg = args[kernel_args + operand.index];
      switch (kFields[static_cast<size_t>(operand.field)].kind) {
        case ValueKind::kInteger:
          values.SetInt(operand.field, arg.cast<int64_t>());
          break;
        case ValueKind::kFloat:
          values.SetFloat(operand.field, arg.cast<double>());
          break;
        case ValueKind::kHandle:
          values.SetHandle(operand.field, arg.cast<void*>());
          break;
      }
    }
    NormalizeLegacyPresence(&values);
    return values;
  }

  void NormalizeLegacyPresence(LaunchValues* values) const {
    if (legacy) tvm_cuda_launch::NormalizeLegacyPresence(values);
  }

  std::vector<Operand> operands;
  size_t num_values{0};
  bool legacy{false};
  KernelRequirements requirements;
};
}  // namespace tvm_cuda_launch
#endif  // TVM_BACKEND_CUDA_LAUNCH_LAUNCH_PLAN_H_
