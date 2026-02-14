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
 * \file src/contrib/msc/framework/tensorrt/codegen_utils.h
 * \brief TensorRT config for codegen.
 */
#ifndef TVM_CONTRIB_MSC_FRAMEWORK_TENSORRT_CODEGEN_UTILS_H_
#define TVM_CONTRIB_MSC_FRAMEWORK_TENSORRT_CODEGEN_UTILS_H_

#include <string>
#include <vector>

#include "../../core/codegen/base_codegen.h"
#include "../../core/codegen/codegen_utils.h"

namespace tvm {
namespace contrib {
namespace msc {

/*!
 * \brief CodeGen helper for tensorrt codegen
 */
class TensorRTCodeGenHelper : public BaseCodeGenHelper {
 public:
  /*! \brief Get describe for default node input*/
  const ffi::String IdxInputBase(const MSCJoint& node, const ffi::String& prefix = "", int idx = 0,
                                 const ffi::String& suffix = "", bool process = false) final {
    const auto& pair = node->ProducerAndIdxOf(idx);
    if (pair.first->optype == "input") {
      return "*" + IdxNodeBase(pair.first, prefix, suffix);
    }
    if (pair.first->optype == "tuple" || pair.first->optype == "get_item") {
      return "*" + IdxNodeBase(pair.first, prefix, suffix);
    }
    return "*" + IdxOutputBase(pair.first, prefix, pair.second, suffix);
  }

  /*! \brief Get describe for default node output*/
  const ffi::String IdxOutputBase(const MSCJoint& node, const ffi::String& prefix = "", int idx = 0,
                                  const ffi::String& suffix = "", bool mark_exit = false) final {
    if (node->optype == "argmax" || node->optype == "argmin") {
      ICHECK_EQ(idx, 0) << "argmax and argmin only has 1 output, get " << idx;
      return IdxNodeBase(node, prefix, suffix) + "->getOutput(1)";
    }
    if (node->optype == "tuple") {
      return IdxNodeBase(node, prefix, suffix) + "[" + std::to_string(idx) + "]";
    }
    if (node->optype == "get_item") {
      ICHECK_EQ(idx, 0) << "get item only has 1 output, get " << idx;
      return IdxNodeBase(node, prefix, suffix);
    }
    return IdxNodeBase(node, prefix, suffix) + "->getOutput(" + std::to_string(idx) + ")";
  }

  /*! \brief Get describe for default node weight*/
  const ffi::String IdxWeightBase(const MSCJoint& node, const ffi::String& wtype,
                                  const ffi::String& suffix = "", bool process = false) final {
    return "mWeights[\"" + node->WeightAt(wtype)->name + "\"]";
  }
};

/*!
 * \brief CodeGen config for tensorrt codegen
 */
struct TensorRTCodeGenConfig {
  int log_level{0};
  int profile_level{0};
  int test_iter{0};
  size_t max_workspace{1 << 20};
  std::string cmake_version{"3.5"};
  std::string dataset{"Dataset"};
  std::string range_file{""};
  std::string precision{"float32"};
  std::string precision_mode{"strict"};
  std::string tensorrt_root{"/usr/local/cuda"};
  std::vector<std::string> extern_libs;
  CODEGEN_CONFIG_MEMBERS
  void Load(ffi::json::Object obj) {
    if (auto it = obj.find(ffi::String("log_level")); it != obj.end()) {
      log_level = static_cast<int>((*it).second.cast<int64_t>());
    }
    if (auto it = obj.find(ffi::String("profile_level")); it != obj.end()) {
      profile_level = static_cast<int>((*it).second.cast<int64_t>());
    }
    if (auto it = obj.find(ffi::String("test_iter")); it != obj.end()) {
      test_iter = static_cast<int>((*it).second.cast<int64_t>());
    }
    if (auto it = obj.find(ffi::String("max_workspace")); it != obj.end()) {
      max_workspace = static_cast<size_t>((*it).second.cast<int64_t>());
    }
    if (auto it = obj.find(ffi::String("cmake_version")); it != obj.end()) {
      cmake_version = std::string((*it).second.cast<ffi::String>());
    }
    if (auto it = obj.find(ffi::String("dataset")); it != obj.end()) {
      dataset = std::string((*it).second.cast<ffi::String>());
    }
    if (auto it = obj.find(ffi::String("range_file")); it != obj.end()) {
      range_file = std::string((*it).second.cast<ffi::String>());
    }
    if (auto it = obj.find(ffi::String("precision")); it != obj.end()) {
      precision = std::string((*it).second.cast<ffi::String>());
    }
    if (auto it = obj.find(ffi::String("precision_mode")); it != obj.end()) {
      precision_mode = std::string((*it).second.cast<ffi::String>());
    }
    if (auto it = obj.find(ffi::String("tensorrt_root")); it != obj.end()) {
      tensorrt_root = std::string((*it).second.cast<ffi::String>());
    }
    if (auto it = obj.find(ffi::String("extern_libs")); it != obj.end()) {
      auto arr = (*it).second.cast<::tvm::ffi::json::Array>();
      extern_libs.clear();
      extern_libs.reserve(arr.size());
      for (const auto& elem : arr) {
        extern_libs.push_back(std::string(elem.cast<ffi::String>()));
      }
    }
    CODEGEN_CONFIG_PARSE
  }
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_FRAMEWORK_TENSORRT_CODEGEN_UTILS_H_
