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
  const String IdxInputBase(const MSCJoint& node, const String& prefix = "", int idx = 0,
                            const String& suffix = "", bool process = false) final {
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
  const String IdxOutputBase(const MSCJoint& node, const String& prefix = "", int idx = 0,
                             const String& suffix = "", bool mark_exit = false) final {
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
  const String IdxWeightBase(const MSCJoint& node, const String& wtype, const String& suffix = "",
                             bool process = false) final {
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
  void Load(dmlc::JSONReader* reader) {
    std::string key;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      if (key == "log_level") {
        reader->Read(&log_level);
      } else if (key == "profile_level") {
        reader->Read(&profile_level);
      } else if (key == "test_iter") {
        reader->Read(&test_iter);
      } else if (key == "max_workspace") {
        reader->Read(&max_workspace);
      } else if (key == "cmake_version") {
        reader->Read(&cmake_version);
      } else if (key == "dataset") {
        reader->Read(&dataset);
      } else if (key == "range_file") {
        reader->Read(&range_file);
      } else if (key == "precision") {
        reader->Read(&precision);
      } else if (key == "precision_mode") {
        reader->Read(&precision_mode);
      } else if (key == "tensorrt_root") {
        reader->Read(&tensorrt_root);
      } else if (key == "extern_libs") {
        reader->Read(&extern_libs);
      } else {
        CODEGEN_CONFIG_PARSE
      }
    }
  }
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_FRAMEWORK_TENSORRT_CODEGEN_UTILS_H_
