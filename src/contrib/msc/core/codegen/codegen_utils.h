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
 * \file src/contrib/msc/core/codegen/codegen_utils.h
 * \brief Common utilities for print.
 */
#ifndef TVM_CONTRIB_MSC_CORE_CODEGEN_CODEGEN_UTILS_H_
#define TVM_CONTRIB_MSC_CORE_CODEGEN_CODEGEN_UTILS_H_

#include <tvm/script/printer/doc.h>

#include <memory>
#include <string>
#include <vector>

#include "../ir/graph.h"
#include "../utils.h"

namespace tvm {
namespace contrib {
namespace msc {

using namespace tvm::script::printer;

#define CODEGEN_CONFIG_MEMBERS             \
  bool training{false};                    \
  bool use_tools{false};                   \
  bool use_plugin{false};                  \
  bool need_test{true};                    \
  std::string tools_scope{""};             \
  std::string tools_tag{"main"};           \
  std::string test_device{"cpu"};          \
  std::string prefix{"res_"};              \
  std::string baseline_folder{"baseline"}; \
  std::vector<size_t> version{0, 0, 0};

#define CODEGEN_CONFIG_PARSE                    \
  if (key == "training") {                      \
    reader->Read(&training);                    \
  } else if (key == "use_tools") {              \
    reader->Read(&use_tools);                   \
  } else if (key == "use_plugin") {             \
    reader->Read(&use_plugin);                  \
  } else if (key == "need_test") {              \
    reader->Read(&need_test);                   \
  } else if (key == "tools_scope") {            \
    reader->Read(&tools_scope);                 \
  } else if (key == "tools_tag") {              \
    reader->Read(&tools_tag);                   \
  } else if (key == "test_device") {            \
    reader->Read(&test_device);                 \
  } else if (key == "prefix") {                 \
    reader->Read(&prefix);                      \
  } else if (key == "version") {                \
    reader->Read(&version);                     \
  } else if (key == "baseline_folder") {        \
    reader->Read(&baseline_folder);             \
  } else {                                      \
    LOG(FATAL) << "Do not support key " << key; \
  }

#define CODEGEN_MEMBERS                                                                           \
 public:                                                                                          \
  virtual const String DType(const DataType& dtype) { return runtime::DLDataType2String(dtype); } \
                                                                                                  \
 protected:                                                                                       \
  const std::shared_ptr<ConfigType> config() { return config_; }                                  \
  const String IdxNodeBase(const MSCJoint& node) {                                                \
    return helper_.IdxNodeBase(node, config()->prefix, "");                                       \
  }                                                                                               \
  const String IdxInputBase(const MSCJoint& node, int idx = 0, bool process = true) {             \
    return helper_.IdxInputBase(node, config()->prefix, idx, "", process && config()->use_tools); \
  }                                                                                               \
  const String IdxOutputBase(const MSCJoint& node, int idx = 0, bool mark_exit = false) {         \
    return helper_.IdxOutputBase(node, config()->prefix, idx, "",                                 \
                                 mark_exit && config()->use_tools);                               \
  }                                                                                               \
  const String IdxWeightBase(const MSCJoint& node, const String& wtype, bool process = true) {    \
    return helper_.IdxWeightBase(node, wtype, "", process && config()->use_tools);                \
  }                                                                                               \
  const String Comment(const MSCJoint& node) { return helper_.Comment(node, config()->prefix); }  \
  int CompareVersion(size_t major, size_t minor, size_t patch) {                                  \
    return CommonUtils::CompareVersion(config()->version, {major, minor, patch});                 \
  }                                                                                               \
                                                                                                  \
 private:                                                                                         \
  std::shared_ptr<ConfigType> config_;                                                            \
  HelperType helper_;

/*!
 * \brief Utils for CodeGen.
 */
class CodeGenUtils {
 public:
  /*!
   * \brief Get indexed node string.
   * \return The String.
   */
  TVM_DLL static const String IdxNode(const MSCJoint& node, const String& prefix,
                                      const String& suffix = "");

  /*!
   * \brief Get indexed output string.
   * \return The String.
   */
  TVM_DLL static const String IdxOutput(const MSCJoint& node, const String& prefix, int idx = 0,
                                        const String& suffix = "");

  /*!
   * \brief Get indexed input string.
   * \return The String.
   */
  TVM_DLL static const String IdxInput(const MSCJoint& node, const String& prefix, int idx = 0,
                                       const String& suffix = "");

  /*!
   * \brief Get indexed weight string.
   * \return The String.
   */
  TVM_DLL static const String IdxWeight(const MSCJoint& node, const String& wtype,
                                        const String& suffix = "");

  /*!
   * \brief Get comment of a node.
   * \return The String.
   */
  TVM_DLL static const String CommentNode(const MSCJoint& node, const String& prefix);
};

/*!
 * \brief Basic CodeGenHelper
 */
class BaseCodeGenHelper {
 public:
  const String GetSuffix(const MSCJoint& node, bool process = false) {
    return process ? "c" + std::to_string(node->index) : "";
  }

  virtual const String IdxNodeBase(const MSCJoint& node, const String& prefix = "",
                                   const String& suffix = "") {
    return CodeGenUtils::IdxNode(node, prefix, suffix);
  }
  virtual const String IdxInputBase(const MSCJoint& node, const String& prefix = "", int idx = 0,
                                    const String& suffix = "", bool process = false) {
    const auto& pair = node->ProducerAndIdxOf(idx);
    size_t output_size = pair.first->outputs.size();
    if (process && (output_size > 1 || pair.first->optype == "tuple")) {
      return CodeGenUtils::IdxNode(pair.first, prefix, suffix) + "_" + std::to_string(pair.second);
    }
    return CodeGenUtils::IdxInput(node, prefix, idx, suffix + GetSuffix(node, process));
  }
  virtual const String IdxOutputBase(const MSCJoint& node, const String& prefix = "", int idx = 0,
                                     const String& suffix = "", bool mark_exit = false) {
    if (mark_exit) {
      if (node->outputs.size() > 1 || node->optype == "tuple") {
        return CodeGenUtils::IdxNode(node, prefix, suffix) + "_" + std::to_string(idx) + "_exit";
      }
      return CodeGenUtils::IdxOutput(node, prefix, idx, suffix + "_exit");
    }
    return CodeGenUtils::IdxOutput(node, prefix, idx, suffix);
  }
  virtual const String IdxWeightBase(const MSCJoint& node, const String& wtype,
                                     const String& suffix = "", bool process = false) {
    return CodeGenUtils::IdxWeight(node, wtype, suffix + GetSuffix(node, process));
  }
  virtual const String Comment(const MSCJoint& node, const String& prefix = "") {
    return CodeGenUtils::CommentNode(node, prefix);
  }
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_CORE_CODEGEN_CODEGEN_UTILS_H_
