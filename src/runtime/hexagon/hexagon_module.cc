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
 * \file hexagon_module.cc
 * \brief The HexagonHostModuleNode
 */
#include "hexagon_module.h"

#include <dmlc/memory_io.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <utility>
#include <vector>

#include "../file_utils.h"

namespace tvm {
namespace runtime {

HexagonHostModuleNode::HexagonHostModuleNode(std::string data, std::string fmt,
                                             std::unordered_map<std::string, FunctionInfo> fmap,
                                             std::string asm_str, std::string obj_str,
                                             std::string ir_str, std::string bc_str,
                                             const std::set<std::string>& packed_c_abi)
    : data_(data),
      fmt_(fmt),
      fmap_(fmap),
      asm_(asm_str),
      obj_(obj_str),
      ir_(ir_str),
      bc_(bc_str),
      packed_c_abi_funcs_(packed_c_abi) {}

PackedFunc HexagonHostModuleNode::GetFunction(const std::string& name,
                                              const ObjectPtr<Object>& sptr_to_self) {
  LOG(FATAL) << "HexagonHostModuleNode::GetFunction is not implemented.";
  return PackedFunc();
}

std::string HexagonHostModuleNode::GetSource(const std::string& format) {
  if (format == "s" || format == "asm") {
    return asm_;
  }
  if (format == "ll") {
    return ir_;
  }
  return "";
}

void HexagonHostModuleNode::SaveToFile(const std::string& file_name, const std::string& format) {
  std::string fmt = runtime::GetFileFormat(file_name, format);
  if (fmt == "so" || fmt == "dll" || fmt == "hexagon") {
    std::string meta_file = GetMetaFilePath(file_name);
    SaveMetaDataToFile(meta_file, fmap_);
#if !defined(__APPLE__)
    std::string c = "cp " + data_ + " " + file_name;
    ICHECK(std::system(c.c_str()) == 0) << "Cannot create " + file_name;
#endif
  } else if (fmt == "s" || fmt == "asm") {
    ICHECK(!asm_.empty()) << "Assembler source not available";
    SaveBinaryToFile(file_name, asm_);
  } else if (fmt == "o" || fmt == "obj") {
    ICHECK(!obj_.empty()) << "Object data not available";
    SaveBinaryToFile(file_name, obj_);
  } else if (fmt == "ll") {
    ICHECK(!ir_.empty()) << "LLVM IR source not available";
    SaveBinaryToFile(file_name, ir_);
  } else if (fmt == "bc") {
    ICHECK(!bc_.empty()) << "LLVM IR bitcode not available";
    SaveBinaryToFile(file_name, bc_);
  } else {
    LOG(FATAL) << "HexagonHostModuleNode::SaveToFile: unhandled format `" << fmt << "'";
  }
}

void HexagonHostModuleNode::SaveToBinary(dmlc::Stream* stream) {
  stream->Write(fmt_);
  stream->Write(fmap_);
  stream->Write(data_);
}

}  // namespace runtime
}  // namespace tvm
