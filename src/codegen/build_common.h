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
 *  Copyright (c) 2017 by Contributors
 *  Common build utilities
 * \file build_common.h
 */
#ifndef TVM_CODEGEN_BUILD_COMMON_H_
#define TVM_CODEGEN_BUILD_COMMON_H_

#include <tvm/codegen.h>
#include <unordered_map>
#include <string>
#include "../runtime/meta_data.h"

namespace tvm {
namespace codegen {
// Extract function information from device function.
inline std::unordered_map<std::string, runtime::FunctionInfo>
ExtractFuncInfo(const Array<LoweredFunc>& funcs) {
  std::unordered_map<std::string, runtime::FunctionInfo> fmap;
  for (LoweredFunc f : funcs) {
    runtime::FunctionInfo info;
    for (size_t i = 0; i < f->args.size(); ++i) {
      info.arg_types.push_back(Type2TVMType(f->args[i].type()));
    }
    for (size_t i = 0; i < f->thread_axis.size(); ++i) {
      info.thread_axis_tags.push_back(f->thread_axis[i]->thread_tag);
    }
    fmap[f->name] = info;
  }
  return fmap;
}
}  // namespace codegen
}  // namespace tvm
#endif  // TVM_CODEGEN_BUILD_COMMON_H_
