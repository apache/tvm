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
 * \file source_utils.cc
 */
#include "source_utils.h"

namespace tvm {
namespace runtime {

std::unordered_map<std::string, std::string> SplitKernels(std::string source,
                                                          std::string delimiter) {
  std::unordered_map<std::string, std::string> split_kernels;
  if (source.size()) {
    size_t begin = source.find(delimiter);
    size_t end = begin;
    while (end != std::string::npos) {
      begin += delimiter.size();
      end = source.find('\n', begin);
      std::string func_name = source.substr(begin, end - begin);
      begin = ++end;
      end = source.find(delimiter, begin);
      std::string func_source =
          source.substr(begin, (end == std::string::npos) ? end : end - begin);
      split_kernels.insert({func_name, func_source});
      begin = end;
    }
  }
  return split_kernels;
}
}  // namespace runtime
}  // namespace tvm
