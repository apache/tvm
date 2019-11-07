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
 * \file src/runtime/contrib/extern_util.h
 * \brief The definition of utility function for the external runtime.
 */

#ifndef TVM_RUNTIME_CONTRIB_EXTERNAL_UTIL_H_
#define TVM_RUNTIME_CONTRIB_EXTERNAL_UTIL_H_

#include <string>
#include <vector>

namespace tvm {
namespace runtime {
namespace contrib {

/*!
 * \brief Split the encoded function name to tokens.
 *
 * \param the function name string.
 *
 * \return a vector of tokenized function name splitted by "_".
 */
static inline std::string GetSubgraphID(const std::string& name) {
  std::string temp = name;
  std::vector<std::string> tokens;
  std::string delimiter = "_";
  size_t pos = 0;
  std::string token;
  while ((pos = temp.find(delimiter)) != std::string::npos) {
    token = temp.substr(0, pos);
    tokens.push_back(token);
    temp.erase(0, pos + delimiter.length());
  }
  tokens.push_back(temp);

  CHECK(tokens.size() >= 2) << "Invalid subgraph name: " << name;
  CHECK(tokens[0] == "subgraph")
      << "Function name does not start with \"subgraph\": " << name;
  return tokens[1];
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_CONTRIB_EXTERNAL_UTIL_H_
