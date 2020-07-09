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
 * \file src/relay/backend/contrib/acl/acl_api.cc
 * \brief A common JSON interface between relay and the ACL runtime module.
 */

#include "acl_api.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace acl {

std::pair<JSONSubGraph, std::vector<runtime::NDArray>> DeserializeSubgraph(
    std::string* serialized_function) {
  dmlc::MemoryStringStream mstrm(serialized_function);
  dmlc::Stream* strm = &mstrm;
  std::string serialized_json;
  strm->Read(&serialized_json);
  std::istringstream is(serialized_json);
  dmlc::JSONReader reader(&is);
  JSONSubGraph function;
  function.Load(&reader);
  std::vector<runtime::NDArray> constants;
  size_t const_count;
  strm->Read(&const_count);
  for (size_t i = 0; i < const_count; i++) {
    runtime::NDArray temp;
    temp.Load(strm);
    constants.push_back(temp);
  }
  return std::make_pair(function, constants);
}

std::string SerializeSubgraph(const JSONSubGraph& subgraph,
                              const std::vector<runtime::NDArray>& constants) {
  std::ostringstream os;
  dmlc::JSONWriter writer(&os);
  subgraph.Save(&writer);
  std::string serialized_subgraph;
  dmlc::MemoryStringStream mstrm(&serialized_subgraph);
  dmlc::Stream* strm = &mstrm;
  strm->Write(os.str());
  strm->Write(constants.size());
  for (const auto& it : constants) {
    it.Save(strm);
  }
  return serialized_subgraph;
}

}  // namespace acl
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
