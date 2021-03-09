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
 * \file param_dict.cc
 * \brief Implementation and registration of parameter dictionary
 * serializing/deserializing functions.
 */
#include "param_dict.h"

#include <dmlc/memory_io.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <utility>
#include <vector>

#include "../../runtime/file_utils.h"

namespace tvm {
namespace relay {

using namespace runtime;

TVM_REGISTER_GLOBAL("tvm.relay._save_param_dict")
    .set_body_typed([](const Map<String, NDArray>& params) {
      std::string s = ::tvm::runtime::SaveParams(params);
      // copy return array so it is owned by the ret value
      TVMRetValue rv;
      rv = TVMByteArray{s.data(), s.size()};
      return rv;
    });
TVM_REGISTER_GLOBAL("tvm.relay._load_param_dict").set_body_typed([](const String& s) {
  return ::tvm::runtime::LoadParams(s);
});

}  // namespace relay
}  // namespace tvm
