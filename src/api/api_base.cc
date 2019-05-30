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
 *  Implementation of basic API functions
 * \file api_base.cc
 */
#include <dmlc/memory_io.h>
#include <tvm/expr.h>
#include <tvm/tensor.h>
#include <tvm/api_registry.h>

namespace tvm {
TVM_REGISTER_API("_format_str")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    CHECK(args[0].type_code() == kNodeHandle);
    std::ostringstream os;
    os << args[0].operator NodeRef();
    *ret = os.str();
  });

TVM_REGISTER_API("_raw_ptr")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    CHECK(args[0].type_code() == kNodeHandle);
    *ret = reinterpret_cast<int64_t>(
        args[0].node_sptr().get());
  });

TVM_REGISTER_API("_save_json")
.set_body_typed<std::string(NodeRef)>(SaveJSON);

TVM_REGISTER_API("_load_json")
.set_body_typed<NodeRef(std::string)>(LoadJSON<NodeRef>);

TVM_REGISTER_API("_TVMSetStream")
.set_body_typed(TVMSetStream);

TVM_REGISTER_API("_save_param_dict")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    CHECK_EQ(args.size() % 2, 0u);
    constexpr uint64_t TVMNDArrayListMagic = 0xF7E58D4F05049CB7;
    size_t num_params = args.size() / 2;
    std::vector<std::string> names;
    names.reserve(num_params);
    std::vector<DLTensor*> arrays;
    arrays.reserve(num_params);
    for (size_t i = 0; i < num_params * 2; i += 2) {
      names.emplace_back(args[i].operator std::string());
      arrays.emplace_back(args[i + 1].operator DLTensor*());
    }
    std::string bytes;
    dmlc::MemoryStringStream strm(&bytes);
    dmlc::Stream* fo = &strm;
    uint64_t header = TVMNDArrayListMagic, reserved = 0;
    fo->Write(header);
    fo->Write(reserved);
    fo->Write(names);
    {
      uint64_t sz = static_cast<uint64_t>(arrays.size());
      fo->Write(sz);
      for (size_t i = 0; i < sz; ++i) {
        tvm::runtime::SaveDLTensor(fo, arrays[i]);
      }
    }
    TVMByteArray arr;
    arr.data = bytes.c_str();
    arr.size = bytes.length();
    *rv = arr;
  });

}  // namespace tvm
