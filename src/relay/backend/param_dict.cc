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
 *  Copyright (c) 2019 by Contributors
 * \file param_dict.cc
 * \brief Implementation and registration of parameter dictionary
 * serializing/deserializing functions.
 */
#include "param_dict.h"

#include <dmlc/memory_io.h>

#include <string>
#include <vector>
#include <utility>

namespace tvm {
namespace relay {

using namespace runtime;

TVM_REGISTER_GLOBAL("tvm.relay._save_param_dict")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    CHECK_EQ(args.size() % 2, 0u);
    // `args` is in the form "key, value, key, value, ..."
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
    uint64_t header = kTVMNDArrayListMagic, reserved = 0;
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

TVM_REGISTER_GLOBAL("tvm.relay._load_param_dict")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    std::string bytes = args[0];
    std::vector<std::string> names;
    dmlc::MemoryStringStream memstrm(&bytes);
    dmlc::Stream* strm = &memstrm;
    uint64_t header, reserved;
    CHECK(strm->Read(&header))
        << "Invalid parameters file format";
    CHECK(header == kTVMNDArrayListMagic)
        << "Invalid parameters file format";
    CHECK(strm->Read(&reserved))
        << "Invalid parameters file format";
    CHECK(strm->Read(&names))
        << "Invalid parameters file format";
    uint64_t sz;
    strm->Read(&sz, sizeof(sz));
    size_t size = static_cast<size_t>(sz);
    CHECK(size == names.size())
        << "Invalid parameters file format";
    tvm::Array<NamedNDArray> ret;
    for (size_t i = 0; i < size; ++i) {
      tvm::runtime::NDArray temp;
      temp.Load(strm);
      auto n = tvm::make_node<NamedNDArrayNode>();
      n->name = std::move(names[i]);
      n->array = temp;
      ret.push_back(NamedNDArray(n));
    }
    *rv = ret;
  });

TVM_REGISTER_NODE_TYPE(NamedNDArrayNode);

}  // namespace relay
}  // namespace tvm
