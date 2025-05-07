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
 * \file rpc_server_env.cc
 * \brief Server environment of the RPC.
 */
#include <tvm/runtime/registry.h>

#include "../file_utils.h"

namespace tvm {
namespace runtime {

std::string RPCGetPath(const std::string& name) {
  // do live lookup everytime as workpath can change.
  const auto f = tvm::ffi::Function::GetGlobal("tvm.rpc.server.workpath");
  ICHECK(f.has_value()) << "require tvm.rpc.server.workpath";
  return (*f)(name).cast<std::string>();
}

TVM_REGISTER_GLOBAL("tvm.rpc.server.upload")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      std::string file_name = RPCGetPath(args[0].cast<std::string>());
      auto data = args[1].cast<std::string>();
      SaveBinaryToFile(file_name, data);
    });

TVM_REGISTER_GLOBAL("tvm.rpc.server.download")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      std::string file_name = RPCGetPath(args[0].cast<std::string>());
      std::string data;
      LoadBinaryFromFile(file_name, &data);
      LOG(INFO) << "Download " << file_name << "... nbytes=" << data.size();
      *rv = ffi::Bytes(data);
    });

TVM_REGISTER_GLOBAL("tvm.rpc.server.remove")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      std::string file_name = RPCGetPath(args[0].cast<std::string>());
      RemoveFile(file_name);
    });

}  // namespace runtime
}  // namespace tvm
