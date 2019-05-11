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
 * \file web_runtime.cc
 */
#include <sys/stat.h>
#include <fstream>

#include "../src/runtime/c_runtime_api.cc"
#include "../src/runtime/cpu_device_api.cc"
#include "../src/runtime/workspace_pool.cc"
#include "../src/runtime/module_util.cc"
#include "../src/runtime/system_lib_module.cc"
#include "../src/runtime/module.cc"
#include "../src/runtime/ndarray.cc"
#include "../src/runtime/registry.cc"
#include "../src/runtime/file_util.cc"
#include "../src/runtime/dso_module.cc"
#include "../src/runtime/rpc/rpc_session.cc"
#include "../src/runtime/rpc/rpc_event_impl.cc"
#include "../src/runtime/rpc/rpc_server_env.cc"
#include "../src/runtime/graph/graph_runtime.cc"
#include "../src/runtime/opengl/opengl_device_api.cc"
#include "../src/runtime/opengl/opengl_module.cc"

namespace tvm {
namespace contrib {

struct RPCEnv {
 public:
  RPCEnv() {
    base_ = "/rpc";
    mkdir(&base_[0], 0777);
  }
  // Get Path.
  std::string GetPath(const std::string& file_name) {
    return base_ + "/" + file_name;
  }

 private:
  std::string base_;
};

TVM_REGISTER_GLOBAL("tvm.rpc.server.workpath")
.set_body_typed<std::string(std::string)>([](std::string path) {
    static RPCEnv env;
    return env.GetPath(path);
  });

TVM_REGISTER_GLOBAL("tvm.rpc.server.load_module")
.set_body_typed<Module(std::string)>([](std::string path) {
    std::string file_name = "/rpc/" + path;
    LOG(INFO) << "Load module from " << file_name << " ...";
    return Module::LoadFromFile(file_name, "");
  });
}  // namespace contrib
}  // namespace tvm

// dummy parallel runtime
int TVMBackendParallelLaunch(
    FTVMParallelLambda flambda,
    void* cdata,
    int num_task) {
  TVMAPISetLastError("Parallel is not supported in Web runtime");
  return -1;
}

int TVMBackendParallelBarrier(int task_id, TVMParallelGroupEnv* penv) {
  return 0;
}
