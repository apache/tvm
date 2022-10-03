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

#include <AEEStdDef.h>
#include <AEEStdErr.h>
#include <remote.h>
#include <rpcmem.h>

#include <algorithm>
#include <ios>
#include <iostream>
#include <string>
#include <vector>

#include "launcher_core.h"
#include "launcher_rpc.h"

AEEResult enable_unsigned_pd(bool enable) {
  remote_rpc_control_unsigned_module data;
  data.domain = CDSP_DOMAIN_ID;
  data.enable = static_cast<int>(enable);
  AEEResult rc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &data, sizeof(data));
  if (rc != AEE_SUCCESS) {
    std::cout << "error " << (enable ? "enabling" : "disabling") << " unsigned PD\n";
  }
  return rc;
}

AEEResult set_remote_stack_size(int size) {
  remote_rpc_thread_params data;
  data.domain = CDSP_DOMAIN_ID;
  data.prio = -1;
  data.stack_size = size;
  AEEResult rc = remote_session_control(FASTRPC_THREAD_PARAMS, &data, sizeof(data));
  if (rc != AEE_SUCCESS) {
    std::cout << "error setting remote stack size: " << std::hex << rc << '\n';
  }
  return rc;
}

struct RPCChannel : public ExecutionSession {
  explicit RPCChannel(const std::string& uri, bool gen_lwp_json = false)
      : ExecutionSession(gen_lwp_json) {
    enable_unsigned_pd(true);
    set_remote_stack_size(128 * 1024);

    int rc = launcher_rpc_open(uri.c_str(), &handle);
    if (rc != AEE_SUCCESS) {
      handle = -1;
    }
  }

  ~RPCChannel() {
    if (handle == -1) {
      return;
    }

    for (void* ptr : allocations) {
      rpcmem_free(ptr);
    }
    if (model_loaded) {
      unload_model();
    }
    launcher_rpc_close(handle);
    handle = -1;
  }

  void* alloc_mem(size_t nbytes, size_t align) override {
    void* host_ptr = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, nbytes);
    if (host_ptr != nullptr) {
      allocations.push_back(host_ptr);
    }
    return host_ptr;
  }

  void free_mem(void* addr) override {
    auto f = std::find(allocations.begin(), allocations.end(), addr);
    if (f != allocations.end()) {
      allocations.erase(f);
      rpcmem_free(addr);
    }
  }

  bool load_model(const std::string& model_path, const std::string& model_json) override {
    AEEResult rc = launcher_rpc_load(handle, model_path.c_str(), model_json.c_str());
    if (rc != AEE_SUCCESS) {
      std::cout << "error loading graph module: " << std::hex << rc << '\n';
    } else {
      model_loaded = true;
    }
    return rc == AEE_SUCCESS;
  }

  bool unload_model() override {
    AEEResult rc = launcher_rpc_unload(handle);
    if (rc != AEE_SUCCESS) {
      std::cout << "error unloading model: " << std::hex << rc << '\n';
    }
    model_loaded = false;
    return rc == AEE_SUCCESS;
  }

  bool set_input(int input_idx, const tensor_meta* input_meta, const void* input_data) override {
    AEEResult rc = launcher_rpc_set_input(
        handle, input_idx, reinterpret_cast<const unsigned char*>(input_meta),
        input_meta->meta_size(), reinterpret_cast<const unsigned char*>(input_data),
        input_meta->data_size());
    if (rc != AEE_SUCCESS) {
      std::cout << "error setting model input no." << input_idx << ": " << std::hex << rc << '\n';
    }
    return rc == AEE_SUCCESS;
  }

  bool run(uint64_t* pcycles, uint64_t* usecs) override {
    AEEResult rc = launcher_rpc_run(handle, pcycles, usecs, gen_lwp_json);
    if (rc != AEE_SUCCESS) {
      std::cout << "error running model: " << std::hex << rc << '\n';
    }
    return rc == AEE_SUCCESS;
  }

  bool get_num_outputs(int* num_outputs) override {
    AEEResult rc = launcher_rpc_get_num_outputs(handle, num_outputs);
    if (rc != AEE_SUCCESS) {
      std::cout << "error getting number of outputs: " << std::hex << rc << '\n';
    }
    return rc == AEE_SUCCESS;
  }

  bool get_output(int output_idx, tensor_meta* output_meta, int meta_size, void* output_data,
                  int data_size) override {
    AEEResult rc = launcher_rpc_get_output(
        handle, output_idx, reinterpret_cast<unsigned char*>(output_meta), meta_size,
        reinterpret_cast<unsigned char*>(output_data), data_size);
    if (rc != AEE_SUCCESS) {
      std::cout << "error getting output no." << output_idx << ": " << std::hex << rc << '\n';
    }
    return rc == AEE_SUCCESS;
  }

  bool model_loaded = false;
  remote_handle64 handle = -1;
  std::vector<void*> allocations;
};

ExecutionSession* create_execution_session(bool gen_lwp_json) {
  auto* session = new RPCChannel(launcher_rpc_URI CDSP_DOMAIN, gen_lwp_json);
  if (session->handle == -1) {
    delete session;
    session = nullptr;
    std::cout << "Error opening FastRPC channel\n";
  }
  return session;
}
