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
 * \file hexagon_session.cc
 */

#include <tvm/runtime/registry.h>

extern "C" {
#include <AEEStdDef.h>
#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <HAP_perf.h>
}

#include <tvm/runtime/logging.h>

#include <string>

#include "../../../rpc/rpc_channel.h"
#include "../../../rpc/rpc_endpoint.h"
#include "../../../rpc/rpc_session.h"
#include "../hexagon_rpc.h"

namespace tvm {
namespace runtime {
namespace hexagon {

class HexagonTransportChannel : public RPCChannel {
 public:
  explicit HexagonTransportChannel(const std::string& uri, int remote_stack_size_bytes,
                                   uint32_t receive_buf_size_bytes) {
    if (_handle != AEE_EUNKNOWN) return;

    enable_unsigned_pd(true);
    set_remote_stack_size(remote_stack_size_bytes);

    AEEResult rc = hexagon_rpc_open(uri.c_str(), &_handle);
    ICHECK(rc == AEE_SUCCESS) << "hexagon_rpc_open failed. URI: " << uri.c_str();

    rc = hexagon_rpc_init(_handle, receive_buf_size_bytes);
    ICHECK(rc == AEE_SUCCESS) << "hexagon_rpc_set_receive_buf_size failed. receive_buf_size_bytes: "
                              << receive_buf_size_bytes;
  }

  size_t Send(const void* data, size_t size) override {
    ICHECK(_handle != AEE_EUNKNOWN) << "RPC handle is not initialized.";
    AEEResult rc =
        hexagon_rpc_send(_handle, static_cast<const unsigned char*>(data), static_cast<int>(size));
    ICHECK(rc == AEE_SUCCESS) << "hexagon_rpc_send failed: " << rc;
    return size;
  }

  size_t Recv(void* data, size_t size) override {
    ICHECK(_handle != AEE_EUNKNOWN) << "RPC handle is not initialized.";
    int64_t written_size = 0;
    AEEResult rc = hexagon_rpc_receive(_handle, static_cast<unsigned char*>(data),
                                       static_cast<int>(size), &written_size);
    ICHECK(rc == AEE_SUCCESS) << "hexagon_rpc_receive failed: " << rc;
    return static_cast<size_t>(written_size);
  }

  AEEResult Close() {
    if (_handle == AEE_EUNKNOWN) return AEE_SUCCESS;
    return hexagon_rpc_close(_handle);
  }

 private:
  AEEResult set_remote_stack_size(int size) {
    remote_rpc_thread_params data;
    data.domain = CDSP_DOMAIN_ID;
    data.prio = -1;
    data.stack_size = size;
    AEEResult rc = remote_session_control(FASTRPC_THREAD_PARAMS, &data, sizeof(data));
    if (rc != AEE_SUCCESS) {
      LOG(ERROR) << "error setting remote stack size: " << std::hex << rc << '\n';
    }
    return rc;
  }

  AEEResult enable_unsigned_pd(bool enable) {
    remote_rpc_control_unsigned_module data;
    data.domain = CDSP_DOMAIN_ID;
    data.enable = static_cast<int>(enable);
    AEEResult rc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &data, sizeof(data));
    if (rc != AEE_SUCCESS) {
      LOG(ERROR) << "Error " << (enable ? "enabling" : "disabling") << " unsigned PD\n";
    }
    return rc;
  }

  remote_handle64 _handle = AEE_EUNKNOWN;
};

TVM_REGISTER_GLOBAL("tvm.contrib.hexagon.create_hexagon_session")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      ICHECK(args.size() >= 4) << args.size() << " is less than 4";

      std::string session_name = args[0];
      int remote_stack_size_bytes = args[1];
      // For simulator, the third parameter is sim_args, ignore it.
      int hexagon_rpc_receive_buf_size_bytes = args[3];
      HexagonTransportChannel* hexagon_channel =
          new HexagonTransportChannel(hexagon_rpc_URI CDSP_DOMAIN, remote_stack_size_bytes,
                                      static_cast<uint32_t>(hexagon_rpc_receive_buf_size_bytes));
      std::unique_ptr<RPCChannel> channel(hexagon_channel);
      auto ep = RPCEndpoint::Create(std::move(channel), session_name, "", nullptr);
      auto sess = CreateClientSession(ep);
      *rv = CreateRPCSessionModule(sess);
    });

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
