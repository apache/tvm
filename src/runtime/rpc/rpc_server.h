/*!
 *  Copyright (c) 2018 by Contributors
 * \file rpc_server.h
 * \brief RPC Server implementation.
 */
#ifndef TVM_RUNTIME_RPC_RPC_SERVER_H_
#define TVM_RUNTIME_RPC_RPC_SERVER_H_

#include "tvm/runtime/c_runtime_api.h"

namespace tvm {
namespace runtime {

/*!
 * \brief RPCServerCreate Creates the RPC Server.
 * \param host The hostname of the server, Default=0.0.0.0
 * \param port The port of the RPC, Default=9090
 * \param port_end The end search port of the RPC, Default=9199
 * \param tracker The address of RPC tracker in host:port format e.g. 10.77.1.234:9190 Default=""
 * \param key The key used to identify the device type in tracker. Default=""
 * \param custom_addr Custom IP Address to Report to RPC Tracker. Default=""
 * \param silent Whether run in silent mode. Default=True
 * \param isProxy Whether to run in proxy mode. Default=False
 */
TVM_DLL void RPCServerCreate(std::string host="",
                             int port=9090,
                             int port_end=9099,
                             std::string tracker_addr="",
                             std::string key="",
                             std::string custom_addr="",
                             bool silent=true,
                             bool is_proxy=false);
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_RPC_RPC_SERVER_H_