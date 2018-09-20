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
 * \brief RPCServerCreate.
 * \param host
 * \param port
 * \param port_end
 * \param tracker_addr
 * \param key
 * \param custom_addr
 * \param silent
 * \param isProxy
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