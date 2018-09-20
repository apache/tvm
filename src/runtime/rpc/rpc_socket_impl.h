/*!
 *  Copyright (c) 2018 by Contributors
 * \file rpc_socket_impl.h
 * \brief Socket based RPC implementation.
 */
#ifndef TVM_RUNTIME_RPC_RPC_SOCKET_IMPL_H_
#define TVM_RUNTIME_RPC_RPC_SOCKET_IMPL_H_

namespace tvm {
namespace runtime {

/*!
 * \brief RPCServerLoop Start the rpc server loop.
 * \param sockfd Socket file descriptor
 */
void RPCServerLoop(int sockfd);
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_RPC_RPC_SOCKET_IMPL_H_