/*!
 *  Copyright (c) 2019 by Contributors
 * \file tcl_socket.h
 * \brief TCP socket wrapper for communicating using Tcl commands
 */
#ifndef TVM_RUNTIME_MICRO_TCL_SOCKET_H_
#define TVM_RUNTIME_MICRO_TCL_SOCKET_H_

#include "../../common/socket.h"

namespace tvm {
namespace runtime {

/*!
 * \brief TCP socket wrapper for communicating using Tcl commands
 */
class TclSocket {
 public:
  /*!
   * \brief constructor to create the socket
   */
  explicit TclSocket();

  /*!
   * \brief destructor to close the socket connection
   */
  ~TclSocket();

  /*!
   * \brief open connection with server
   * \param addr server address
   */
  void Connect(tvm::common::SockAddr addr);

  /*
   * \brief send command to the server and await a reply
   *
   * \return the reply
   */
  std::string SendCommand(std::string cmd);

 private:
  /*! \brief underlying TCP socket being wrapped */
  tvm::common::TCPSocket tcp_socket_;

  /*! \brief character denoting the end of a Tcl command */
  static const constexpr char kCommandTerminateToken = '\x1a';
  /*! \brief size of the buffer used to send messages (in bytes) */
  static const constexpr size_t kSendBufSize = 4096;
  /*! \brief size of the buffer used to receive messages (in bytes) */
  static const constexpr size_t kReplyBufSize = 4096;
};

}  // namespace runtime
}  // namespace tvm
#endif  //TVM_RUNTIME_MICRO_TCL_SOCKET_H_
