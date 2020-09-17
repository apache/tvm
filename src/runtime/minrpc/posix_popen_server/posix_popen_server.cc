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

// Disable constructor to bring minimum dep on c++ABI.
#define TVM_ARENA_HAS_DESTRUCTOR 0

#include <unistd.h>

#include <cstdlib>

#include "minrpc_server.h"

namespace tvm {
namespace runtime {

/*!
 * \brief IOHandler based on posix API.
 */
class PosixIOHandler {
 public:
  explicit PosixIOHandler(int read_fd = 0, int write_fd = 1)
      : read_fd_(read_fd), write_fd_(write_fd) {}

  void MessageStart(uint64_t packet_nbytes) {}

  void MessageDone() {}

  ssize_t PosixRead(void* data, size_t size) { return read(read_fd_, data, size); }

  ssize_t PosixWrite(const void* data, size_t size) { return write(write_fd_, data, size); }

  void Exit(int code) { exit(code); }

  void Close() {
    if (read_fd_ != 0) close(read_fd_);
    if (write_fd_ != 0) close(write_fd_);
  }

 private:
  int read_fd_{0};
  int write_fd_{1};
};

/*! \brief Type for the posix version of min rpc server. */
using PosixMinRPCServer = MinRPCServer<PosixIOHandler>;

}  // namespace runtime
}  // namespace tvm

int main(int argc, char* argv[]) {
  if (argc != 3) return -1;
  // pass the descriptor via arguments.
  tvm::runtime::PosixIOHandler handler(atoi(argv[1]), atoi(argv[2]));
  tvm::runtime::PosixMinRPCServer server(&handler);
  bool is_running = true;
  while (is_running) {
    is_running = server.ProcessOnePacket();
  }

  return 0;
}
