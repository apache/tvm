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
 * \file rpc_pipe_impl.cc
 * \brief Pipe-based RPC channel.
 */
// Linux only for now, as linux is the most common usecase.
#if defined(__linux__) || defined(__ANDROID__)

#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>

#include <tvm/runtime/registry.h>
#include <memory>
#include <cstdlib>

#include "rpc_endpoint.h"
#include "rpc_local_session.h"
#include "../../support/pipe.h"

namespace tvm {
namespace runtime {

class PipeChannel final : public RPCChannel {
 public:
  explicit PipeChannel(int readfd, int writefd, pid_t child_pid)
      : readfd_(readfd), writefd_(writefd), child_pid_(child_pid) {
  }

  ~PipeChannel() {
    Close();
  }

  size_t Send(const void* data, size_t size) final {
    ssize_t n = write(writefd_, data, size);
    if (n == -1) {
      LOG(FATAL) << "Pipe write error";
    }
    return static_cast<size_t>(n);
  }

  size_t Recv(void* data, size_t size) final {
    ssize_t n = read(readfd_, data, size);
    if (n == -1) {
      LOG(FATAL) << "Pipe read error";
    }
    return static_cast<size_t>(n);
  }

  void Close() {
    close(readfd_);
    close(writefd_);
    kill(child_pid_, SIGKILL);
  }

 private:
  int readfd_;
  int writefd_;
  pid_t child_pid_;
};


Module CreatePipeClient(std::vector<std::string> cmd) {
  int parent2child[2];
  int child2parent[2];
  CHECK_EQ(pipe(parent2child), 0);
  CHECK_EQ(pipe(child2parent), 0);

  int parent_read = child2parent[0];
  int parent_write = parent2child[1];
  int child_read = parent2child[0];
  int child_write = child2parent[1];

  pid_t pid = fork();
  if (pid == 0) {
    // child process
    close(parent_read);
    close(parent_write);
    std::string sread_pipe = std::to_string(child_read);
    std::string swrite_pipe = std::to_string(child_write);
    std::vector<char*> argv;
    for (auto& str : cmd) {
      argv.push_back(dmlc::BeginPtr(str));
    }
    argv.push_back(dmlc::BeginPtr(sread_pipe));
    argv.push_back(dmlc::BeginPtr(swrite_pipe));
    argv.push_back(nullptr);
    execvp(argv[0], &argv[0]);
  }
  // parent process
  close(child_read);
  close(child_write);

  auto endpt = RPCEndpoint::Create(
      std::unique_ptr<PipeChannel>(
          new PipeChannel(parent_read, parent_write, pid)),
      "pipe", "pipe");
  endpt->InitRemoteSession(TVMArgs(nullptr, nullptr, 0));
  return CreateRPCSessionModule(CreateClientSession(endpt));
}

TVM_REGISTER_GLOBAL("rpc.CreatePipeClient")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  std::vector<std::string> cmd;
  for (int i = 0; i < args.size(); ++i) {
    cmd.push_back(args[i].operator std::string());
  }
  *rv = CreatePipeClient(cmd);
});


}  // namespace runtime
}  // namespace tvm
#endif
