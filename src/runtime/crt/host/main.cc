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
 * \file main.cc
 * \brief main entry point for host subprocess-based CRT
 */
#include <inttypes.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/aot_executor_module.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/microtvm_rpc_server.h>
#include <unistd.h>

#include <iostream>

#include "crt_config.h"

#ifdef TVM_HOST_USE_GRAPH_EXECUTOR_MODULE
#include <tvm/runtime/crt/graph_executor_module.h>
#endif

extern "C" {

ssize_t MicroTVMWriteFunc(void* context, const uint8_t* data, size_t num_bytes) {
  ssize_t to_return = write(STDOUT_FILENO, data, num_bytes);
  fflush(stdout);
  fsync(STDOUT_FILENO);
  return to_return;
}
}

static char** g_argv = NULL;

int testonly_reset_server(TVMValue* args, int* type_codes, int num_args, TVMValue* out_ret_value,
                          int* out_ret_tcode, void* resource_handle) {
  execvp(g_argv[0], g_argv);
  perror("microTVM runtime: error restarting");
  return -1;
}

int main(int argc, char** argv) {
  g_argv = argv;
  TVMPlatformInitialize();
  microtvm_rpc_server_t rpc_server = MicroTVMRpcServerInit(&MicroTVMWriteFunc, nullptr);

#ifdef TVM_HOST_USE_GRAPH_EXECUTOR_MODULE
  CHECK_EQ(TVMGraphExecutorModule_Register(), kTvmErrorNoError,
           "failed to register GraphExecutor TVMModule");
#endif

  int error = TVMFuncRegisterGlobal("tvm.testing.reset_server",
                                    (TVMFunctionHandle)&testonly_reset_server, 0);
  if (error) {
    fprintf(
        stderr,
        "microTVM runtime: internal error (error#: %x) registering global packedfunc; exiting\n",
        error);
    return 2;
  }

  setbuf(stdin, NULL);
  setbuf(stdout, NULL);

  for (;;) {
    uint8_t c;
    int ret_code = read(STDIN_FILENO, &c, 1);
    if (ret_code < 0) {
      perror("microTVM runtime: read failed");
      return 2;
    } else if (ret_code == 0) {
      fprintf(stderr, "microTVM runtime: 0-length read, exiting!\n");
      return 2;
    }
    uint8_t* cursor = &c;
    size_t bytes_to_process = 1;
    while (bytes_to_process > 0) {
      tvm_crt_error_t err = MicroTVMRpcServerLoop(rpc_server, &cursor, &bytes_to_process);
      if (err == kTvmErrorPlatformShutdown) {
        break;
      } else if (err != kTvmErrorNoError) {
        char buf[1024];
        snprintf(buf, sizeof(buf), "microTVM runtime: MicroTVMRpcServerLoop error: %08x", err);
        perror(buf);
        return 2;
      }
    }
  }
  return 0;
}
