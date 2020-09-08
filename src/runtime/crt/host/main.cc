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
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/utvm_rpc_server.h>
#include <unistd.h>

#include <chrono>
#include <iostream>

#include "crt_config.h"

using namespace std::chrono;

extern "C" {

ssize_t UTvmWriteFunc(void* context, const uint8_t* data, size_t num_bytes) {
  ssize_t to_return = write(STDOUT_FILENO, data, num_bytes);
  fflush(stdout);
  fsync(STDOUT_FILENO);
  return to_return;
}

void TVMPlatformAbort(tvm_crt_error_t error_code) {
  std::cerr << "TVMPlatformAbort: " << error_code << std::endl;
  throw "Aborted";
}

high_resolution_clock::time_point g_utvm_start_time;
int g_utvm_timer_running = 0;

int TVMPlatformTimerStart() {
  if (g_utvm_timer_running) {
    std::cerr << "timer already running" << std::endl;
    return -1;
  }
  g_utvm_start_time = high_resolution_clock::now();
  g_utvm_timer_running = 1;
  return 0;
}

int TVMPlatformTimerStop(double* res_us) {
  if (!g_utvm_timer_running) {
    std::cerr << "timer not running" << std::endl;
    return -1;
  }
  auto utvm_stop_time = high_resolution_clock::now();
  duration<double, std::micro> time_span(utvm_stop_time - g_utvm_start_time);
  *res_us = time_span.count();
  g_utvm_timer_running = 0;
  return 0;
}
}

uint8_t memory[512 * 1024];

static char** g_argv = NULL;

int testonly_reset_server(TVMValue* args, int* type_codes, int num_args, TVMValue* out_ret_value,
                          int* out_ret_tcode, void* resource_handle) {
  execvp(g_argv[0], g_argv);
  perror("utvm runtime: error restarting");
  return -1;
}

int main(int argc, char** argv) {
  g_argv = argv;
  utvm_rpc_server_t rpc_server =
      UTvmRpcServerInit(memory, sizeof(memory), 8, &UTvmWriteFunc, nullptr);

  if (TVMFuncRegisterGlobal("tvm.testing.reset_server", (TVMFunctionHandle)&testonly_reset_server,
                            0)) {
    fprintf(stderr, "utvm runtime: internal error registering global packedfunc; exiting\n");
    return 2;
  }

  setbuf(stdin, NULL);
  setbuf(stdout, NULL);

  for (;;) {
    uint8_t c;
    int ret_code = read(STDIN_FILENO, &c, 1);
    if (ret_code < 0) {
      perror("utvm runtime: read failed");
      return 2;
    } else if (ret_code == 0) {
      fprintf(stderr, "utvm runtime: 0-length read, exiting!\n");
      return 2;
    }
    if (UTvmRpcServerReceiveByte(rpc_server, c) != 1) {
      abort();
    }
    if (!UTvmRpcServerLoop(rpc_server)) {
      execvp(argv[0], argv);
      perror("utvm runtime: error restarting");
      return 2;
    }
  }
  return 0;
}
