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

#include "src/standalone_crt/include/tvm/runtime/crt/microtvm_rpc_server.h"
#include "src/standalone_crt/include/tvm/runtime/crt/logging.h"
#include "src/model.h"
microtvm_rpc_server_t server;

// Called by TVM to write serial data to the UART.
ssize_t write_serial(void* unused_context, const uint8_t* data, size_t size) {
  Serial.write(data, size);
  return size;
}

void setup() {
  server = MicroTVMRpcServerInit(write_serial, NULL);
  TVMLogf("microTVM Arduino runtime - running");
  Serial.begin(115200);
}

void loop() {
  int to_read = Serial.available();
  uint8_t data[to_read];
  size_t bytes_read = Serial.readBytes(data, to_read);
  uint8_t* arr_ptr = data;
  uint8_t** data_ptr = &arr_ptr;
  if (bytes_read > 0) {
    size_t bytes_remaining = bytes_read;
    while (bytes_remaining > 0) {
      // Pass the received bytes to the RPC server.
      tvm_crt_error_t err = MicroTVMRpcServerLoop(server, data_ptr, &bytes_remaining);
      if (err != kTvmErrorNoError && err != kTvmErrorFramingShortPacket) {
        TVMPlatformAbort(err);
      }
    }
  }
}
