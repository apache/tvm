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

#ifndef TVM_RUNTIME_HEXAGON_RPC_SIMULATOR_HEXAGON_SIM_PROTO_H_
#define TVM_RUNTIME_HEXAGON_RPC_SIMULATOR_HEXAGON_SIM_PROTO_H_

struct Message {
  enum : uint32_t {
    kNone = 0,
    kAck,
    kTerminate,
    kReceiveStart,
    kReceiveEnd,
    kSendStart,
    kSendEnd,
  };
  enum : uint32_t {
    null_va = 0,
  };

  uint32_t code;
  uint32_t len;
  uint32_t va;
} __attribute__((packed));

// Protocol:
//
// Copying data from host to remote:
//
// Host >-- [ kReceiveStart, len,      null_va ] --> Remote
// * Remote client prepares a buffer with at least `len` bytes.
// Host <-- [ kAck,          buf_size, buf_ptr ] <-- Remote
// * Host writes `nbytes` into buffer, `nbytes` <= `len`.
// Host >-- [ kReceiveEnd,   nbytes,   buf_ptr ] --> Remote
// * Remote client processes the data.
// Host <-- [ kAck,          ___,      ___     ] <-- Remote
//
// Copying data from remote to host:
//
// Host >-- [ kSendStart,    len,      null_va ] --> Remote
// * Remote client returns pointer to the buffer with the data to be read.
// * There should be at least `len` bytes ready in the buffer.
// Host <-- [ kAck,          buf_size, buf_ptr ] <-- Remote
// * Host reads `nbytes` from buffer, `nbytes` <= `buf_size`.
// Host >-- [ kSendEnd   ,   nbytes,   buf_ptr ] --> Remote
// * Remote client processes the data.
// Host <-- [ kAck,          ___,      ___     ] <-- Remote
//
// Teminating server:
//
// Host >-- [ kTerminate,    ___,      ___     ] --> Remote
// Host <-- [ kAck,          ___,      ___     ] <-- Remote
// * Host continues execution of the client.
// * Client terminates.

#define DISPATCH_FUNCTION_NAME dispatch_875b2e3a28186123
#define MESSAGE_BUFFER_NAME message_buffer_71d6a7b93c318d7e

#endif  // TVM_RUNTIME_HEXAGON_RPC_SIMULATOR_HEXAGON_SIM_PROTO_H_
