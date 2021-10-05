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

#ifndef TVM_RUNTIME_HEXAGON_SIM_HEXAGON_SIM_PROTO_H_
#define TVM_RUNTIME_HEXAGON_SIM_HEXAGON_SIM_PROTO_H_

// Protocol:

// Host >-- [ code:MsgReq,  len:amount requested, va:_       ] --> Remote
// Host <-- [ code:MsqAck,  len:amount provided,  va:address ] --< Remote
// Host >-- [ code:message, len:payload length,   va:address ] --> Remote
// Host <-- [ code:None,    len:response length,  va:address ] --< Remote

enum : uint32_t {
  kNone,
  kMsgReq,
  kMsgAck,
  kAlloc,
  kFree,
  kCopy,
  kLoad,
  kUnload,
  kResolve,
  kCall,
  kFlush,
  kAllocVtcm
};

struct Message {
  uint32_t code;
  uint32_t len;
  uint32_t va;
} __attribute__((packed));

struct MsgAlloc {
  uint32_t size;
  uint32_t align;
} __attribute__((packed));

struct MsgPointer {
  uint32_t va;
} __attribute__((packed));

struct MsgCopy {
  uint32_t dst;
  uint32_t src;
  uint32_t len;
} __attribute__((packed));

struct MsgCall {
  uint32_t func_va;     // offset:  0
  uint32_t scalar_num;  //          4
  uint32_t stack_num;   //          8
  uint32_t data[];      //         12
} __attribute__((packed));

#endif  // TVM_RUNTIME_HEXAGON_SIM_HEXAGON_SIM_PROTO_H_
