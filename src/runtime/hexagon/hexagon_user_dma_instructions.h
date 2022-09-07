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

#ifndef TVM_RUNTIME_HEXAGON_HEXAGON_USER_DMA_INSTRUCTIONS_H_
#define TVM_RUNTIME_HEXAGON_HEXAGON_USER_DMA_INSTRUCTIONS_H_

namespace tvm {
namespace runtime {
namespace hexagon {

inline unsigned int dmpause() {
  unsigned int dm0 = 0;
  asm volatile(" %0 = dmpause" : "=r"(dm0));
  return dm0;
}

inline void dmstart(void* next) { asm volatile(" dmstart(%0)" : : "r"(next)); }

inline void dmlink(void* tail, void* next) {
  asm volatile(" dmlink(%0, %1)" : : "r"(tail), "r"(next));
}

inline unsigned int dmpoll() {
  unsigned int dm0 = 0;
  asm volatile(" %0 = dmpoll" : "=r"(dm0));
  return dm0;
}

inline unsigned int dmwait() {
  unsigned int dm0 = 0;
  asm volatile(" %0 = dmwait" : "=r"(dm0));
  return dm0;
}

inline void dmresume(unsigned int dm0) { asm volatile(" dmresume(%0)" : : "r"(dm0)); }

inline unsigned int dmsyncht() {
  unsigned int dm0 = 0;
  asm volatile(" %0 = dmsyncht" : "=r"(dm0));
  return dm0;
}

inline unsigned int dmtlbsynch() {
  unsigned int dm0 = 0;
  asm volatile(" %0 = dmtlbsynch" : "=r"(dm0));
  return dm0;
}

inline unsigned int dmcfgrd(unsigned int dmindex) {
  unsigned int data = 0;
  asm volatile(" %0 = dmcfgrd(%1)" : "=r"(data) : "r"(dmindex));
  return data;
}

inline void dmcfgwr(unsigned int dmindex, unsigned int data) {
  asm volatile(" dmcfgwr(%0, %1)" : : "r"(dmindex), "r"(data));
}

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_HEXAGON_HEXAGON_USER_DMA_INSTRUCTIONS_H_
