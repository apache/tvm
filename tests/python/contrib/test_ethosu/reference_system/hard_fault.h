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
#ifndef TVM_RUNTIME_CONTRIB_ETHOS_U_HARD_FAULT_H_
#define TVM_RUNTIME_CONTRIB_ETHOS_U_HARD_FAULT_H_

struct ExcContext {
  uint32_t r0;
  uint32_t r1;
  uint32_t r2;
  uint32_t r3;
  uint32_t r12;
  uint32_t lr;
  uint32_t pc;
  uint32_t xPsr;
};
void HardFault_Handler() {
  int irq;
  struct ExcContext* e;
  uint32_t sp;
  asm volatile(
      "mrs %0, ipsr            \n"  // Read IPSR (Exception number)
      "sub %0, #16             \n"  // Get it into IRQn_Type range
      "tst lr, #4              \n"  // Select the stack which was in use
      "ite eq                  \n"
      "mrseq %1, msp           \n"
      "mrsne %1, psp           \n"
      "mov %2, sp              \n"
      : "=r"(irq), "=r"(e), "=r"(sp));
  printf("Hard fault. irq=%d, pc=0x%08lu, lr=0x%08lu, xpsr=0x%08lu, sp=0x%08lu\n", irq, e->pc,
         e->lr, e->xPsr, sp);
  printf("%11s cfsr=0x%08lu bfar=0x%08lu\n", "", SCB->CFSR, SCB->BFAR);
  printf("EXITTHESIM\n");
  while (1 == 1)
    ;
}

#endif  // TVM_RUNTIME_CONTRIB_ETHOS_U_HARD_FAULT_H_
