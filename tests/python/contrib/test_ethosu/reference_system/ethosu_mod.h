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
#ifndef TVM_RUNTIME_CONTRIB_ETHOS_U_ETHOSU_MOD_H_
#define TVM_RUNTIME_CONTRIB_ETHOS_U_ETHOSU_MOD_H_

#include <ARMCM55.h>
// TODO(@grant-arm): Remove device specific information once RTOS support is available
#include <ethosu_driver.h>
#include <stdio.h>

#include "ethosu_55.h"

struct ethosu_driver ethosu0_driver;

void ethosuIrqHandler0() { ethosu_irq_handler(&ethosu0_driver); }

// Initialize Arm(R) Ethos(TM)-U NPU driver
int EthosuInit() {
  if (ethosu_init(&ethosu0_driver, (void*)ETHOSU_BASE_ADDRESS, NULL, 0, 1, 1)) {
    printf("Failed to initialize NPU.\n");
    return -1;
  }

  // Assumes SCB->VTOR points to RW memory
  NVIC_SetVector(ETHOSU_IRQ, (uint32_t)&ethosuIrqHandler0);
  NVIC_EnableIRQ(ETHOSU_IRQ);

  return 0;
}

#endif  // TVM_RUNTIME_CONTRIB_ETHOS_U_ETHOSU_MOD_H_
