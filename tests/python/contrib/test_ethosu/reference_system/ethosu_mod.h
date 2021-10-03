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

struct ethosu_driver* ethosu0_driver = &ethosu_drv;

void ethosuIrqHandler0() { ethosu_irq_handler(ethosu0_driver); }

// Initialize Arm(R) Ethos(TM)-U NPU driver
int EthosuInit() {
  if (ethosu_init(ethosu0_driver, (void*)ETHOSU_BASE_ADDRESS, NULL, 0, 1, 1)) {
    printf("Failed to initialize NPU.\n");
    return -1;
  }

  // Display Arm(R) Ethos(TM)-U version information useful for debugging issues
  struct ethosu_version version;
  ethosu_get_version(ethosu0_driver, &version);
  printf(
      "version={major=%u, minor=%u, status=%u}, product={major=%u}, arch={major=%u, minor=%u, "
      "patch=%u}\n",
      version.id.version_major, version.id.version_minor, version.id.version_status,
      version.id.product_major, version.id.arch_major_rev, version.id.arch_minor_rev,
      version.id.arch_patch_rev);
  printf("macs_per_cc=%u, cmd_stream_version=%u, shram_size=%u\n", version.cfg.macs_per_cc,
         version.cfg.cmd_stream_version, version.cfg.shram_size);

  // Assumes SCB->VTOR points to RW memory
  NVIC_SetVector(ETHOSU_IRQ, (uint32_t)&ethosuIrqHandler0);
  NVIC_EnableIRQ(ETHOSU_IRQ);

  return 0;
}

#endif  // TVM_RUNTIME_CONTRIB_ETHOS_U_ETHOSU_MOD_H_
