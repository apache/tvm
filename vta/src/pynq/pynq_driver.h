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
 *
 * \file pynq_driver.h
 * \brief VTA driver for Zynq SoC boards with Pynq support (see pynq.io).
 */

#ifndef VTA_PYNQ_PYNQ_DRIVER_H_
#define VTA_PYNQ_PYNQ_DRIVER_H_

#ifdef __cplusplus
extern "C" {
#endif
#include <assert.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#if defined(__arm__) || defined(__aarch64__)
#include <libxlnk_cma.h>
#else
void* cma_alloc(size_t size, int cached);
void cma_free(void* buf);
uint32_t cma_get_phy_addr(void* buf);
void cma_flush_cache(void* buf, unsigned int phys_addr, int size);
void cma_invalidate_cache(void* buf, unsigned int phys_addr, int size);
#endif

void *VTAMapRegister(uint32_t addr);
void VTAUnmapRegister(void *vta);
void VTAWriteMappedReg(void* base_addr, uint32_t offset, uint32_t val);
uint32_t VTAReadMappedReg(void* base_addr, uint32_t offset);

/*! \brief VTA configuration register start value */
#define VTA_START 0x1
/*! \brief VTA configuration register auto-restart value */
#define VTA_AUTORESTART 0x81
/*! \brief VTA configuration register done value */
#define VTA_DONE 0x1

#ifdef __cplusplus
}
#endif
#endif  // VTA_PYNQ_PYNQ_DRIVER_H_
