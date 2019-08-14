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
 *  Copyright (c) 2018 by Contributors
 * \file driver.h
 * \brief Driver interface that is used by runtime.
 *
 * Driver's implementation is device specific.
 */

#ifndef VTA_DRIVER_H_
#define VTA_DRIVER_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>

/*! \brief Memory management constants for cached memory */
#define VTA_CACHED 1
/*! \brief Memory management constants for non-cached memory */
#define VTA_NOT_CACHED 0

/*! \brief Physically contiguous buffer size limit */
#ifndef VTA_MAX_XFER
#define VTA_MAX_XFER (1<<25)
#endif

/*! PAGE SIZE */
#define VTA_PAGE_BITS 12
#define VTA_PAGE_BYTES (1 << VTA_PAGE_BITS)

/*! \brief Device resource context  */
typedef void * VTADeviceHandle;

/*! \brief physical address */
#ifdef USE_TSIM
typedef uint64_t vta_phy_addr_t;
#else
typedef uint32_t vta_phy_addr_t;
#endif

/*!
 * \brief Allocate a device resource handle
 * \return The device handle.
 */
VTADeviceHandle VTADeviceAlloc();

/*!
 * \brief Free a device handle
 * \param handle The device handle to be freed.
 */
void VTADeviceFree(VTADeviceHandle handle);

/*!
 * \brief Launch the instructions block until done.
 * \param device The device handle.
 * \param insn_phy_addr The physical address of instruction stream.
 * \param insn_count Instruction count.
 * \param wait_cycles The maximum of cycles to wait
 *
 * \return 0 if running is successful, 1 if timeout.
 */
#ifdef USE_TSIM
int VTADeviceRun(VTADeviceHandle device,
                 vta_phy_addr_t insn_phy_addr,
                 vta_phy_addr_t uop_phy_addr,
                 vta_phy_addr_t inp_phy_addr,
                 vta_phy_addr_t wgt_phy_addr,
                 vta_phy_addr_t acc_phy_addr,
                 vta_phy_addr_t out_phy_addr,
                 uint32_t insn_count,
                 uint32_t wait_cycles);
#else
int VTADeviceRun(VTADeviceHandle device,
                 vta_phy_addr_t insn_phy_addr,
                 uint32_t insn_count,
                 uint32_t wait_cycles);
#endif

/*!
 * \brief Allocates physically contiguous region in memory readable/writeable by FPGA.
 * \param size Size of the region in Bytes.
 * \param cached Region can be set to not cached (write-back) if set to 0.
 * \return A pointer to the allocated region.
 */
void* VTAMemAlloc(size_t size, int cached);

/*!
 * \brief Frees a physically contiguous region in memory readable/writeable by FPGA.
 * \param buf Buffer to free.
 */
void VTAMemFree(void* buf);

/*!
 * \brief Returns a physical address to the region of memory allocated with VTAMemAlloc.
 * \param buf Pointer to memory region allocated with VTAMemAlloc.
 * \return The physical address of the memory region.
 */
vta_phy_addr_t VTAMemGetPhyAddr(void* buf);

/*!
 * \brief Performs a copy operation from host memory to buffer allocated with VTAMemAlloc.
 * \param dst The desination buffer in FPGA-accessible memory. Has to be allocated with VTAMemAlloc.
 * \param src The source buffer in host memory.
 * \param size Size of the region in Bytes.
 */
void VTAMemCopyFromHost(void* dst, const void* src, size_t size);

/*!
 * \brief Performs a copy operation from buffer allocated with VTAMemAlloc to host memory.
 * \param dst The destination buffer in host memory.
 * \param src The source buffer in FPGA-accessible memory. Has to be allocated with VTAMemAlloc.
 * \param size Size of the region in Bytes.
 */
void VTAMemCopyToHost(void* dst, const void* src, size_t size);

/*!
 * \brief Flushes the region of memory out of the CPU cache to DRAM.
 * \param vir_addr Pointer to memory region allocated with VTAMemAlloc to be flushed.
 *                 This need to be the virtual address.
 * \param phy_addr Pointer to memory region allocated with VTAMemAlloc to be flushed.
 *                 This need to be the physical address.
 * \param size Size of the region to flush in Bytes.
 */
void VTAFlushCache(void* vir_addr, vta_phy_addr_t phy_addr, int size);

/*!
 * \brief Invalidates the region of memory that is cached.
 * \param vir_addr Pointer to memory region allocated with VTAMemAlloc to be invalidated.
 *                 This need to be the virtual address.
 * \param phy_addr Pointer to memory region allocated with VTAMemAlloc to be invalidated.
 *                 This need to be the physical address.
 * \param size Size of the region to invalidate in Bytes.
 */
void VTAInvalidateCache(void* vir_addr, vta_phy_addr_t phy_addr, int size);

#ifdef __cplusplus
}
#endif
#endif  // VTA_DRIVER_H_
