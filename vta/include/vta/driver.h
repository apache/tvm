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
#define VTA_MAX_XFER (1<<22)
#endif

/*! \brief Device resource context  */
typedef void * VTADeviceHandle;

/*! \brief physical address */
typedef uint32_t vta_phy_addr_t;

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
int VTADeviceRun(VTADeviceHandle device,
                 vta_phy_addr_t insn_phy_addr,
                 uint32_t insn_count,
                 uint32_t wait_cycles);

/*!
 * \brief Allocates physically contiguous region in memory (limited by MAX_XFER).
 * \param size Size of the region in Bytes.
 * \param cached Region can be set to not cached (write-back) if set to 0.
 * \return A pointer to the allocated region.
 */
void* VTAMemAlloc(size_t size, int cached);

/*!
 * \brief Frees a physically contiguous region in memory.
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
 * \brief Flushes the region of memory out of the CPU cache to DRAM.
 * \param buf Pointer to memory region allocated with VTAMemAlloc to be flushed.
 *            This need to be the physical address.
 * \param size Size of the region to flush in Bytes.
 */
void VTAFlushCache(vta_phy_addr_t buf, int size);

/*!
 * \brief Invalidates the region of memory that is cached.
 * \param buf Pointer to memory region allocated with VTAMemAlloc to be invalidated.
 *            This need to be the physical address.
 * \param size Size of the region to invalidate in Bytes.
 */
void VTAInvalidateCache(vta_phy_addr_t buf, int size);

/*!
 * \brief Programming the bit stream on the FPGA.
 * \param bitstream The path to the bit stream file.
 */
void VTAProgram(const char* bitstream);

#ifdef __cplusplus
}
#endif
#endif  // VTA_DRIVER_H_
