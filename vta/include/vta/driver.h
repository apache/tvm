/*!
 *  Copyright (c) 2018 by Contributors
 * \file vta_driver.h
 * \brief General driver interface.
 */

#ifndef VTA_DRIVER_H_
#define VTA_DRIVER_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdint.h>

/*! \brief Memory management constants with libxlnk_cma */
#define CACHED 1
/*! \brief Memory management constants with libxlnk_cma */
#define NOT_CACHED 0

/*! \brief VTA command handle */
typedef void * VTAHandle;

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
uint32_t VTAGetMemPhysAddr(void* buf);

/*!
 * \brief Flushes the region of memory out of the CPU cache to DRAM.
 * \param buf Pointer to memory region allocated with VTAMemAlloc to be flushed.
 * \param size Size of the region to flush in Bytes.
 */
void VTAFlushCache(void* buf, int size);

/*!
 * \brief Invalidates the region of memory that is cached.
 * \param buf Pointer to memory region allocated with VTAMemAlloc to be invalidated.
 * \param size Size of the region to invalidate in Bytes.
 */
void VTAInvalidateCache(void* buf, int size);

/*!
 * \brief Returns a memory map to FPGA configuration registers.
 * \param addr The base physical address of the configuration registers.
 * \param length The size of the memory mapped region in bytes.
 * \return A pointer to the memory mapped region.
 */
void *VTAMapRegister(unsigned addr, size_t length);

/*!
 * \brief Deletes the configuration register memory map.
 * \param vta The memory mapped region.
 * \param length The size of the memory mapped region in bytes.
 */
void VTAUnmapRegister(void *vta, size_t length);

/*!
 * \brief Writes to a memory mapped configuration register.
 * \param vta_base The handle to the memory mapped configuration registers.
 * \param offset The offset of the register to write to.
 * \param val The value to be written to the memory mapped register.
 */
void VTAWriteMappedReg(VTAHandle vta_base, unsigned offset, unsigned val);

/*!
 * \brief Reads from the memory mapped configuration register.
 * \param vta_base The handle to the memory mapped configuration registers.
 * \param offset The offset of the register to read from.
 * \return The value read from the memory mapped register.
 */
unsigned VTAReadMappedReg(VTAHandle vta_base, unsigned offset);

/*!
 * \brief Programming the bit stream on the FPGA.
 * \param bitstream The path to the bit stream file.
 */
void VTAProgram(const char* bitstream);

#ifdef __cplusplus
}
#endif
#endif // VTA_DRIVER_H_
