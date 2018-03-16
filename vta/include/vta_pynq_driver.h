/*!
 *  Copyright (c) 2018 by Contributors
 * \file vta_pynq_driver.h
 * \brief VTA driver for Pynq board.
 */

#ifndef VTA_PYNQ_DRIVER_H_
#define VTA_PYNQ_DRIVER_H_

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

#ifdef __arm__
#include "libxlnk_cma.h"
#else
void* cma_alloc(size_t size, int cached);
void cma_free(void* buf);
uint32_t cma_get_phy_addr(void* buf);
void xlnkFlushCache(void* buf, int size);
void xlnkInvalidateCache(void* buf, int size);
#endif

/*! \brief VTA command handle */
typedef void * VTAHandle;

/*! \brief DMA command handle */
typedef struct {
  /*! \brief Register map to the AXI DMA control registers*/
  void *dma_register_map;
  /*! \brief Transmit data descriptor*/
  void *mm2s_descriptor_register_map;
  /*! \brief Receive data descriptor*/
  void *s2mm_descriptor_register_map;
  /*! \brief Transmit data descriptor physical address*/
  uint32_t mm2s_descriptor_phy;
  /*! \brief Receive data descriptor physical address*/
  uint32_t s2mm_descriptor_phy;
  /*! \brief Descriptor size */
  uint32_t descriptor_size;
  /*! \brief Transaction count for tx channel */
  uint32_t mm2s_count;
  /*! \brief Transaction count for rx channel */
  uint32_t s2mm_count;
  /*! \brief Multi-channel mode enable */
  int multichannel_en;
} DMAHandle;

/*! \brief partial bitstream status file path */
#define BS_IS_PARTIAL "/sys/devices/soc0/amba/f8007000.devcfg/is_partial_bitstream"
/*! \brief bitstream destination file path */
#define BS_XDEVCFG "/dev/xdevcfg"

/*! \brief Path to /dev/mem */
#define DEV_MEM_PATH "/dev/mem"
/*! \brief MMIO driver constant */
#define MMIO_WORD_LENGTH 4
/*! \brief MMIO driver constant */
#define MMIO_WORD_MASK (~(MMIO_WORD_LENGTH - 1))

/*! \brief VTA configuration register address range */
#define VTA_RANGE 0x100
/*! \brief VTA configuration register start value */
#define VTA_START 0x1
/*! \brief VTA configuration register auto-restart value */
#define VTA_AUTORESTART 0x81
/*! \brief VTA configuration register done value */
#define VTA_DONE 0x1

/*! \brief VTA fetch stage configuration register address
*   from auto-generated XPAR_FETCH_0_S_AXI_CONTROL_BUS_BASEADDR define
*   in xparameters.h (under build/vivado/<design name>/export/bsp/ps7_cortexa9_0/include)
*/
#define VTA_FETCH_ADDR    0x43C00000
/*! \brief VTA compute stage configuration register address
*   from auto-generated XPAR_COMPUTE_0_S_AXI_CONTROL_BUS_BASEADDR define
*   in xparameters.h (under build/vivado/<design name>/export/bsp/ps7_cortexa9_0/include)
*/
#define VTA_COMPUTE_ADDR  0x43C10000
/*! \brief VTA compute stage configuration register address
*   from auto-generated XPAR_LOAD_0_S_AXI_CONTROL_BUS_BASEADDR define
*   in xparameters.h (under build/vivado/<design name>/export/bsp/ps7_cortexa9_0/include)
*/
#define VTA_LOAD_ADDR     0x43C20000
/*! \brief VTA store stage configuration register address
*   from auto-generated XPAR_STORE_0_S_AXI_CONTROL_BUS_BASEADDR define
*   in xparameters.h (under build/vivado/<design name>/export/bsp/ps7_cortexa9_0/include)
*/
#define VTA_STORE_ADDR    0x43C30000

/*! \brief Memory management constants with libxlnk_cma */
#define CACHED 1
/*! \brief Memory management constants with libxlnk_cma */
#define NOT_CACHED 0

/*! \brief log2 of SDS buffer size limit */
#define LOG_MAX_XFER 22
/*! \brief SDS buffer size limit */
#define MAX_XFER (1<<LOG_MAX_XFER)

/*!
 * \brief Returns a memory map to FPGA configuration registers.
 * \param addr The base physical address of the configuration registers.
 * \param length The size of the memory mapped region in bytes.
 * \return A pointer to the memory mapped region.
 */
void *MapRegister(unsigned addr, size_t length);

/*!
 * \brief Deletes the configuration register memory map.
 * \param vta The memory mapped region.
 * \param length The size of the memory mapped region in bytes.
 */
void UnmapRegister(void *vta, size_t length);

/*!
 * \brief Writes to a memory mapped configuration register.
 * \param vta_base The handle to the memory mapped configuration registers.
 * \param offset The offset of the register to write to.
 * \param val The value to be written to the memory mapped register.
 */
void WriteMappedReg(VTAHandle vta_base, unsigned offset, unsigned val);

/*!
 * \brief Reads from the memory mapped configuration register.
 * \param vta_base The handle to the memory mapped configuration registers.
 * \param offset The offset of the register to read from.
 * \return The value read from the memory mapped register.
 */
unsigned ReadMappedReg(VTAHandle vta_base, unsigned offset);

/*!
 * \brief Programming the bit stream on the FPGA.
 * \param bitstream The path to the bit stream file.
 */
void ProgramVTA(const char* bitstream);

#ifdef __cplusplus
}
#endif
#endif  // VTA_PYNQ_DRIVER_H_