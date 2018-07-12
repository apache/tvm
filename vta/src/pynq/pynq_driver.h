/*!
 *  Copyright (c) 2018 by Contributors
 * \file vta_pynq_driver.h
 * \brief VTA driver for Pynq board.
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

#ifdef __arm__
#include <libxlnk_cma.h>
#else
void* cma_alloc(size_t size, int cached);
void cma_free(void* buf);
uint32_t cma_get_phy_addr(void* buf);
#endif
void xlnkFlushCache(void* buf, int size);
void xlnkInvalidateCache(void* buf, int size);

void *VTAMapRegister(uint32_t addr, size_t length);
void VTAUnmapRegister(void *vta, size_t length);
void VTAWriteMappedReg(void* base_addr, uint32_t offset, uint32_t val);
uint32_t VTAReadMappedReg(void* base_addr, uint32_t offset);

/*! \brief (Pynq only) Partial bitstream status file path */
#define VTA_PYNQ_BS_IS_PARTIAL "/sys/devices/soc0/amba/f8007000.devcfg/is_partial_bitstream"
/*! \brief (Pynq only) Bitstream destination file path */
#define VTA_PYNQ_BS_XDEVCFG "/dev/xdevcfg"

/*! \brief (Pynq only) Path to /dev/mem */
#define VTA_PYNQ_DEV_MEM_PATH "/dev/mem"
/*! \brief (Pynq only) MMIO driver constant */
#define VTA_PYNQ_MMIO_WORD_LENGTH 4
/*! \brief (Pynq only) MMIO driver constant */
#define VTA_PYNQ_MMIO_WORD_MASK (~(MMIO_WORD_LENGTH - 1))

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

#ifdef __cplusplus
}
#endif
#endif  // VTA_PYNQ_PYNQ_DRIVER_H_
