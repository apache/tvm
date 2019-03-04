/*!
 *  Copyright (c) 2018 by Contributors
 * \file de10-nano_driver.h
 * \brief VTA driver for DE10_Nano board.
 */

#ifndef VTA_DE10_NANO_DE10_NANO_DRIVER_H_
#define VTA_DE10_NANO_DE10_NANO_DRIVER_H_

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

void *VTAMapRegister(uint32_t addr, size_t length);
void VTAUnmapRegister(void *vta, size_t length);
void VTAWriteMappedReg(void* base_addr, uint32_t offset, uint32_t val);
uint32_t VTAReadMappedReg(void* base_addr, uint32_t offset);

/*! \brief (DE10_Nano only) Bitstream destination file path */
#define VTA_DE10_NANO_BS_XDEVCFG "/dev/fpga0"

/*! \brief (DE10_Nano only) Path to /dev/mem */
#define VTA_DE10_NANO_DEV_MEM_PATH "/dev/mem"
/*! \brief (DE10_Nano only) MMIO driver constant */
#define VTA_DE10_NANO_MMIO_WORD_LENGTH 4
/*! \brief (DE10_Nano only) MMIO driver constant */
#define VTA_DE10_NANO_MMIO_WORD_MASK (~(MMIO_WORD_LENGTH - 1))

/*! \brief VTA configuration register address range */
#define VTA_RANGE 0x100
/*! \brief VTA configuration register start value */
#define VTA_START 0x1
/*! \brief VTA configuration register auto-restart value */
#define VTA_AUTORESTART 0x81
/*! \brief VTA configuration register done value */
#define VTA_DONE 0x1

/*! \brief VTA fetch stage configuration register address
*   from auto-generated FETCH_0_S_AXI_CONTROL_BUS_BASEADDR define
*   in hps_0.h (under build/hardware/intel/hls/<design name>/)
*/
#define VTA_FETCH_ADDR    0xFF220000
/*! \brief VTA compute stage configuration register address
*   from auto-generated LOAD_0_S_AXI_CONTROL_BUS_BASEADDR define
*   in hps_0.h (under build/hardware/intel/hls/<design name>/)
*/
#define VTA_LOAD_ADDR     0xFF221000
/*! \brief VTA compute stage configuration register address
*   from auto-generated COMPUTE_0_S_AXI_CONTROL_BUS_BASEADDR define
*   in hps_0.h (under build/hardware/intel/hls/<design name>/)
*/
#define VTA_COMPUTE_ADDR  0xFF222000
/*! \brief VTA store stage configuration register address
*   from auto-generated STORE_0_S_AXI_CONTROL_BUS_BASEADDR define
*   in hps_0.h (under build/hardware/intel/hls/<design name>/)
*/
#define VTA_STORE_ADDR    0xFF223000

#ifdef __cplusplus
}
#endif
#endif  // VTA_DE10_NANO_DE10_NANO_DRIVER_H_
