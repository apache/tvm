/*!
 *  Copyright (c) 2018 by Contributors
 * \file pynq_driver.c
 * \brief VTA driver for Pynq board.
 */

#include <vta/driver.h>
#include <thread>
#include "pynq_driver.h"


void* VTAMemAlloc(size_t size, int cached) {
  return cma_alloc(size, cached);
}

void VTAMemFree(void* buf) {
  cma_free(buf);
}

vta_phy_addr_t VTAMemGetPhyAddr(void* buf) {
  return cma_get_phy_addr(buf);
}

void VTAFlushCache(vta_phy_addr_t buf, int size) {
  xlnkFlushCache(reinterpret_cast<void*>(buf), size);
}

void VTAInvalidateCache(vta_phy_addr_t buf, int size) {
  xlnkInvalidateCache(reinterpret_cast<void*>(buf), size);
}

void *VTAMapRegister(uint32_t addr, size_t length) {
  // Align the base address with the pages
  uint32_t virt_base = addr & ~(getpagesize() - 1);
  // Calculate base address offset w.r.t the base address
  uint32_t virt_offset = addr - virt_base;
  // Open file and mmap
  uint32_t mmap_file = open(VTA_PYNQ_DEV_MEM_PATH, O_RDWR|O_SYNC);
  return mmap(NULL,
              (length+virt_offset),
              PROT_READ|PROT_WRITE,
              MAP_SHARED,
              mmap_file,
              virt_base);
}

void VTAUnmapRegister(void *vta, size_t length) {
  // Unmap memory
  int status = munmap(vta, length);
  assert(status == 0);
}

void VTAWriteMappedReg(void* base_addr, uint32_t offset, uint32_t val) {
  *((volatile uint32_t *) (reinterpret_cast<char *>(base_addr) + offset)) = val;
}

uint32_t VTAReadMappedReg(void* base_addr, uint32_t offset) {
  return *((volatile uint32_t *) (reinterpret_cast<char *>(base_addr) + offset));
}

class VTADevice {
 public:
  VTADevice() {
    // VTA stage handles
    vta_fetch_handle_ = VTAMapRegister(VTA_FETCH_ADDR, VTA_RANGE);
    vta_load_handle_ = VTAMapRegister(VTA_LOAD_ADDR, VTA_RANGE);
    vta_compute_handle_ = VTAMapRegister(VTA_COMPUTE_ADDR, VTA_RANGE);
    vta_store_handle_ = VTAMapRegister(VTA_STORE_ADDR, VTA_RANGE);
  }

  ~VTADevice() {
    // Close VTA stage handle
    VTAUnmapRegister(vta_fetch_handle_, VTA_RANGE);
    VTAUnmapRegister(vta_load_handle_, VTA_RANGE);
    VTAUnmapRegister(vta_compute_handle_, VTA_RANGE);
    VTAUnmapRegister(vta_store_handle_, VTA_RANGE);
  }

  int Run(vta_phy_addr_t insn_phy_addr,
          uint32_t insn_count,
          uint32_t wait_cycles) {
    // NOTE: Register address map is derived from the auto-generated
    // driver files available under hardware/build/vivado/<design>/export/driver
    // FETCH @ 0x10 : Data signal of insn_count_V
    VTAWriteMappedReg(vta_fetch_handle_, 0x10, insn_count);
    // FETCH @ 0x18 : Data signal of insns_V
    VTAWriteMappedReg(vta_fetch_handle_, 0x18, insn_phy_addr);
    // LOAD @ 0x10 : Data signal of inputs_V
    VTAWriteMappedReg(vta_load_handle_, 0x10, 0);
    // LOAD @ 0x18 : Data signal of weight_V
    VTAWriteMappedReg(vta_load_handle_, 0x18, 0);
    // COMPUTE @ 0x20 : Data signal of uops_V
    VTAWriteMappedReg(vta_compute_handle_, 0x20, 0);
    // COMPUTE @ 0x28 : Data signal of biases_V
    VTAWriteMappedReg(vta_compute_handle_, 0x28, 0);
    // STORE @ 0x10 : Data signal of outputs_V
    VTAWriteMappedReg(vta_store_handle_, 0x10, 0);

    // VTA start
    VTAWriteMappedReg(vta_fetch_handle_, 0x0, VTA_START);
    VTAWriteMappedReg(vta_load_handle_, 0x0, VTA_AUTORESTART);
    VTAWriteMappedReg(vta_compute_handle_, 0x0, VTA_AUTORESTART);
    VTAWriteMappedReg(vta_store_handle_, 0x0, VTA_AUTORESTART);

    // Loop until the VTA is done
    unsigned t, flag = 0;
    for (t = 0; t < wait_cycles; ++t) {
      flag = VTAReadMappedReg(vta_compute_handle_, 0x18);
      if (flag == VTA_DONE) break;
      std::this_thread::yield();
    }
    // Report error if timeout
    return t < wait_cycles ? 0 : 1;
  }

 private:
  // VTA handles (register maps)
  void* vta_fetch_handle_{nullptr};
  void* vta_load_handle_{nullptr};
  void* vta_compute_handle_{nullptr};
  void* vta_store_handle_{nullptr};
};

VTADeviceHandle VTADeviceAlloc() {
  return new VTADevice();
}

void VTADeviceFree(VTADeviceHandle handle) {
  delete static_cast<VTADevice*>(handle);
}

int VTADeviceRun(VTADeviceHandle handle,
                 vta_phy_addr_t insn_phy_addr,
                 uint32_t insn_count,
                 uint32_t wait_cycles) {
  return static_cast<VTADevice*>(handle)->Run(
      insn_phy_addr, insn_count, wait_cycles);
}

void VTAProgram(const char* bitstream) {
  int elem;
  FILE *src, *dst, *partial;
  partial = fopen(VTA_PYNQ_BS_IS_PARTIAL, "w");
  if (partial == NULL) {
    printf("Cannot open partial config file %s\n", VTA_PYNQ_BS_IS_PARTIAL);
        fclose(partial);
        exit(1);
  }
  fputc('0', partial);
  fclose(partial);
  src = fopen(bitstream, "rb");
  if (src == NULL) {
    printf("Cannot open bitstream %s\n", bitstream);
    exit(1);
  }
  dst = fopen(VTA_PYNQ_BS_XDEVCFG, "wb");
  if (dst == NULL) {
    printf("Cannot open device file %s\n", VTA_PYNQ_BS_XDEVCFG);
    fclose(dst);
    exit(1);
  }
  elem = fgetc(src);
  while (elem != EOF) {
    fputc(elem, dst);
    elem = fgetc(src);
  }
  fclose(src);
  fclose(dst);
}
