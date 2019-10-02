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
 * \file pynq_driver.c
 * \brief VTA driver for Pynq board.
 */

#include <vta/driver.h>
#include <thread>
#include "pynq_driver.h"

#define VTA_BASE_ADDR 0x43c00000

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
    vta_handle_ = VTAMapRegister(VTA_BASE_ADDR, VTA_RANGE);
  }

  ~VTADevice() {
    // Close VTA stage handle
    VTAUnmapRegister(vta_handle_, VTA_RANGE);
  }

  int Run(vta_phy_addr_t insn_phy_addr,
          uint32_t insn_count,
          uint32_t wait_cycles) {
    VTAWriteMappedReg(vta_handle_, 0x04, 0);
    VTAWriteMappedReg(vta_handle_, 0x08, insn_count);
    VTAWriteMappedReg(vta_handle_, 0x0c, insn_phy_addr);
    VTAWriteMappedReg(vta_handle_, 0x10, 0);
    VTAWriteMappedReg(vta_handle_, 0x14, 0);
    VTAWriteMappedReg(vta_handle_, 0x18, 0);
    VTAWriteMappedReg(vta_handle_, 0x1c, 0);
    VTAWriteMappedReg(vta_handle_, 0x20, 0);
    VTAWriteMappedReg(vta_handle_, 0x0, VTA_START);

    // Loop until the VTA is done
    unsigned t, flag = 0;
    for (t = 0; t < wait_cycles; ++t) {
      flag = VTAReadMappedReg(vta_handle_, 0x0);
      if (flag & 2) break;
      std::this_thread::yield();
    }
    uint32_t cycles = 0;
    cycles = VTAReadMappedReg(vta_handle_, 0x4);
    printf("cycles:%d\n", cycles);
    // Report error if timeout
    return t < wait_cycles ? 0 : 1;
  }

 private:
  void* vta_handle_{nullptr};
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
