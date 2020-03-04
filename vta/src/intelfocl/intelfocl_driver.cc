#include <string>
#include <iostream>
#include <vta/driver.h>
#include "intelfocl_device.h"

#define MEM_ADDR_IDENTIFIER (0x18000000)

static IntelFOCLDevice focl_device;

static inline void* mem_get_addr(ifocl_mem_off_t offset)
{
    void *ret = (void *) (offset + MEM_ADDR_IDENTIFIER);
    return ret;
}

static inline ifocl_mem_off_t mem_get_offset(const void *addr)
{
    ifocl_mem_off_t ret = (ifocl_mem_off_t) addr - MEM_ADDR_IDENTIFIER;
    return ret;
}

void* VTAMemAlloc(size_t size, int cached) {
    (void) cached;
    ifocl_mem_off_t offset = focl_device.alloc(size);
    if ( offset == IFOCL_MEM_OFF_ERR ) return NULL;
    void *addr = mem_get_addr(offset);
    return addr;
}

void VTAMemFree(void *buf) {
    ifocl_mem_off_t offset = mem_get_offset(buf);
    focl_device.free(offset);
}

vta_phy_addr_t VTAMemGetPhyAddr(void* buf) {
    ifocl_mem_off_t offset = mem_get_offset(buf);
    return (vta_phy_addr_t) offset;
}

void VTAMemCopyFromHost(void* dst, const void* src, size_t size) {
    ifocl_mem_off_t dst_offset = mem_get_offset(dst);
    focl_device.write_mem(dst_offset, src, size);
}

void VTAMemCopyToHost(void* dst, const void* src, size_t size) {
    ifocl_mem_off_t src_offset = mem_get_offset(src);
    focl_device.read_mem(src_offset, dst, size);
}

void VTAFlushCache(void * offset, vta_phy_addr_t buf, int size) {
    std::cout << "VTAFlushCache not implemented for Intel OpenCL for FPGA devices" << std::endl;
}

void VTAInvalidateCache(void * offset, vta_phy_addr_t buf, int size) {
    std::cout << "VTAInvalidateCache not implemented for Intel OpenCL for FPGA devices" << std::endl;
}

VTADeviceHandle VTADeviceAlloc() {
    return (VTADeviceHandle) &focl_device;
}

void VTADeviceFree(VTADeviceHandle handle) {
    (void) handle;
}

int VTADeviceRun(VTADeviceHandle handle,
        vta_phy_addr_t insn_phy_addr,
        uint32_t insn_count,
        uint32_t wait_cycles)
{
    (void) wait_cycles;
    ifocl_mem_off_t offset = (ifocl_mem_off_t) insn_phy_addr;
    return focl_device.execute_instructions(offset, insn_count);
}
