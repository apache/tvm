// 2021-08-09 11:02
#include "tvm_runtime.h"

#define CRT_MEMORY_NUM_PAGES 16384
#define CRT_MEMORY_PAGE_SIZE_LOG2 10

static uint8_t
    g_crt_memory[CRT_MEMORY_NUM_PAGES * (1 << CRT_MEMORY_PAGE_SIZE_LOG2)];
static MemoryManagerInterface* g_memory_manager;

void TVMLogf(const char* msg, ...) {
    va_list args;
    va_start(args, msg);
    vfprintf(stderr, msg, args);
    va_end(args);
}

void __attribute__((noreturn)) TVMPlatformAbort(tvm_crt_error_t error_code) {
    fprintf(stderr, "TVMPlatformAbort: %d\n", error_code);
    exit(-1);
}

tvm_crt_error_t TVMPlatformMemoryAllocate(
    size_t num_bytes, DLDevice dev, void** out_ptr) {
    return g_memory_manager->Allocate(
        g_memory_manager, num_bytes, dev, out_ptr);
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
    return g_memory_manager->Free(g_memory_manager, ptr, dev);
}

tvm_crt_error_t TVMPlatformTimerStart() {
    return kTvmErrorFunctionCallNotImplemented;
}

tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) {
    return kTvmErrorFunctionCallNotImplemented;
}

TVMModuleHandle tvm_runtime_create() {
    DLDevice dev;
    dev.device_type = (DLDeviceType)kDLCPU;
    dev.device_id = 0;

    // get pointers
    PageMemoryManagerCreate(
        &g_memory_manager, g_crt_memory, sizeof(g_crt_memory),
        CRT_MEMORY_PAGE_SIZE_LOG2);
    TVMInitializeRuntime();
    TVMPackedFunc pf;
    TVMArgs args = TVMArgs_Create(NULL, NULL, 0);
    TVMPackedFunc_InitGlobalFunc(&pf, "runtime.SystemLib", &args);
    TVMPackedFunc_Call(&pf);

    TVMModuleHandle mod_syslib = TVMArgs_AsModuleHandle(&pf.ret_value, 0);
    return mod_syslib;
}
