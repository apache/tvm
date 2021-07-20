#ifndef IMPLEMENTATION
#define IMPLEMENTATION

#include "Arduino.h"
#include "standalone_crt/include/tvm/runtime/c_runtime_api.h"
#include "standalone_crt/include/tvm/runtime/crt/graph_executor.h"
#include "standalone_crt/include/tvm/runtime/crt/logging.h"
#include "standalone_crt/include/tvm/runtime/crt/stack_allocator.h"

size_t TVMPlatformFormatMessage(char* out_buf, size_t out_buf_size_bytes, const char* fmt,
                                va_list args) {
  return 0;
}

void TVMLogf(const char* msg, ...) {
  /*char buffer[256];
  int size;
  va_list args;
  va_start(args, msg);
  size = vsprintf(buffer, msg, args);
  va_end(args);
  TVMPlatformWriteSerial(buffer, (uint32_t)size);*/
}

// Blink code for debugging purposes
void TVMPlatformAbort(tvm_crt_error_t error) {
  // pinMode(LED_BUILTIN, OUTPUT);
  for (;;) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(250);
    digitalWrite(LED_BUILTIN, LOW);
    delay(250);
    digitalWrite(LED_BUILTIN, HIGH);
    delay(250);
    digitalWrite(LED_BUILTIN, LOW);
    delay(750);
  }
}

// Heap for use by TVMPlatformMemoryAllocate.

// Called by TVM to allocate memory.
tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  if (num_bytes == 0) {
    num_bytes = sizeof(int);
  }
  *out_ptr = malloc(num_bytes);
  return (*out_ptr == NULL) ? kTvmErrorPlatformNoMemory : kTvmErrorNoError;
}

// Called by TVM to deallocate memory.
tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  free(ptr);
  return kTvmErrorNoError;
}

unsigned long g_utvm_start_time;

#define MILLIS_TIL_EXPIRY 200

int g_utvm_timer_running = 0;

tvm_crt_error_t TVMPlatformTimerStart() {
  if (g_utvm_timer_running) {
    return kTvmErrorPlatformTimerBadState;
  }
  g_utvm_timer_running = 1;
  g_utvm_start_time = micros();
  return kTvmErrorNoError;
}

tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) {
  if (!g_utvm_timer_running) {
    return kTvmErrorPlatformTimerBadState;
  }
  g_utvm_timer_running = 0;
  unsigned long g_utvm_stop_time = micros() - g_utvm_start_time;
  *elapsed_time_seconds = ((double)g_utvm_stop_time) / 1e6;
  return kTvmErrorNoError;
}

tvm_crt_error_t TVMPlatformGenerateRandom(uint8_t* buffer, size_t num_bytes) {
  for (size_t i = 0; i < num_bytes; i++) {
    buffer[i] = rand();
  }
  return kTvmErrorNoError;
}

#endif
