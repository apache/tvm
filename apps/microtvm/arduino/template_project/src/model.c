#ifndef TVM_IMPLEMENTATION_ARDUINO
#define TVM_IMPLEMENTATION_ARDUINO

#include "model.h"

#include "Arduino.h"
#include "standalone_crt/include/tvm/runtime/crt/internal/aot_executor/aot_executor.h"
#include "standalone_crt/include/tvm/runtime/crt/stack_allocator.h"

// AOT memory array
static uint8_t g_aot_memory[WORKSPACE_SIZE];
extern tvm_model_t tvmgen_default_network;
tvm_workspace_t app_workspace;

// Blink code for debugging purposes
void TVMPlatformAbort(tvm_crt_error_t error) {
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

void TVMLogf(const char* msg, ...) {}

tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  return StackMemoryManager_Allocate(&app_workspace, num_bytes, out_ptr);
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  return StackMemoryManager_Free(&app_workspace, ptr);
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

void TVMInitialize() { StackMemoryManager_Init(&app_workspace, g_aot_memory, WORKSPACE_SIZE); }

void TVMExecute(void* input_data, void* output_data) {
  void* inputs[1] = {
      input_data,
  };
  void* outputs[1] = {
      output_data,
  };
  int ret_val = tvm_runtime_run(&tvmgen_default_network, inputs, outputs);
}

#endif
