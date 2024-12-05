if(WITH_TVM_MODULE)
    set(TVM_MODULE_DIR "../build")
    set(TVM_SOURCES "${TVM_MODULE_DIR}/codegen/host/src/${WITH_TVM_MODULE}_lib0.c"
                    "${TVM_MODULE_DIR}/codegen/host/src/${WITH_TVM_MODULE}_lib1.c"
                    "${TVM_MODULE_DIR}/runtime/src/runtime/crt/memory/stack_allocator.c"
                    "${TVM_MODULE_DIR}/runtime/src/runtime/crt/common/crt_backend_api.c"
    )
    set(TVM_INCLUDES "../include"
                     "${TVM_MODULE_DIR}/include"
                     "${TVM_MODULE_DIR}/runtime/include"
                     "${TVM_MODULE_DIR}/codegen/host/include"
    )
endif()

idf_component_register(
    SRCS mfcc.cc
         mfcc_preprocessor.cc
         fragment_detector.cc
         mic_reader.cc
         audio_rx_slot.cc
         main.cc
         "${TVM_SOURCES}"
         "${NMSIS}/DSP/Source/FastMathFunctions/riscv_cos_f32.c"
         "${NMSIS}/DSP/Source/CommonTables/riscv_common_tables.c"
         "${NMSIS}/DSP/Source/CommonTables/riscv_const_structs.c"
         "${NMSIS}/DSP/Source/TransformFunctions/riscv_rfft_fast_f32.c"
         "${NMSIS}/DSP/Source/TransformFunctions/riscv_cfft_f32.c"
         "${NMSIS}/DSP/Source/TransformFunctions/riscv_cfft_radix8_f32.c"
         "${NMSIS}/DSP/Source/TransformFunctions/riscv_bitreversal2.c"
         "${NMSIS}/DSP/Source/TransformFunctions/riscv_rfft_fast_init_f32.c"
         "${NMSIS}/DSP/Source/TransformFunctions/riscv_cfft_init_f32.c"
    INCLUDE_DIRS include
         "${TVM_INCLUDES}"
         "${NMSIS}/DSP/Include"
         "${NMSIS}/Core/Include"
    )

if(WITH_TVM_MODULE)
    add_compile_definitions(TVM_MODULE=${WITH_TVM_MODULE})
    target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-error=unused-const-variable -Wformat=0)
endif()

# Board
#   A_TARGET_ESP32C3_XIAO // xiao-esp32-c3
#   A_TARGET_ESP32C3_LCD  // xiao-esp32-c3 + lcd
#   A_TARGET_ESP32C3_MINI // mini D1 PLUS v1.2
add_compile_definitions(A_TARGET_ESP32C3_XIAO)
add_compile_definitions(A_SAMPLE_RATE=16000)
# Mic (INMP441)
#   BPS: 8 16 (24) 32
add_compile_definitions(A_HW_BITS_PER_SAMPLE_DATA_24BIT)
add_compile_definitions(A_HW_BIT_SHIFT)
# Stream
#   BPS: 8 (16) 24 32
add_compile_definitions(A_BITS_PER_SAMPLE_DATA_16BIT)
#   Volume amp: X2 X4 X8 X16 X32 (X64) X128
add_compile_definitions(A_VOLUME_X64)
# Process
#   MFCC buffer init: (ENVIRONMENT) ZERO NOISE
add_compile_definitions(A_MFCC_ENVIRONMENT_PREPARED_TEMPLATE)
#   KWS opts
add_compile_definitions(KWS_NUM_MFCC=10)
add_compile_definitions(KWS_WIN_MS=40)
add_compile_definitions(KWS_STRIDE_MS=20)
