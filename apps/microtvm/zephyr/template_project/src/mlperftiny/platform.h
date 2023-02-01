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

#ifndef APPS_MICROTVM_ZEPHYR_MLPERFTINY_PLATFORM_H_
#define APPS_MICROTVM_ZEPHYR_MLPERFTINY_PLATFORM_H_

#include <stdarg.h>
#include <stdint.h>
#include <tvm/runtime/crt/error_codes.h>
#include <unistd.h>

#define MODEL_KWS 1
#define MODEL_VWW 2
#define MODEL_AD 3
#define MODEL_IC 4

extern const unsigned char g_wakeup_sequence[];
extern size_t g_output_data_len;

#if TARGET_MODEL == 3
extern float* g_output_data;
#else
extern int8_t* g_output_data;
#endif

extern float g_quant_scale;
extern int8_t g_quant_zero;

#define TVM_UART_DEFAULT_BAUDRATE 115200

/*!
 * \brief Initialize TVM runtime.
 */
void TVMRuntimeInit();

/*!
 * \brief Run TVM inference.
 */
void TVMInfer(void* input_ptr);

/*!
 * \brief Quantize float to int8.
 * \param value Input data in float.
 * \param scale Quantization scale factor.
 * \param zero_point Quantization zero point.
 */
int8_t QuantizeFloatToInt8(float value, float scale, int zero_point);

/*!
 * \brief Read Uart Rx buffer.
 * \param data Pointer to read data.
 * \param data_size_bytes Read request size in bytes.
 *
 * \return Number of data read in bytes.
 */
char TVMPlatformUartRxRead();

/*!
 * \brief Write data in serial.
 * \param data Pointer to data to write.
 * \param size Size of data in bytes.
 *
 * \return Number of write in bytes.
 */
uint32_t TVMPlatformWriteSerial(const char* data, uint32_t size);

/*!
 * \brief Initialize Uart.
 * \param baudrate Desired UART baudrate.
 */
void TVMPlatformUARTInit(uint32_t baudrate = TVM_UART_DEFAULT_BAUDRATE);

#endif /* APPS_MICROTVM_ZEPHYR_MLPERFTINY_PLATFORM_H_ */
