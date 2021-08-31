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

#ifndef TVM_APPS_MICROTVM_ZEPHYR_AOT_DEMO_INCLUDE_ZEPHYR_UART_H_
#define TVM_APPS_MICROTVM_ZEPHYR_AOT_DEMO_INCLUDE_ZEPHYR_UART_H_

#include <stdint.h>

// Used to read data from the UART.

/*!
 * \brief Read Uart Rx buffer.
 * \param data Pointer to read data.
 * \param data_size_bytes Read request size in bytes.
 *
 * \return Number of data read in bytes.
 */
uint32_t TVMPlatformUartRxRead(uint8_t* data, uint32_t data_size_bytes);


/*!
 * \brief Initialize Uart.
 */
void TVMPlatformUARTInit();

void TVMLogf(const char* msg, ...);

#endif /* TVM_APPS_MICROTVM_ZEPHYR_AOT_DEMO_INCLUDE_ZEPHYR_UART_H_ */
