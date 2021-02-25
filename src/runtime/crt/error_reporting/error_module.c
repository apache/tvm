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
 * \file error_module.c
 * \brief Defines an error module used by RPC.
 */

#include <tvm/runtime/crt/error_reporting/error_module.h>
#include <checksum.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

bool ErrorModuleIsValid(ErrorModule* error_ptr) {
  if ((error_ptr->magic_num == kErrorModuleMagicNumber) &&
       ErrorModuleIsCRCValid(error_ptr)) {
    return true;
  }
  return false;
}

bool ErrorModuleIsCRCValid(ErrorModule* error_ptr) {
  uint16_t crc = ErrorModuleCalculateCRC(error_ptr);
  if (crc == error_ptr->crc) return true;
  return false;
}

void ErrorModuleSetError(ErrorModule* error_ptr, error_source_t source, uint16_t reason) {
  error_ptr->magic_num = kErrorModuleMagicNumber;
  error_ptr->source = source;
  error_ptr->reason = reason;
  error_ptr->crc = ErrorModuleCalculateCRC(error_ptr);
}

uint16_t ErrorModuleCalculateCRC(ErrorModule* error_ptr) {
  uint8_t message[16];
  message[0] = error_ptr->magic_num;
  message[1] = error_ptr->source;
  message[2] = (uint8_t)((error_ptr->reason & 0xFF00) >> 8);
  message[3] = (uint8_t)(error_ptr->reason & 0x00FF);
  return crc_ccitt_1d0f(message, 4);
}

uint8_t ErrorModuleGenerateMessage(ErrorModule* error_ptr, uint8_t* message) {
  size_t num_bytes = 0;
  message[0] = error_ptr->magic_num;
  message[1] = error_ptr->source;
  uint16_t reason = error_ptr->reason;
  message[2] = (uint8_t)((reason & 0xFF00) >> 8);
  message[3] = (uint8_t)(reason & 0x00FF);
  uint16_t crc_16 = ErrorModuleCalculateCRC(error_ptr);
  message[4] = (uint8_t)((crc_16 & 0xFF00) >> 8);
  message[5] = (uint8_t)(crc_16 & 0x00FF);
  num_bytes += 6;
  return num_bytes;
}

#ifdef __cplusplus
}
#endif
