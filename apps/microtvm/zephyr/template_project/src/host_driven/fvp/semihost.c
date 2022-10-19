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

/*
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "semihost.h"

int32_t stdout_fd;
int32_t stdin_fd;

uint32_t semihost_cmd(uint32_t opcode, void* arg) {
  uint32_t ret_val;
  __asm__ volatile(
      "mov r0, %[opcode]\n\t"
      "mov r1, %[arg]\n\t"
      "bkpt #0xab\n\r"
      "mov %[ret_val], r0"
      : [ ret_val ] "=r"(ret_val)
      : [ opcode ] "r"(opcode), [ arg ] "r"(arg)
      : "r1", "memory");

  return ret_val;
}

int32_t stdout_fd;
int32_t stdin_fd;

void init_semihosting() {
  // https://github.com/ARM-software/abi-aa/blob/main/semihosting/semihosting.rst#sys-open-0x01
  struct {
    const char* file_name;
    uint32_t mode;
    uint32_t file_name_len;
  } params;
  params.file_name = ":tt";
  params.mode = 5;  // "wb"
  params.file_name_len = 3;
  stdout_fd = semihost_cmd(0x01, &params);

  params.mode = 0;
  stdin_fd = semihost_cmd(0x01, &params);
}

ssize_t semihost_read(uint8_t* data, size_t size) {
  struct {
    uint32_t file_handle;
    const uint8_t* data;
    uint32_t size;
  } read_req;
  read_req.file_handle = stdin_fd;
  read_req.data = data;
  read_req.size = size;
  uint32_t ret_val = semihost_cmd(0x06, &read_req);
  return size - ret_val;
}

ssize_t semihost_write(void* unused_context, const uint8_t* data, size_t size) {
  struct {
    uint32_t file_handle;
    const uint8_t* data;
    uint32_t size;
  } write_req;
  write_req.file_handle = stdout_fd;
  write_req.data = data;
  write_req.size = size;
  uint32_t ret_val = semihost_cmd(0x05, &write_req);
  return size - ret_val;
}
