/*
Copyright 2020 EEMBC and The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
This file is a modified version of the original EEMBC implementation of ee_lib.
The file name has been changed and some functions removed.
==============================================================================*/

/// \file
/// \brief Internally-implemented methods required to perform inference.

#include <stddef.h>
#include <stdint.h>

#ifndef MLPERF_TINY_V0_1_API_INTERNALLY_IMPLEMENTED_H_
#define MLPERF_TINY_V0_1_API_INTERNALLY_IMPLEMENTED_H_

#define EE_MONITOR_VERSION "2.2.0"
#define EE_FW_VERSION "ULPMark for tinyML Firmware V0.0.1"

/* Version 1.0 of the benchmark only supports these models */
#define EE_MODEL_VERSION_KWS01 "kws01"
#define EE_MODEL_VERSION_VWW01 "vww01"
#define EE_MODEL_VERSION_AD01 "ad01"
#define EE_MODEL_VERSION_IC01 "ic01"

typedef enum { EE_ARG_CLAIMED, EE_ARG_UNCLAIMED } arg_claimed_t;
typedef enum { EE_STATUS_OK = 0, EE_STATUS_ERROR } ee_status_t;

#define EE_DEVICE_NAME "dut"

#define EE_CMD_SIZE 80u
#define EE_CMD_DELIMITER " "
#define EE_CMD_TERMINATOR '%'

#define EE_CMD_NAME "name"
#define EE_CMD_TIMESTAMP "timestamp"

#define EE_MSG_READY "m-ready\r\n"
#define EE_MSG_INIT_DONE "m-init-done\r\n"
#define EE_MSG_NAME "m-name-%s-[%s]\r\n"

#define EE_ERR_CMD "e-[Unknown command: %s]\r\n"

void ee_serial_callback(char);
void ee_serial_command_parser_callback(char*);
void ee_benchmark_initialize(void);
long ee_hexdec(char*);
void ee_infer(size_t n, size_t n_warmup);
size_t ee_get_buffer(uint8_t* buffer, size_t max_len);
arg_claimed_t ee_buffer_parse(char* command);
arg_claimed_t ee_profile_parse(char* command);
uint8_t* ee_get_buffer_pointer();

#endif /* MLPERF_TINY_V0_1_API_INTERNALLY_IMPLEMENTED_H_ */
