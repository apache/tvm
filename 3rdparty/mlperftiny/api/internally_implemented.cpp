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
The file name has been changed and some functions removed. Malloc has been
replaced by a fixed-size array.
==============================================================================*/
/// \file
/// \brief Internally-implemented methods required to perform inference.

#include "internally_implemented.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "submitter_implemented.h"

// Command buffer (incoming commands from host)
char volatile g_cmd_buf[EE_CMD_SIZE + 1];
size_t volatile g_cmd_pos = 0u;

// Generic buffer to db input.
uint8_t gp_buff[MAX_DB_INPUT_SIZE];
size_t g_buff_size = 0u;
size_t g_buff_pos = 0u;

/**
 * Since the serial port ISR may be connected before the loop is ready, this
 * flag turns off the parser until the main routine is ready.
 */
bool g_state_parser_enabled = false;

/**
 * This function assembles a command string from the UART. It should be called
 * from the UART ISR for each new character received. When the parser sees the
 * termination character, the user-defined th_command_ready() command is called.
 * It is up to the application to then dispatch this command outside the ISR
 * as soon as possible by calling ee_serial_command_parser_callback(), below.
 */
void ee_serial_callback(char c) {
  if (c == EE_CMD_TERMINATOR) {
    g_cmd_buf[g_cmd_pos] = (char)0;
    th_command_ready(g_cmd_buf);
    g_cmd_pos = 0;
  } else {
    g_cmd_buf[g_cmd_pos] = c;
    g_cmd_pos = g_cmd_pos >= EE_CMD_SIZE ? EE_CMD_SIZE : g_cmd_pos + 1;
  }
}

/**
 * This is the minimal parser required to test the monitor; profile-specific
 * commands are handled by whatever profile is compiled into the firmware.
 *
 * The most basic commands are:
 *
 * name             Print m-name-NAME, where NAME defines the intent of the f/w
 * timestamp        Generate a signal used for timestamping by the framework
 */
/*@-mustfreefresh*/
/*@-nullpass*/
void ee_serial_command_parser_callback(char *p_command) {
  char *tok;

  if (g_state_parser_enabled != true) {
    return;
  }

  tok = strtok(p_command, EE_CMD_DELIMITER);

  if (strncmp(tok, EE_CMD_NAME, EE_CMD_SIZE) == 0) {
    th_printf(EE_MSG_NAME, EE_DEVICE_NAME, TH_VENDOR_NAME_STRING);
  } else if (strncmp(tok, EE_CMD_TIMESTAMP, EE_CMD_SIZE) == 0) {
    th_timestamp();
  } else if (ee_profile_parse(tok) == EE_ARG_CLAIMED) {
  } else {
    th_printf(EE_ERR_CMD, tok);
  }

  th_printf(EE_MSG_READY);
}

/**
 * Perform the basic setup.
 */
void ee_benchmark_initialize(void) {
  th_serialport_initialize();
  th_timestamp_initialize();
  th_final_initialize();
  th_printf(EE_MSG_INIT_DONE);
  // Enable the command parser here (the callback is connected)
  g_state_parser_enabled = true;
  // At this point, the serial monitor should be up and running,
  th_printf(EE_MSG_READY);
}

arg_claimed_t ee_profile_parse(char *command) {
  char *p_next; /* strtok already primed from ee_main.c */

  if (strncmp(command, "profile", EE_CMD_SIZE) == 0) {
    th_printf("m-profile-[%s]\r\n", EE_FW_VERSION);
    th_printf("m-model-[%s]\r\n", TH_MODEL_VERSION);
  } else if (strncmp(command, "help", EE_CMD_SIZE) == 0) {
    th_printf("%s\r\n", EE_FW_VERSION);
    th_printf("\r\n");
    /* These are the three common functions for all IoTConnect f/w. */
    th_printf("help         : Print this information\r\n");
    th_printf("name         : Print the name of the device\r\n");
    th_printf("timestsamp   : Generate a timetsamp\r\n");
    /* These are profile-specific commands. */
    th_printf("db SUBCMD    : Manipulate a generic byte buffer\r\n");
    th_printf("  load N     : Allocate N bytes and set load counter\r\n");
    th_printf("  db HH[HH]* : Load 8-bit hex byte(s) until N bytes\r\n");
    th_printf("  print [N=16] [offset=0]\r\n");
    th_printf("             : Print N bytes at offset as hex\r\n");
    th_printf(
        "infer N [W=0]: Load input, execute N inferences after W warmup "
        "loops\r\n");
    th_printf("results      : Return the result fp32 vector\r\n");
  } else if (ee_buffer_parse(command) == EE_ARG_CLAIMED) {
  } else if (strncmp(command, "infer", EE_CMD_SIZE) == 0) {
    size_t n = 1;
    size_t w = 10;
    int i;

    /* Check for inference iterations */
    p_next = strtok(NULL, EE_CMD_DELIMITER);
    if (p_next) {
      i = atoi(p_next);
      if (i <= 0) {
        th_printf("e-[Inference iterations must be >0]\r\n");
        return EE_ARG_CLAIMED;
      }
      n = (size_t)i;
      /* Check for warmup iterations */
      p_next = strtok(NULL, EE_CMD_DELIMITER);
      if (p_next) {
        i = atoi(p_next);
        if (i < 0) {
          th_printf("e-[Inference warmup must be >=0]\r\n");
          return EE_ARG_CLAIMED;
        }
        w = (size_t)i;
      }
    }

    ee_infer(n, w);
  } else if (strncmp(command, "results", EE_CMD_SIZE) == 0) {
    th_results();
  } else {
    return EE_ARG_UNCLAIMED;
  }
  return EE_ARG_CLAIMED;
}

/**
 * Inference without feature engineering. The inpput tensor is expected to
 * have been loaded from the buffer via the th_load_tensor() function, which in
 * turn was loaded from the interface via `db` commands.
 *
 * For testing, you can pre-load known-good data into the buffer during the
 * th_final_initialize() function.
 *
 */
void ee_infer(size_t n, size_t n_warmup) {
  th_load_tensor(); /* if necessary */
  th_printf("m-warmup-start-%d\r\n", n_warmup);
  while (n_warmup-- > 0) {
    th_infer(); /* call the API inference function */
  }
  th_printf("m-warmup-done\r\n");
  th_printf("m-infer-start-%d\r\n", n);
  th_timestamp();
  th_pre();
  while (n-- > 0) {
    th_infer(); /* call the API inference function */
  }
  th_post();
  th_timestamp();
  th_printf("m-infer-done\r\n");
  th_results();
}

arg_claimed_t ee_buffer_parse(char *p_command) {
  char *p_next;

  if (strncmp(p_command, "db", EE_CMD_SIZE) != 0) {
    return EE_ARG_UNCLAIMED;
  }

  p_next = strtok(NULL, EE_CMD_DELIMITER);

  if (p_next == NULL) {
    th_printf("e-[Command 'db' requires a subcommand]\r\n");
  } else if (strncmp(p_next, "load", EE_CMD_SIZE) == 0) {
    p_next = strtok(NULL, EE_CMD_DELIMITER);

    if (p_next == NULL) {
      th_printf("e-[Command 'db load' requires the # of bytes]\r\n");
    } else {
      g_buff_size = (size_t)atoi(p_next);
      if (g_buff_size == 0) {
        th_printf("e-[Command 'db load' must be >0 bytes]\r\n");
      } else {
        g_buff_pos = 0;
        if (g_buff_size > MAX_DB_INPUT_SIZE) {
          th_printf("Supplied buffer size %d exceeds maximum of %d\n",
                    g_buff_size, MAX_DB_INPUT_SIZE);
        } else {
          th_printf("m-[Expecting %d bytes]\r\n", g_buff_size);
        }
      }
    }
  } else if (strncmp(p_next, "print", EE_CMD_SIZE) == 0) {
    size_t i = 0;
    const size_t max = 8;
    for (; i < g_buff_size; ++i) {
    if ((i + max) % max == 0 || i == 0) {
        th_printf("m-buffer-");
    }
    /* N.B. Not every `printf` supports the spacing prefix! */
    th_printf("%02x", gp_buff[i]);
    if (((i + 1) % max == 0) || ((i + 1) == g_buff_size)) {
        th_printf("\r\n");
    } else {
        th_printf("-");
    }
    }
    if (i % max != 0) {
    th_printf("\r\n");
    }
  } else {
    size_t numbytes;
    char test[3];
    long res;

    /* Two hexdigits per byte */
    numbytes = th_strnlen(p_next, EE_CMD_SIZE);

    if ((numbytes & 1) != 0) {
      th_printf("e-[Insufficent number of hex digits]\r\n");
      return EE_ARG_CLAIMED;
    }
    test[2] = 0;
    for (size_t i = 0; i < numbytes;) {
      test[0] = p_next[i++];
      test[1] = p_next[i++];
      res = ee_hexdec(test);
      if (res < 0) {
        th_printf("e-[Invalid hex digit '%s']\r\n", test);
        return EE_ARG_CLAIMED;
      } else {
        gp_buff[g_buff_pos] = (uint8_t)res;
        g_buff_pos++;
        if (g_buff_pos == g_buff_size) {
          th_printf("m-load-done\r\n");
          /* Disregard the remainder of the digits when done. */
          return EE_ARG_CLAIMED;
        }
      }
    }
  }
  return EE_ARG_CLAIMED;
}

/**
 * @brief convert a hexidecimal string to a signed long
 * will not produce or process negative numbers except
 * to signal error.
 *
 * @param hex without decoration, case insensitive.
 *
 * @return -1 on error, or result (max (sizeof(long)*8)-1 bits)
 *
 */
long ee_hexdec(char *hex) {
  char c;
  long dec = 0;
  long ret = 0;

  while (*hex && ret >= 0) {
    c = *hex++;
    if (c >= '0' && c <= '9') {
      dec = c - '0';
    } else if (c >= 'a' && c <= 'f') {
      dec = c - 'a' + 10;
    } else if (c >= 'A' && c <= 'F') {
      dec = c - 'A' + 10;
    } else {
      return -1;
    }
    ret = (ret << 4) + dec;
  }
  return ret;
}

/**
 * @brief get the buffer resulting from the last db command. Returns length 0
 * if the db command has not been used yet.
 *
 * @param buffer to fill with bytes from internal buffer filled by db commands.
 * @param maximum number of bytes to copy into provided buffer. This is
 * typically the length of the provided buffer.
 *
 * @return number of bytes copied from internal buffer.
 *
 */
size_t ee_get_buffer(uint8_t* buffer, size_t max_len) {
  int len = max_len < g_buff_pos ? max_len : g_buff_pos;
  if (buffer != nullptr) {
    memcpy(buffer, gp_buff, len * sizeof(uint8_t));
  }
  return len;
}

uint8_t* ee_get_buffer_pointer() { return gp_buff; }
