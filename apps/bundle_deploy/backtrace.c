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

#define _GNU_SOURCE
#include "backtrace.h"

#include <dlfcn.h>
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

const char* g_argv0 = NULL;

void tvm_platform_abort_backtrace() {
  void* trace[200];
  int nptrs = backtrace(trace, sizeof(trace) / sizeof(void*));
  fprintf(stderr, "backtrace: %d\n", nptrs);
  if (nptrs < 0) {
    perror("backtracing");
  } else {
    backtrace_symbols_fd(trace, nptrs, STDOUT_FILENO);

    char cmd_buf[1024];
    for (int i = 0; i < nptrs; i++) {
      Dl_info info;
      if (dladdr(trace[i], &info)) {
        fprintf(stderr, "symbol %d: %s %s %p (%p)\n", i, info.dli_sname, info.dli_fname,
                info.dli_fbase, (void*)(trace[i] - info.dli_fbase));
        snprintf(cmd_buf, sizeof(cmd_buf), "addr2line --exe=%s -p -i -a -f %p", g_argv0,
                 (void*)(trace[i] - info.dli_fbase));
        int result = system(cmd_buf);
        if (result < 0) {
          perror("invoking backtrace command");
        }
      } else {
        fprintf(stderr, "symbol %d: %p (unmapped)\n", i, trace[i]);
      }
    }
  }
}
