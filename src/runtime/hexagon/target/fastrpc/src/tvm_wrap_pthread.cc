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
 * Implement a wrapper around pthread_create that sets the thread stack
 * size to a chosen value.
 *
 * TVM runtime uses std::thread, but the C++ standard does not provide
 * any means of controlling thread attributes (like stack size). Because
 * of that, any thread created by the std::thread constructor will use
 * default attributes. The default stack size for a thread in QuRT is 16kB.
 * This has proven to be insufficient in the past, so we need to increase
 * it.
 * When libtvm_runtime.so is linked, a linker flag --wrap=pthread_create
 * is used, which causes the linker to rename all uses of pthread_create
 * with references to __wrap_pthread_create. This file implements the
 * __wrap function to set the larger stack size and call the actual
 * pthread_create. The call to pthread_create here must not be renamed,
 * so this function cannot be included in the TVM runtime binary.
 * Instead, it's implemented in a separate shared library.
 */

#include <pthread.h>

#include "HAP_farf.h"

static constexpr size_t kThreadStackSize = 128 * 1024;  // 128kB

// Make sure the function has C linkage.
extern "C" {
int __wrap_pthread_create(pthread_t* restrict thread,
                          const pthread_attr_t* restrict attr,
                          void* (*start)(void*), void* restrict arg);
}

int __wrap_pthread_create(pthread_t* restrict thread,
                          const pthread_attr_t* restrict attr,
                          void* (*start)(void*), void* restrict arg) {
  pthread_attr_t def_attr;
  if (attr == nullptr) {
    if (int rc = pthread_attr_init(&def_attr)) {
      FARF(ERROR, "pthread_attr_init failed: rc=%08x", rc);
      return rc;
    }
    if (int rc = pthread_attr_setstacksize(&def_attr, kThreadStackSize)) {
      FARF(ERROR, "pthread_attr_setstacksize failed: rc=%08x", rc);
      return rc;
    }
    attr = &def_attr;
  }
  size_t stack_size = 0;
  if (int rc = pthread_attr_getstacksize(attr, &stack_size)) {
    FARF(ERROR, "pthread_attr_setstacksize failed: rc=%08x", rc);
    return rc;
  }
  FARF(ALWAYS, "launching thread with stack_size=%zu", stack_size);
  int t = pthread_create(thread, attr, start, arg);
  if (int rc = pthread_attr_destroy(&def_attr)) {
    FARF(ERROR, "pthread_attr_destroy failed (after pthread_create): rc=%08x",
         rc);
  }
  return t;
}
