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

#if defined(__hexagon__)

#include <errno.h>
#include <stdlib.h>

extern "C" {
int posix_memalign(void** memptr, size_t alignment, size_t size)
    __attribute__((nothrow));
}

__attribute__((nothrow)) int posix_memalign(void** memptr, size_t alignment,
                                            size_t size) {
  if (void* p = memalign(alignment, size)) {
    *memptr = p;
    return 0;
  }

  return ENOMEM;
}
#endif
