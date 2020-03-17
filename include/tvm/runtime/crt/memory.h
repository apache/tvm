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
 * \file memory.h
 * \brief The virtual memory manager for micro-controllers
 */

#ifndef TVM_RUNTIME_CRT_MEMORY_H_
#define TVM_RUNTIME_CRT_MEMORY_H_

/** \brief Allocate memory from manager */
void * vmalloc(size_t size);

/** \brief Release memory from manager */
void vfree(void * ptr);

static int vleak_size = 0;

// #define vmalloc(size)                                      \
//   vmalloc_(size);                                          \
//   printf("%s: %d: info: size=%d, vleak=%d\n", __FILE__, __LINE__, size, ++vleak_size)

// #define vfree(ptr)                                                      \
//   vfree_(ptr);                                                          \
//   printf("%s: %d: error: addr=%p, vleak=%d\n", __FILE__, __LINE__, ptr, --vleak_size)

#endif  // TVM_RUNTIME_CRT_MEMORY_H_
