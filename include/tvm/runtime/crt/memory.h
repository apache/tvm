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
 * \file tvm/runtime/crt/memory.h
 * \brief The virtual memory manager for micro-controllers
 */

#ifndef TVM_RUNTIME_CRT_MEMORY_H_
#define TVM_RUNTIME_CRT_MEMORY_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>

extern int vleak_size;

/*!
 * \brief Allocate memory from manager
 * \param size The size of memory
 * \return The virtual address
 */
void* vmalloc(size_t size);

/*!
 * \brief Reallocate memory from manager
 * \param ptr The pointer to the memory area to be reallocated
 * \param size The size of memory
 * \return The virtual address
 */
void* vrealloc(void* ptr, size_t size);

/*!
 * \brief Free the memory.
 * \param ptr The pointer to the memory to deallocate
 * \return The virtual address
 */
void vfree(void* ptr);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TVM_RUNTIME_CRT_MEMORY_H_
