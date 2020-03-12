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
 *
 * \file cma_api.h
 * \brief API for contigous memory allocation driver.
 */

#ifndef VTA_DE10NANO_CMA_API_H_
#define VTA_DE10NANO_CMA_API_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/**
 * \brief Initialize CMA api (basically perform open() syscall).
 *
 * \return Returns 0 on SUCCESS. On FAILURE returns -1 and errno is set
 * accordingly.
 */
int cma_init(void);


/**
 * \brief Release CMA api (basically perform close() syscall).
 *
 * \return Returns 0 on SUCCESS. On FAILURE returns -1 and errno is set
 * accordingly.
 */
int cma_release(void);


/**
 * \brief Allocate cached, physically contigous memory.
 *
 * \param size Size in bytes.
 *
 * \return Returns NULL on FAILURE. Otherwise pointer to valid userspace
 * memory.
 */
void *cma_alloc_cached(size_t size);


/**
 * \brief Allocate noncached, physically contigous memory.
 *
 * \param size Size in bytes.
 *
 * \return Returns NULL on FAILURE. Otherwise pointer to valid userspace
 * memory.
 */
void *cma_alloc_noncached(size_t size);


/**
 * \brief Release physically contigous memory.
 *
 * \param mem Pointer to previously allocated contiguous memory.
 *
 * \return Returns 0 on SUCCESS, -1 on FAILURE.
 */
int cma_free(void *mem);


/**
 * \brief Get physical memory of cma memory block (should be used for DMA).
 *
 * \param mem Pointer to previously allocated contiguous memory.
 *
 * \return Returns address on SUCCESS, 0 on FAILURE.
 */
unsigned cma_get_phy_addr(void *mem);


#ifdef __cplusplus
}
#endif
#endif  // VTA_DE10NANO_CMA_API_H_
