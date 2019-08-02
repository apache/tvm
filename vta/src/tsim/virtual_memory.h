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
 *  Copyright (c) 2019 by Contributors
 * \file virtual_memory.h
 * \brief The virtual memory manager for TSIM driver.
 */

#ifndef VTA_TSIM_VIRTUAL_MEMORY_H_
#define VTA_TSIM_VIRTUAL_MEMORY_H_

#include <cstdint>

enum VMemCopyType {
  kVirtualMemCopyFromHost = 0,
  kVirtualMemCopyToHost = 1
};

#define VTA_VMEM_PAGEFILE "/tmp/vta_tsim_vmem_pagefile.sys"

/*!
 * \brief Virtual memory based memory allocation
 */
void * vmalloc(uint64_t size);

/*!
 * \brief Virtual memory based memory release
 */
void vfree(void * ptr);

/*!
 * \brief Memory copy between virtual and logical
 */
void vmemcpy(void * dst, const void * src, uint64_t size, VMemCopyType dir);

/*!
 * \brief Map virtual address to logical address
 */
void * vmem_get_log_addr(uint64_t vaddr);

#endif /* VTA_TSIM_VIRTUAL_MEMORY_H_ */
