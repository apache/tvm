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

#ifndef TVM_APPS_MICROTVM_ZEPHYR_HOST_DRIVEN_SEMIHOST_H_
#define TVM_APPS_MICROTVM_ZEPHYR_HOST_DRIVEN_SEMIHOST_H_

#include <kernel.h>
#include <unistd.h>
#include <zephyr.h>

void init_semihosting();

ssize_t semihost_read(uint8_t* data, size_t size);

ssize_t semihost_write(void* unused_context, const uint8_t* data, size_t size);

#endif /* TVM_APPS_MICROTVM_ZEPHYR_HOST_DRIVEN_SEMIHOST_H_ */
