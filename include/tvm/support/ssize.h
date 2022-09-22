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
 * \file ssize.h
 * \brief this file aims to define ssize_t for Windows platform
 */

#ifndef TVM_SUPPORT_SSIZE_H_
#define TVM_SUPPORT_SSIZE_H_

#if defined(_MSC_VER)
#if defined(_WIN32)
using ssize_t = int32_t;
#else
using ssize_t = int64_t;
#endif
#endif

#endif  // TVM_SUPPORT_SSIZE_H_
