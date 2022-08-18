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
 * \file src/relay/backend/contrib/ethosn/ethosn_api_version.h
 * \brief Arm(R) Ethos(TM)-N versioning
 */
#include "ethosn_support_library/Support.hpp"

#ifndef TVM_RELAY_BACKEND_CONTRIB_ETHOSN_ETHOSN_API_VERSION_H_
#define TVM_RELAY_BACKEND_CONTRIB_ETHOSN_ETHOSN_API_VERSION_H_

/*!
 * \brief To be used as a temperory switch to ensure
 * compatibility with the previous version of the api
 * while needed e.g. by docker images. Can be removed
 * along with associated compatibility measures when no
 * longer necessary.
 */

#if ETHOSN_SUPPORT_LIBRARY_VERSION_MAJOR == 3 && ETHOSN_SUPPORT_LIBRARY_VERSION_MINOR == 0 && \
    ETHOSN_SUPPORT_LIBRARY_VERSION_PATCH == 1
#define _ETHOSN_API_VERSION_ 2205
#elif ETHOSN_SUPPORT_LIBRARY_VERSION_MAJOR == 3 && ETHOSN_SUPPORT_LIBRARY_VERSION_MINOR == 0 && \
    ETHOSN_SUPPORT_LIBRARY_VERSION_PATCH == 0
#define _ETHOSN_API_VERSION_ 2111
#else
#if ETHOSN_SUPPORT_LIBRARY_VERSION_MAJOR == 1 && ETHOSN_SUPPORT_LIBRARY_VERSION_MINOR == 1 && \
    ETHOSN_SUPPORT_LIBRARY_VERSION_PATCH == 0
#define _ETHOSN_API_VERSION_ 2108
#else
#ifndef ETHOSN_API_VERSION
#define _ETHOSN_API_VERSION_ 2102
#elif ~(~ETHOSN_API_VERSION + 0) == 0 && ~(~ETHOSN_API_VERSION + 1) == 1
#define _ETHOSN_API_VERSION_ 2102
#else
#define _ETHOSN_API_VERSION_ ETHOSN_API_VERSION
#endif
#endif
#endif

#endif  // TVM_RELAY_BACKEND_CONTRIB_ETHOSN_ETHOSN_API_VERSION_H_
