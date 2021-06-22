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
#if ETHOSN_SUPPORT_LIBRARY_VERSION_PATCH == 2
#define _ETHOSN_API_VERSION_ 2102
#else
#ifndef ETHOSN_API_VERSION
#define _ETHOSN_API_VERSION_ 2011
#elif ~(~ETHOSN_API_VERSION + 0) == 0 && ~(~ETHOSN_API_VERSION + 1) == 1
#define _ETHOSN_API_VERSION_ 2011
#else
#define _ETHOSN_API_VERSION_ ETHOSN_API_VERSION
#endif
#endif

#endif  // TVM_RELAY_BACKEND_CONTRIB_ETHOSN_ETHOSN_API_VERSION_H_
