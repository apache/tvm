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

/* Explicitly declare posix_memalign function */
#if _POSIX_C_SOURCE < 200112L
#undef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

/*! Support low-level debugging in MISRA-C runtime */
#define TVM_CRT_DEBUG 0

/*! Maximum supported dimension in NDArray */
#define TVM_CRT_MAX_NDIM 6
/*! Maximum supported arguments in generated functions */
#define TVM_CRT_MAX_ARGS 10

/*! Maximum inputs in a GraphRuntimeNode */
#define GRAPH_RUNTIME_NODE_MAX_INPUTS 300
/*! Maximum supported contexts in a GraphRuntime */
#define GRAPH_RUNTIME_MAX_CONTEXTS 1
/*! Maximum supported nodes in a GraphRuntime */
#define GRAPH_RUNTIME_MAX_NODES 400
/*! Maximum input nodes in a GraphRuntime */
#define GRAPH_RUNTIME_MAX_INPUT_NODES 300
/*! Maximum nodes in a GraphRuntime for quick entry indexing */
#define GRAPH_RUNTIME_MAX_NODE_ROW_PTR 300
/*! Maximum output entries in a GraphRuntime */
#define GRAPH_RUNTIME_MAX_OUTPUTS 300

#include "../../src/runtime/crt/crt_runtime_api.c"
#include "../../src/runtime/crt/crt_backend_api.c"
#include "../../src/runtime/crt/graph_runtime.c"
#include "../../src/runtime/crt/load_json.c"
#include "../../src/runtime/crt/ndarray.c"

