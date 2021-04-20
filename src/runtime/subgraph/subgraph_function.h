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
#ifndef TVM_RUNTIME_SUBGRAPH_SUBGRAPH_FUNCTION_H_
#define TVM_RUNTIME_SUBGRAPH_SUBGRAPH_FUNCTION_H_
#include "subgraph_data.h"
#include <memory>
#include <vector>

using namespace std;
using namespace tvm::runtime;
typedef vector<shared_ptr<RuntimeItem>> SHARED_RUNTIME_VEC;

void subgraph_init(Array<Module> graphRuntimes, SHARED_RUNTIME_VEC* runtimes);
void subgraph_run(const SHARED_RUNTIME_VEC& runtimes, bool synch = false);
inline void subgraph_queue_push(QUEUE* queue, Array<NDArray> arrays);
bool subgraph_queue_poll(QUEUE* queue, RuntimeData* runtimeData);
bool subgraph_poll(vector<NDArray>* output, const SHARED_RUNTIME_VEC& runtimes,
                   const bool sync = false);
void subgraph_stop(const SHARED_RUNTIME_VEC &runtimes);

#endif  // TVM_RUNTIME_SUBGRAPH_SUBGRAPH_FUNCTION_H_
