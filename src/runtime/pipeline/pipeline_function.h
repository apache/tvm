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
#ifndef TVM_RUNTIME_PIPELINE_PIPELINE_FUNCTION_H_
#define TVM_RUNTIME_PIPELINE_PIPELINE_FUNCTION_H_
#include <memory>
#include <unordered_map>
#include <vector>

#include "pipeline_data.h"

using namespace std;
using namespace tvm::runtime;
typedef vector<shared_ptr<RuntimeItem>> SHARED_RUNTIME_VEC;
typedef unordered_map<int, unordered_map<int, unordered_map<int, int>>> PIPELINE_CONF;

void pipeline_init(Array<Module> graphRuntimes, SHARED_RUNTIME_VEC* runtimes,
                   PIPELINE_CONF* pipeline_conf);
void pipeline_run(const SHARED_RUNTIME_VEC& runtimes);
inline void pipeline_queue_push(QUEUE* queue, vector<shared_ptr<OutputData>>* outputs);
bool pipeline_queue_poll(QUEUE* queue, RuntimeData* runtimeData);
bool pipeline_poll(vector<NDArray>* output, const SHARED_RUNTIME_VEC& runtimes,
                   const bool bSync = false);
void pipeline_stop(const SHARED_RUNTIME_VEC& runtimes);

#endif  // TVM_RUNTIME_PIPELINE_PIPELINE_FUNCTION_H_
