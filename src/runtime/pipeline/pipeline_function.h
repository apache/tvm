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

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "pipeline_data.h"

using namespace std;
using namespace tvm::runtime;
typedef vector<shared_ptr<RuntimeItem>> SHARED_RUNTIME_VEC;
typedef unordered_map<int, unordered_map<int, unordered_map<int, string>>> PIPELINE_CONF;
typedef unordered_map<int, unordered_map<string, string>> MOD_CONF;
typedef shared_ptr<TensorData> TENSOR_DATA;
typedef unordered_map<int, TENSOR_DATA> DLDATA_MAP;
typedef unordered_map<int, DLDATA_MAP> MOD_DLDATA_MAP;
typedef shared_ptr<MOD_DLDATA_MAP> MOD_DLDATA_MAP_PTR;

vector<Module> pipeline_get_graphRuntime(Array<Module> modules, const MOD_CONF& mod_conf);
size_t pipeline_init(Array<Module> modules, SHARED_RUNTIME_VEC* runtimes,
                     const PIPELINE_CONF& pipeline_conf, const MOD_CONF& mod_conf);
void pipeline_run(const SHARED_RUNTIME_VEC& runtimes, const MOD_DLDATA_MAP_PTR indxInputs);
inline void pipeline_queue_push(QUEUE* queue, vector<shared_ptr<OutputData>>* outputs);
bool pipeline_queue_poll(QUEUE* queue, RuntimeData* runtimeData);
bool pipeline_poll(vector<NDArray>* output, const SHARED_RUNTIME_VEC& runtimes,
                   const bool bSync = false);
void pipeline_stop(const SHARED_RUNTIME_VEC& runtimes);
void pipeline_setinput(MOD_DLDATA_MAP_PTR input_int_map, const int index, const DLTensor* data_in,
                       const int modIndx);

#endif  // TVM_RUNTIME_PIPELINE_PIPELINE_FUNCTION_H_
