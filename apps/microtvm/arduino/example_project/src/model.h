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

#ifndef IMPLEMENTATION_H_
#define IMPLEMENTATION_H_

#define WORKSPACE_SIZE $workspace_size_bytes

#ifdef __cplusplus
extern "C" {
#endif

void TVMInitialize();

// TODO template these void* values once MLF format has input and output data
void TVMExecute(void* input_data, void* output_data);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IMPLEMENTATION_H_
