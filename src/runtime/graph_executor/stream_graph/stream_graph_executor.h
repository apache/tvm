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


#ifndef TVM_RUNTIME_GRAPH_EXECUTOR_STREAM_GRAPH_STREAM_GRAPH_EXECUTOR_H_
#define TVM_RUNTIME_GRAPH_EXECUTOR_STREAM_GRAPH_STREAM_GRAPH_EXECUTOR_H_
#include "../graph_executor.h"
namespace tvm {
namespace runtime {

class StreamGraphExecutor : public GraphExecutor {
 public:

  virtual void StartCapture() = 0;

  virtual void RunGraph() = 0;

  virtual void EndCapture() = 0;

  virtual PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) = 0;

};

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_GRAPH_EXECUTOR_STREAM_GRAPH_STREAM_GRAPH_EXECUTOR_H_
