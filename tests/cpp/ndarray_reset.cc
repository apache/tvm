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

#include <bits/stdc++.h>
#include <gtest/gtest.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/ndarray.h>


using namespace tvm::runtime;

TEST(NDArray,MemoryManagement){
  auto handle = NDArray::Empty({200,20,20,20},DLDataType{kDLFloat,32,4},DLDevice{kDLCPU,0});
  EXPECT_EQ(handle.use_count(),1);
  auto handle_view = handle.CreateView({20,20,20,20},DLDataType{kDLFloat,32,4});
  EXPECT_EQ(handle.use_count(),2);
  handle_view.reset();
  EXPECT_EQ(handle.use_count(),1);
  handle.reset();
  EXPECT_EQ(handle.use_count(),0);
}

TEST(NDArray,ContainerReset){
  std::vector<NDArray> t;
  t.resize(20);
  for(const auto& h:t){
    EXPECT_EQ(h.defined(),false);
  }
}