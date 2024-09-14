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
#include <gtest/gtest.h>
#include <tvm/ffi/error.h>

namespace {

using namespace tvm::ffi;

void ThrowRuntimeError() { TVM_FFI_THROW(RuntimeError) << "test0"; }

TEST(Error, Traceback) {
  EXPECT_THROW(
      {
        try {
          ThrowRuntimeError();
        } catch (const Error& error) {
          EXPECT_EQ(error->message, "test0");
          EXPECT_EQ(error->kind, "RuntimeError");
          std::string what = error.what();
          EXPECT_NE(what.find("line"), std::string::npos);
          EXPECT_NE(what.find("ThrowRuntimeError"), std::string::npos);
          EXPECT_NE(what.find("RuntimeError: test0"), std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);
}

TEST(CheckError, Traceback) {
  EXPECT_THROW(
      {
        try {
          TVM_FFI_ICHECK_GT(2, 3);
        } catch (const Error& error) {
          EXPECT_EQ(error->kind, "InternalError");
          std::string what = error.what();
          EXPECT_NE(what.find("line"), std::string::npos);
          EXPECT_NE(what.find("2 > 3"), std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);
}
}  // namespace
