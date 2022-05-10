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
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <vector>

#include "../src/support/utils.h"

// Workaround for missing symbol in some QuRT builds
#include "qurt.h"
extern "C" {
__attribute__((weak)) int pthread_key_delete(pthread_key_t key) {
  return qurt_tls_delete_key((int)key);
}
}

namespace tvm {
namespace runtime {
namespace hexagon {

TVM_REGISTER_GLOBAL("hexagon.run_all_tests").set_body([](TVMArgs args, TVMRetValue* rv) {
  // gtest args are passed into this packed func as a singular string
  // split gtest args using <space> delimiter and build argument vector
  std::vector<std::string> parsed_args = tvm::support::Split(args[0], ' ');
  std::vector<char*> argv;

  // add executable name
  argv.push_back(const_cast<char*>("hexagon_run_all_tests"));

  // add parsed arguments
  for (int i = 0; i < parsed_args.size(); ++i) {
    argv.push_back(const_cast<char*>(parsed_args[i].data()));
  }

  // end of parsed arguments
  argv.push_back(nullptr);

  // set argument count
  int argc = argv.size() - 1;

  // initialize gtest with arguments and run
  ::testing::InitGoogleTest(&argc, argv.data());
  *rv = RUN_ALL_TESTS();
});

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
