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

namespace tvm {
namespace runtime {
namespace hexagon {

class GtestPrinter : public testing::EmptyTestEventListener {
  void OnTestProgramStart(const testing::UnitTest& unit_test) override {
    gtest_out_ << "[==========] Running " << unit_test.test_to_run_count() << " test(s) from "
               << unit_test.test_suite_to_run_count() << " test suite(s).\n";
  }

  void OnTestProgramEnd(const testing::UnitTest& unit_test) override {
    gtest_out_ << "[==========] " << unit_test.test_to_run_count() << " test(s) from "
               << unit_test.test_suite_to_run_count() << " test suite(s) ran. ("
               << unit_test.elapsed_time() << " ms total)\n";
    gtest_out_ << "[  PASSED  ] " << unit_test.successful_test_count() << " test(s)\n";

    if (unit_test.failed_test_count()) {
      gtest_out_ << "[  FAILED  ] " << unit_test.failed_test_count() << " test(s)\n";
    }
  }

  void OnTestSuiteStart(const testing::TestSuite& test_suite) override {
    gtest_out_ << "[----------] " << test_suite.test_to_run_count() << " test(s) from "
               << test_suite.name() << "\n";
  }

  void OnTestSuiteEnd(const testing::TestSuite& test_suite) override {
    gtest_out_ << "[----------] " << test_suite.test_to_run_count() << " test(s) from "
               << test_suite.name() << " (" << test_suite.elapsed_time() << " ms total)\n";
  }

  void OnTestStart(const testing::TestInfo& test_info) override {
    gtest_out_ << "[ RUN      ] " << test_info.test_suite_name() << "." << test_info.name() << "\n";
  }

  void OnTestEnd(const testing::TestInfo& test_info) override {
    for (int i = 0; i < test_info.result()->total_part_count(); ++i) {
      gtest_out_ << test_info.result()->GetTestPartResult(i).message() << "\n";
    }
    if (test_info.result()->Passed()) {
      gtest_out_ << "[       OK ]";
    } else {
      gtest_out_ << "[  FAILED  ]";
    }
    gtest_out_ << " " << test_info.test_suite_name() << "." << test_info.name() << " ("
               << test_info.result()->elapsed_time() << " ms)\n";
  }

  std::stringstream gtest_out_;

 public:
  std::string GetOutput() { return gtest_out_.str(); }
};

TVM_REGISTER_GLOBAL("hexagon.run_unit_tests").set_body([](TVMArgs args, TVMRetValue* rv) {
  // gtest args are passed into this packed func as a singular string
  // split gtest args using <space> delimiter and build argument vector
  std::vector<std::string> parsed_args = tvm::support::Split(args[0], ' ');
  std::vector<char*> argv;

  // add executable name
  argv.push_back(const_cast<char*>("hexagon_run_unit_tests"));

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

  // add printer to capture gtest output in a string
  GtestPrinter* gprinter = new GtestPrinter();
  testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
  listeners.Append(gprinter);

  int gtest_error_code = RUN_ALL_TESTS();
  std::string gtest_output = gprinter->GetOutput();
  std::stringstream gtest_error_code_and_output;
  gtest_error_code_and_output << gtest_error_code << std::endl;
  gtest_error_code_and_output << gtest_output;
  *rv = gtest_error_code_and_output.str();
  delete gprinter;
});

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
