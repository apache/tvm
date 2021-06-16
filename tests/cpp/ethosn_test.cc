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

/*!
 * \file tests/cpp/ethosn_test.cc
 * \brief Ethos-N test suite.
 */

#include <gtest/gtest.h>

#include "../../src/relay/backend/contrib/ethosn/ethosn_variant.h"

TEST(MakeVariant, Basic) {
  EXPECT_EQ(MakeVariant("Ethos-N77", "", ""), "Ethos-N77");
  EXPECT_EQ(MakeVariant("Ethos-N78", "2", "4"), "Ethos-N78_2TOPS_4PLE_RATIO");
  EXPECT_EQ(MakeVariant("ethos-n78", "2", "4"), "Ethos-N78_2TOPS_4PLE_RATIO");
  EXPECT_EQ(MakeVariant("Ethos-N78", "4", "4"), "Ethos-N78_4TOPS_4PLE_RATIO");
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
