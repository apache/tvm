# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

GTEST_LIB=$(GTEST_PATH)/lib/
GTEST_INC=$(GTEST_PATH)/include/

TEST_SRC = $(wildcard tests/cpp/*_test.cc)
TEST = $(patsubst tests/cpp/%_test.cc, tests/cpp/%_test, $(TEST_SRC))

tests/cpp/%_test: tests/cpp/%_test.cc lib/libnnvm.a
	$(CXX) -std=c++11 $(CFLAGS) -MM -MT tests/cpp/$* $< >tests/cpp/$*.d
	$(CXX) -std=c++11 $(CFLAGS) -I$(GTEST_INC) -o $@ $(filter %.cc %.a, $^)  \
		-L$(GTEST_LIB)  $(LDFLAGS) -lgtest

-include tests/cpp/*.d
