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
# pylint: disable=invalid-name, missing-function-docstring
"""Common functions for popen_pool test cases"""

TEST_GLOBAL_STATE_1 = 0
TEST_GLOBAL_STATE_2 = 0
TEST_GLOBAL_STATE_3 = 0


def initializer(test_global_state_1, test_global_state_2, test_global_state_3):
    global TEST_GLOBAL_STATE_1, TEST_GLOBAL_STATE_2, TEST_GLOBAL_STATE_3
    TEST_GLOBAL_STATE_1 = test_global_state_1
    TEST_GLOBAL_STATE_2 = test_global_state_2
    TEST_GLOBAL_STATE_3 = test_global_state_3


def after_initializer():
    global TEST_GLOBAL_STATE_1, TEST_GLOBAL_STATE_2, TEST_GLOBAL_STATE_3
    return TEST_GLOBAL_STATE_1, TEST_GLOBAL_STATE_2, TEST_GLOBAL_STATE_3
