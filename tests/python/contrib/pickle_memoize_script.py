#!/usr/bin/env python3

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

import sys

import tvm


@tvm.contrib.pickle_memoize.memoize("test_memoize_save_data", save_at_exit=True)
def get_data_saved():
    return 42


@tvm.contrib.pickle_memoize.memoize("test_memoize_transient_data", save_at_exit=False)
def get_data_transient():
    return 42


def main():
    assert len(sys.argv) == 3, "Expect arguments SCRIPT NUM_SAVED NUM_TRANSIENT"

    num_iter_saved = int(sys.argv[1])
    num_iter_transient = int(sys.argv[2])

    for _ in range(num_iter_saved):
        get_data_saved()
    for _ in range(num_iter_transient):
        get_data_transient()


if __name__ == "__main__":
    main()
