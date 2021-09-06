#!/bin/bash -e
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
#
# Usage: base_box_test.sh <MICROTVM_DEVICE>
#     Execute microTVM Zephyr tests.
#

set -e
set -x

if [ "$#" -lt 1 ]; then
    echo "Usage: base_box_test.sh <MICROTVM_DEVICE>"
    exit -1
fi

microtvm_device=$1

pytest tests/micro/zephyr/test_zephyr.py --microtvm-device=${microtvm_device}

if [ $microtvm_device == "stm32f746xx" ]; then
    echo "NOTE: skipped test_zephyr_aot.py on $microtvm_device -- known failure"
else
    pytest tests/micro/zephyr/test_zephyr_aot.py --microtvm-device=${microtvm_device}
fi
