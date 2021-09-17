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
# Usage: base_box_test.sh <ZEPHYR_BOARD>
#     Execute microTVM Zephyr tests.
#

set -e
set -x

if [ "$#" -lt 1 ]; then
    echo "Usage: base_box_test.sh <ZEPHYR_BOARD>"
    exit -1
fi

board=$1

pytest tests/micro/zephyr/test_zephyr.py --zephyr-board=${board}

if [ $board == "stm32f746g_disco" ] || [ $board == "nucleo_f746zg" ]; then
    echo "NOTE: skipped test_zephyr_aot.py on $board -- known failure"
else
    pytest tests/micro/zephyr/test_zephyr_aot.py --zephyr-board=${board}
fi

pytest tests/micro/zephyr/test_zephyr_armv7m.py --zephyr-board=${board}
