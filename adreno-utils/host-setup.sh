#!/usr/bin/env bash
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

DEVICE=$1
TRACKER_PORT=$2
DEVICE_PORT=$3

adb -s $DEVICE reverse tcp:${TRACKER_PORT} tcp:${TRACKER_PORT}
adb -s $DEVICE forward tcp:${DEVICE_PORT} tcp:${DEVICE_PORT}
adb -s $DEVICE forward tcp:$((DEVICE_PORT + 1)) tcp:$((DEVICE_PORT + 1))
adb -s $DEVICE forward tcp:$((DEVICE_PORT + 2)) tcp:$((DEVICE_PORT + 2))
adb -s $DEVICE forward tcp:$((DEVICE_PORT + 3)) tcp:$((DEVICE_PORT + 3))
adb -s $DEVICE forward tcp:$((DEVICE_PORT + 4)) tcp:$((DEVICE_PORT + 4))
python3 -m tvm.exec.query_rpc_tracker --port ${TRACKER_PORT}
