#!/bin/bash
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
PROJROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../" && pwd )"

# Derive target specified by vta_config.json
VTA_CONFIG=${PROJROOT}/vta/config/vta_config.py
TARGET=$(python ${VTA_CONFIG} --target)

export PYTHONPATH=${PYTHONPATH}:${PROJROOT}/python:${PROJROOT}/vta/python
export PYTHONPATH=${PYTHONPATH}:/home/xilinx/pynq
python3 -m vta.exec.rpc_server --tracker fleet:9190 --key $TARGET
