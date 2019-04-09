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
echo "Build the libraries.."
mkdir -p lib
make
echo "Run the example"
export LD_LIBRARY_PATH=../../build:${LD_LIBRARY_PATH}
export DYLD_LIBRARY_PATH=../../build:${DYLD_LIBRARY_PATH}

echo "Run the deployment with all in one packed library..."
lib/cpp_deploy_pack

echo "Run the deployment with all in normal library..."
lib/cpp_deploy_normal
