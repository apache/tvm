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

if(USE_ETHOSU)
  tvm_file_glob(GLOB COMPILER_ETHOSU_SRCS
                src/relay/backend/contrib/ethosu/*
                src/contrib/ethosu/cascader/*
                src/contrib/ethosu/cascader/parts/*
                src/tir/contrib/ethosu/*)
  list(APPEND COMPILER_SRCS ${COMPILER_ETHOSU_SRCS})
else()
  # Keeping just utils.cc because it has Object definitions
  # used by python side
  tvm_file_glob(GLOB COMPILER_ETHOSU_SRCS
                src/relay/backend/contrib/ethosu/utils.cc
                src/contrib/ethosu/cascader/*
                src/contrib/ethosu/cascader/parts/*)
  list(APPEND COMPILER_SRCS ${COMPILER_ETHOSU_SRCS})
endif(USE_ETHOSU)
