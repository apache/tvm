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

if(USE_MSC)
    tvm_file_glob(GLOB_RECURSE MSC_CORE_SOURCE "src/contrib/msc/*.cc")
    list(APPEND COMPILER_SRCS ${MSC_CORE_SOURCE})

    tvm_file_glob(GLOB_RECURSE MSC_RUNTIME_SOURCE "src/runtime/contrib/msc/*.cc")
    list(APPEND RUNTIME_SRCS ${MSC_RUNTIME_SOURCE})

    if(USE_TENSORRT_RUNTIME)
        add_definitions("-DTENSORRT_ROOT_DIR=\"${TENSORRT_ROOT_DIR}\"")
    endif()

    message(STATUS "Build with MSC support...")
endif()
