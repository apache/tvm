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
# This script provides
#   - GIT_FOUND - true if the command line client was found
#   - GIT_EXECUTABLE - path to git command line client
#   - TVM_GIT_COMMIT_HASH
find_package(Git QUIET)
if (${GIT_FOUND})
  message(STATUS "Git found: ${GIT_EXECUTABLE}")
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
                  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                  OUTPUT_VARIABLE TVM_GIT_COMMIT_HASH
                  ERROR_QUIET
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "Found TVM_GIT_COMMIT_HASH=${TVM_GIT_COMMIT_HASH}")
else()
  message(WARNING "Git not found")
  set(TVM_GIT_COMMIT_HASH "git-not-found")
endif()
