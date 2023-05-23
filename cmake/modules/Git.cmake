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
#   - TVM_GIT_COMMIT_HASH - The git commit hash found, or "NOT-FOUND" if anything went wrong
#   - TVM_GIT_COMMIT_TIME - The git commit time, or "NOT-FOUND" if antything went wrong
find_package(Git QUIET)
if (${GIT_FOUND})
  message(STATUS "Git found: ${GIT_EXECUTABLE}")
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
                  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                  OUTPUT_VARIABLE TVM_GIT_COMMIT_HASH
                  RESULT_VARIABLE _TVM_GIT_RESULT
                  ERROR_VARIABLE _TVM_GIT_ERROR
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_STRIP_TRAILING_WHITESPACE)
  if (${_TVM_GIT_RESULT} EQUAL 0)
    message(STATUS "Found TVM_GIT_COMMIT_HASH=${TVM_GIT_COMMIT_HASH}")
  else()
    message(STATUS "Not a git repo")
    set(TVM_GIT_COMMIT_HASH "NOT-FOUND")
  endif()

  execute_process(COMMAND ${GIT_EXECUTABLE} show -s --format=%ci HEAD
                  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                  OUTPUT_VARIABLE TVM_GIT_COMMIT_TIME
                  RESULT_VARIABLE _TVM_GIT_RESULT
                  ERROR_VARIABLE _TVM_GIT_ERROR
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_STRIP_TRAILING_WHITESPACE)
  if (${_TVM_GIT_RESULT} EQUAL 0)
    message(STATUS "Found TVM_GIT_COMMIT_TIME=${TVM_GIT_COMMIT_TIME}")
  else()
    set(TVM_GIT_COMMIT_TIME "NOT-FOUND")
  endif()
else()
  message(WARNING "Git not found")
  set(TVM_GIT_COMMIT_HASH "NOT-FOUND")
  set(TVM_GIT_COMMIT_TIME "NOT-FOUND")
endif()
