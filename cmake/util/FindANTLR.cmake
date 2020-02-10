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

#######################################################
# Enhanced version of find ANTLR.
#
# Usage:
#   find_antlr(${USE_ANTLR})
#
# - When USE_ANTLR=ON, use auto search by first trying to find antlr4 program,
#                      then trying to find antlr-*-complete.jar
# - When USE_ANTLR=/path/to/antlr-*-complete.jar, use provided jar
#
# Provide variables:
# - ANTLR4
#
macro(find_antlr use_antlr)
  set(JAVA_HOME $ENV{JAVA_HOME})
  if (NOT DEFINED JAVA_HOME)
    # Hack to get system to search for Java itself.
    message(STATUS "JAVA_HOME is not defined. Set it to ensure proper use")
    set(JAVA_HOME "/usr")
  endif()
  if(MSVC)
    set(JAVA_PROGRAM ${JAVA_HOME}/java.exe)
  else()
    set(JAVA_PROGRAM ${JAVA_HOME}/bin/java)
  endif()
  message(STATUS "Using Java at " ${JAVA_PROGRAM})

  if (${use_antlr} STREQUAL "ON")
    find_program(ANTLR4 antlr4)
    if (NOT ANTLR4)
      file(GLOB_RECURSE ANTLR4JAR
          /usr/local/lib/antlr-*-complete.jar
          /usr/local/Cellar/*antlr-*-complete.jar)

      # Get the first element of the list of antlr jars.
      # Sort and reverse the list so the item selected is the highest
      #   version in lib or else in Cellar if no lib installation exists.
      list(SORT ANTLR4JAR)
      list(REVERSE ANTLR4JAR)
      list(GET ANTLR4JAR 0 ANTLR4JAR)

      set(ANTLR4 ${JAVA_PROGRAM} -jar ${ANTLR4JAR})
    endif()
  elseif(NOT ${use_antlr} STREQUAL "OFF")
    set(ANTLR4 ${JAVA_PROGRAM} -jar ${use_antlr})
  endif()
  message(STATUS "ANTLR4=${ANTLR4}")
endmacro(find_antlr)
