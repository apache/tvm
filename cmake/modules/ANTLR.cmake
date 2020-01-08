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
if(USE_ANTLR)
  find_antlr(${USE_ANTLR})
  if(ANTLR4)

    set(RELAY_PARSER_DIR
      ${CMAKE_CURRENT_SOURCE_DIR}/python/tvm/relay/grammar)

    set(RELAY_PARSER
      ${RELAY_PARSER_DIR}/py3/RelayVisitor.py
      ${RELAY_PARSER_DIR}/py3/RelayParser.py
      ${RELAY_PARSER_DIR}/py3/RelayLexer.py)


    # Generate ANTLR grammar for parsing.
    add_custom_command(OUTPUT ${RELAY_PARSER}
      COMMAND ${ANTLR4} -visitor -no-listener -Dlanguage=Python3 ${RELAY_PARSER_DIR}/Relay.g4 -o ${RELAY_PARSER_DIR}/py3
      DEPENDS ${RELAY_PARSER_DIR}/Relay.g4
      WORKING_DIRECTORY ${RELAY_PARSER_DIR})

    add_custom_target(relay_parser ALL DEPENDS ${RELAY_PARSER})
  else()
    message(FATAL_ERROR "Can't find ANTLR4")
  endif()
endif(USE_ANTLR)
