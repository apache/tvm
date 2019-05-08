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

file(GLOB TSIM_SW_SRC src/driver.cc)
add_library(driver SHARED ${TSIM_SW_SRC})
target_include_directories(driver PRIVATE ${VTA_DIR}/include)

if(APPLE)
  set_target_properties(driver PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif(APPLE)
