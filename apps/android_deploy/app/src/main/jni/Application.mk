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

ifndef config
	ifneq ("$(wildcard ./config.mk)","")
	  config ?= config.mk
	else
	  config ?= make/config.mk
	endif
endif

include $(config)

APP_STL := c++_static

APP_CPPFLAGS += -DDMLC_LOG_STACK_TRACE=0 -DTVM4J_ANDROID=1 -std=c++14 -Oz -frtti
ifeq ($(USE_OPENCL), 1)
	APP_CPPFLAGS += -DTVM_OPENCL_RUNTIME=1
endif
