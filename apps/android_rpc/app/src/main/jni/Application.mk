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

# We target every architecture except armeabi here, for two reasons:
# 1) armeabi is deprecated in NDK r16 and removed in r17
# 2) vulkan is not supported in armeabi
APP_ABI ?= armeabi-v7a arm64-v8a x86 x86_64 mips
APP_STL := c++_shared

APP_CPPFLAGS += -DDMLC_LOG_STACK_TRACE=0 -DTVM4J_ANDROID=1 -std=c++14 -Oz -frtti
ifeq ($(USE_OPENCL), 1)
    APP_CPPFLAGS += -DTVM_OPENCL_RUNTIME=1
endif

ifeq ($(USE_VULKAN), 1)
    APP_CPPFLAGS += -DTVM_VULKAN_RUNTIME=1
    APP_LDFLAGS += -lvulkan
endif

ifeq ($(USE_SORT), 1)
    APP_CPPFLAGS += -DUSE_SORT=1
endif
