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

ifndef TVM_PATH
$(error TVM_PATH must be set and point at your TVM installation)
endif

ifndef MODEL_PATH
$(error MODEL_PATH must be set and point at your model implementation)
endif

ifndef BUILD_PATH
$(error BUILD_PATH must be set and point at where your models are built)
endif

ifndef IMAGE_PATH
$(error IMAGE_PATH must be set and point at where your images are stored)
endif

SRC_PATH = $(TVM_PATH)/tests/crt/contrib/stm32/src
TVM_CRT_PATH = $(TVM_PATH)/src/runtime/crt/common
STM32_RUNTIME_PATH = $(TVM_PATH)/src/runtime/crt/contrib/stm32

#
# Model sources
#
C_SOURCES := $(wildcard ${MODEL_PATH}/*.c)

#
# TVM sources
#
C_SOURCES += $(TVM_CRT_PATH)/crt_backend_api.c
C_SOURCES += $(STM32_RUNTIME_PATH)/runtime.c
C_SOURCES += $(STM32_RUNTIME_PATH)/ai_runtime_api.c

#
# Application sources
#
C_SOURCES += $(SRC_PATH)/main.c

vpath %.c $(sort $(dir $(C_SOURCES)))

#
# Build
#

BUILD_DIR = $(MODEL_PATH)

TARGET = network.exe

OBJECTS = $(addprefix $(BUILD_DIR)/,$(notdir $(C_SOURCES:.c=.o)))

CXX = gcc -m32 -g

DEFINES =
INCLUDES = -I$(TVM_PATH)/3rdparty/dlpack/include -I$(TVM_PATH)/include -I$(STM32_RUNTIME_PATH)

CFLAGS = $(DEFINES) $(INCLUDES)
LDFLAGS = -lm

all: $(BUILD_DIR)/$(TARGET)

$(BUILD_DIR)/$(TARGET): $(OBJECTS)
	$(CXX) $(CFLAGS) -o $@  $^ $(LDFLAGS)

$(BUILD_DIR)/main.o: main.c
	$(CXX) -DBUILD_PATH=\"$(BUILD_PATH)\" -DIMAGE_PATH=\"$(IMAGE_PATH)\"$(CFLAGS) -I$(MODEL_PATH) -c $< -o $@

$(BUILD_DIR)/%.o: %.c
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm $(BUILD_DIR)/*.o
	rm $(BUILD_DIR)/$(TARGET)
