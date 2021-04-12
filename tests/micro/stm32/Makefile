
ifndef MODEL_PATH
$(error MODEL_PATH must be set and point at your model implementation)
endif

#
# Top-level TVM directory
#
TVM_PATH = $(shell pwd)/../../..

RUNTIME_PATH = $(TVM_PATH)/src/runtime/stm32

#
# Model sources
#
C_SOURCES := $(wildcard ${MODEL_PATH}/*.c)

#
# TVM sources
#
C_SOURCES += $(RUNTIME_PATH)/runtime.c
C_SOURCES += $(RUNTIME_PATH)/ai_runtime_api.c

#
# Application sources
#
C_SOURCES += main.c 

vpath %.c $(sort $(dir $(C_SOURCES)))

#
# Build
#

BUILD_DIR = $(MODEL_PATH)

TARGET = network.exe

OBJECTS = $(addprefix $(BUILD_DIR)/,$(notdir $(C_SOURCES:.c=.o)))

CXX = gcc -m32 -g

DEFINES =
INCLUDES = -I$(TVM_PATH)/3rdparty/dlpack/include -I$(TVM_PATH)/include -I$(TVM_PATH)/src/runtime/stm32

CFLAGS = $(DEFINES) $(INCLUDES)
LDFLAGS = -lm

all: $(BUILD_DIR)/$(TARGET)

$(BUILD_DIR)/$(TARGET): $(OBJECTS)
	$(CXX) $(CFLAGS) -o $@  $^ $(LDFLAGS)

$(BUILD_DIR)/main.o: main.c
	$(CXX) $(CFLAGS) -I$(MODEL_PATH) -c $< -o $@

$(BUILD_DIR)/%.o: %.c
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm $(BUILD_DIR)/*.o
	rm $(BUILD_DIR)/$(TARGET)
