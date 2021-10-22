
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

# Common Makefile rules for the STM32 microcontroller project.

ifeq ($(strip $(CORE)),)
$(error CORE is not defined)
endif

CFLAGS += ${CFLAGS-${CORE}}

# APP Common options
CFLAGS += -Wall
CFLAGS += -fdata-sections -ffunction-sections -fstack-usage
CFALGS += -std=gnu11
CFLAGS += --specs=nano.specs
CFLAGS += -DCHECK_STM32_FAMILY

# APP DEBUG options
_APP_DEBUG_OPT = -g3 
# _APP_DEBUG_OPT += -gdwarf-2

ifeq ($(APP_DEBUG), 1)
CFLAGS += ${_APP_DEBUG_OPT} -O0 -DDEBUG
else
CFLAGS += ${_APP_DEBUG_OPT} ${APP_OPTIM} -DNDEBUG
endif

#
# Model sources
#
MODEL_PATH_NO_SLASH = $(realpath -s ${MODEL_PATH})
C_SOURCES += $(wildcard ${MODEL_PATH_NO_SLASH}/*.c)

CFLAGS += -I$(MODEL_PATH)

#
# TVM files -----------------------------------------
#

TVM_CRT_PATH = $(TVM_PATH)/src/runtime/crt/common
STM32_RUNTIME_PATH = $(TVM_PATH)/src/runtime/crt/contrib/stm32

CFLAGS += -DTLM_COUNTERS

C_SOURCES += $(TVM_CRT_PATH)/crt_backend_api.c
C_SOURCES += $(STM32_RUNTIME_PATH)/runtime.c
C_SOURCES += $(STM32_RUNTIME_PATH)/ai_runtime_api.c

CFLAGS += -I$(TVM_PATH)/include \
	  -I$(TVM_PATH)/3rdparty/dlpack/include \
	  -I$(TVM_PATH)/include/tvm/runtime \
	  -I$(STM32_RUNTIME_PATH)

#
# Application files -----------------------------------------
#

C_SOURCES += ${ROOT_DIR}/Src/app_x-cube-ai.c

C_SOURCES += ${ROOT_DIR}/Src/sysmem.c
C_SOURCES += ${ROOT_DIR}/Src/syscalls.c

# Extra USB_CDC support
ifeq ($(strip $(USE_USB_CDC)),y)
ifeq ($(strip $(USB_CDC_IS_AVAILABLE)), y)
CFLAGS += -DUSE_USB_CDC_CLASS=1
endif
endif

ifeq ($(VALID), 1)

CFLAGS += -DUSE_VALID
CFLAGS += -DAI_PB_TEST=1
CFLAGS += -DNO_X_CUBE_AI_RUNTIME=1

CFLAGS += -I${ROOT_DIR}/Inc/Validation
CFLAGS += -I${ROOT_DIR}/Inc/Misc

C_SOURCES += ${ROOT_DIR}/Src/Validation/aiPbMgr.c
C_SOURCES += ${ROOT_DIR}/Src/Validation/aiPbIO.c

C_SOURCES += ${ROOT_DIR}/Src/Validation/aiValidation.c

C_SOURCES += ${ROOT_DIR}/Src/Validation/pb_common.c
C_SOURCES += ${ROOT_DIR}/Src/Validation/pb_decode.c
C_SOURCES += ${ROOT_DIR}/Src/Validation/pb_encode.c
C_SOURCES += ${ROOT_DIR}/Src/Validation/stm32msg.pb.c

else

#
# aiSysPerformance -----------------------------------------
#
# Include specific files for aiSystemPerformance application

CFLAGS += -I${ROOT_DIR}/Inc/SystemPerformance
CFLAGS += -I${ROOT_DIR}/Inc/Misc

C_SOURCES += ${ROOT_DIR}/Src/SystemPerformance/aiSystemPerformance.c

endif

C_SOURCES += ${ROOT_DIR}/Src/Misc/aiTestTvmHelper.c
C_SOURCES += ${ROOT_DIR}/Src/Misc/aiTestUtility.c

ifneq ($(strip $(BAUDRATE)),)
CFLAGS += -DBAUDRATE=${BAUDRATE}
endif

#
# C-STARTUP/HAL/BSP files -----------------------------------------
#
HAL_DIR_ROOT = $(X_CUBE_PATH)/Drivers/${STM32}_HAL_Driver
BSP_DIR_ROOT = $(X_CUBE_PATH)/Drivers/BSP
CMSIS_DIR_ROOT = $(X_CUBE_PATH)/Drivers/CMSIS

include ${BOARDS_DIR}/${BOARD}/files.mk

CFLAGS += -I${ROOT_DIR}/Inc
CFLAGS += -I${BOARDS_DIR}/${BOARD}/Inc

# CMSIS
CFLAGS += -I$(CMSIS_DIR_ROOT)/Include
CFLAGS += -I$(CMSIS_DIR_ROOT)/Device/ST/${STM32}/Include

# libraries

LIBS = -lc -lm -lnosys
#LDFLAGS = $(MCU) -specs=nano.specs -T$(LDSCRIPT) $(LIBDIR) $(LIBS) -Wl,-Map=$(BUILD_DIR)/$(TARGET).map,--cref -Wl,--gc-sections
LDFLAGS += $(LDFLAGS-$(CORE)) -specs=nano.specs $(LIBS) -Wl,-Map=$(BUILD_DIR)/$(BOARD).map,--cref -Wl,--gc-sections
LDFLAGS += -u _printf_float
LDFLAGS += -Wl,--wrap=malloc -Wl,--wrap=free

#LDFLAGS += -Wl,-Map=$(BUILD_DIR)/$(APP_NAME).map,--cref -Wl,--gc-sections -static
##LDFLAGS += -specs=nosys.specs -specs=nano.specs -lstdc++ -lsupc++
#LDFLAGS += -specs=nosys.specs -specs=nano.specs
##LDFLAGS += -Wl,--start-group -lc -lm -Wl,--end-group -Wl,--print-memory-usage
#LDFLAGS += -Wl,--start-group -lc -lm -Wl,--end-group
LDFLAGS += -Wl,--print-memory-usage


# list of objects
OBJECTS = $(addprefix $(BUILD_DIR)/,$(notdir $(C_SOURCES:.c=.o)))
vpath %.c $(sort $(dir $(C_SOURCES)))

# list of ASM program objects
OBJECTS += $(addprefix $(BUILD_DIR)/,$(notdir $(ASM_SOURCES:.s=.o)))
vpath %.s $(sort $(dir $(ASM_SOURCES)))

######################################
# target
######################################

.PHONY: all clean info flash app.clean

all: $(BUILD_DIR) info.short $(BUILD_DIR)/$(APP_NAME).bin size

$(BUILD_DIR)/$(APP_NAME).elf: $(OBJECTS)
	@mkdir -p $(@D)
	@echo "LD(${APP_NAME})  $@ (VALID=$(VALID))"
	${q}$(CC) $(OBJECTS) $(LDFLAGS) -o $@
	@echo "LIST(${APP_NAME})  $(BUILD_DIR)/$(APP_NAME).elf (VALID=$(VALID))"
	${q}$(OBJDUMP) -h -S  $(BUILD_DIR)/$(APP_NAME).elf  > $(BUILD_DIR)/$(APP_NAME).list

size: $(BUILD_DIR)/$(APP_NAME).elf
	@echo "SIZE(${APP_NAME})  $(BUILD_DIR)/$(APP_NAME).elf (VALID=$(VALID))"
	${q}$(SIZE) $(BUILD_DIR)/$(APP_NAME).elf

$(BUILD_DIR)/$(APP_NAME).bin: $(BUILD_DIR)/$(APP_NAME).elf
	@echo "OBJ(${APP_NAME})  $@"
	${q}$(OBJCOPY) -O binary $(BUILD_DIR)/$(APP_NAME).elf $(BUILD_DIR)/$(APP_NAME).bin

$(BUILD_DIR)/%.o: %.c
	$(CC) -c $(CFLAGS) -Wa,-a,-ad,-alms=$(BUILD_DIR)/$(notdir $(<:.c=.lst)) $< -o $@

$(BUILD_DIR)/%.o: %.s
	$(CC) -c $(CFLAGS) $< -o $@

$(BUILD_DIR):
	mkdir -p $@

info.short:
	@echo "I: Building AI test APP for STM32=${STM32} (APP_DEBUG=${APP_DEBUG})"
	@echo "I: TOOLS_DIR=${_TOOLS_DIR}"

info: info.short
	@echo "----------------------------------------------------------------"
	@echo "TOP               = ${TOP}"
	@echo "BUILD_DIR         = $(BUILD_DIR) (ROOT_DIR=${ROOT_DIR})"
	@echo "TARGET            = ${BOARD} (DEVICE_CORE=${DEVICE_CORE})"
	@echo "DEVICE            = ${DEVICE} (STM32=${STM32} CORE=${CORE})"
	@echo "APP_DEBUG         = $(APP_DEBUG) (APP_OPTIM=${APP_OPTIM} V=${V})"
	@echo "CFLAGS-CORE       = ${CFLAGS-${CORE}}"
	@echo "APP_NAME          = ${APP_NAME}"
	@echo "TOOLS             = GCC=${_TOOLS_VER} (MAKE=${MAKE_VERSION})"
	@echo "CFLAGS            = $(CFLAGS)"
	@echo "LDFLAGS           = $(LDFLAGS)"
