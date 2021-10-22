
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

# Source files specific for this target

# APP files
C_SRC_APP += ${BOARDS_DIR}/${BOARD}/Src/main.c
C_SRC_APP += ${BOARDS_DIR}/${BOARD}/Src/stm32f4xx_hal_msp.c
C_SRC_APP += ${BOARDS_DIR}/${BOARD}/Src/stm32f4xx_it.c
C_SRC_APP += ${BOARDS_DIR}/${BOARD}/Src/system_stm32f4xx.c

CFLAGS += -DUSE_HAL_DRIVER

# Startup file
ASM_SRC_APP += ${BOARDS_DIR}/${BOARD}/Asm/startup_stm32f412zgtx.s

# HAL files

C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32f4xx_hal.c
C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32f4xx_hal_cortex.c
#C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32f4xx_hal_crc.c
# C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32f4xx_hal_dma.c
# C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32f4xx_hal_dma_ex.c
# C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32f4xx_hal_exti.c
# C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32f4xx_hal_flash.c
# C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32f4xx_hal_flash_ex.c
# C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32f4xx_hal_flash_ramfunc.c
C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32f4xx_hal_gpio.c
C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32f4xx_hal_pwr.c
C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32f4xx_hal_pwr_ex.c
C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32f4xx_hal_rcc.c
C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32f4xx_hal_rcc_ex.c
# C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32f4xx_hal_tim.c
# C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32f4xx_hal_tim_ex.c
C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32f4xx_hal_uart.c

CFLAGS += -I${HAL_DIR_ROOT}/Inc

C_SOURCES += ${C_SRC_APP} ${C_SRC_HAL}
ASM_SOURCES += ${ASM_SRC_APP}
