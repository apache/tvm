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

ifeq ($(strip $(DEVICE_CORE)),cortex-m4)

FPU = fpv4-sp-d16
FPU_ABI = hard
CORE = cortex-m4

CFLAGS-${CORE} += -mthumb -mcpu=$(CORE) -mfpu=${FPU}
CFLAGS-${CORE} += -mfloat-abi=${FPU_ABI}

ifeq ($(strip $(USE_CMSIS_DSP)),y)

ifeq (${STM32},STM32L4xx)
CFLAGS-${CORE} += -D__FPU_PRESENT=1
else
CFLAGS-${CORE} += -D__FPU_PRESENT=1U
endif

CFLAGS-${CORE} += -DARM_MATH_CM4
CFLAGS-${CORE} += -DARM_MATH
CFLAGS-${CORE} += -DARM_MATH_DSP
CFLAGS-${CORE} += -DARM_MATH_LOOPUNROLL
endif

LDFLAGS-$(CORE) += -mcpu=${CORE} -mthumb -mfpu=${FPU}  -mfloat-abi=${FPU_ABI}
endif
