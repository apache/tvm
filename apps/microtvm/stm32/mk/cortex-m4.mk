
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