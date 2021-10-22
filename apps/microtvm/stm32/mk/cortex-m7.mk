
ifeq ($(strip $(DEVICE_CORE)),cortex-m7)

FPU = fpv5-sp-d16
FPU_ABI = hard
CORE = cortex-m7

CFLAGS-${CORE} += -mthumb -mcpu=$(CORE) -mfpu=${FPU}
CFLAGS-${CORE} += -mfloat-abi=${FPU_ABI}

ifeq ($(strip $(USE_CMSIS_DSP)),y)

CFLAGS-${CORE} += -DARM_MATH_CM7
CFLAGS-${CORE} += -DARM_MATH
CFLAGS-${CORE} += -D__FPU_PRESENT=1

CFLAGS-${CORE} += -DARM_MATH_DSP
CFLAGS-${CORE} += -DARM_MATH_LOOPUNROLL
endif

LDFLAGS-$(CORE) += -mcpu=${CORE} -mthumb -mfpu=${FPU}  -mfloat-abi=${FPU_ABI}
endif