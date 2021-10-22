
ifeq ($(strip $(DEVICE_CORE)),cortex-m33)

# FPU extension - single precision
FPU = fpv5-sp-d16
FPU_ABI = hard
CORE = cortex-m33

CFLAGS-${CORE} += -mthumb -mcpu=$(CORE) -mfpu=${FPU}
CFLAGS-${CORE} += -mfloat-abi=${FPU_ABI}


ifeq ($(strip $(USE_CMSIS_DSP)),y)

# see CMSIS/DSP/Include/arm_match.h (V1.5.3)
#
# ARM_MATH_ARMV8MML - to include "core_armv8mml.h"
# __FPU_PRESENT - FPU supported
# __DSP_PRESENT - Armv8-M Mainline core supports DSP instructions

CFLAGS-${CORE}-CMSIS += -DARM_MATH_ARMV8MML
CFLAGS-${CORE}-CMSIS += -D__DSP_PRESENT=1
CFLAGS-${CORE}-CMSIS += -D__FPU_PRESENT=1U

CFLAGS-${CORE} += -DARM_MATH_DSP
CFLAGS-${CORE} += -DARM_MATH_LOOPUNROLL

endif

LDFLAGS-$(CORE) += -mcpu=${CORE} -mthumb -mfpu=${FPU}  -mfloat-abi=${FPU_ABI}
endif