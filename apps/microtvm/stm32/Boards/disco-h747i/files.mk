
#
# Application C sources
#
C_SRC_APP += ${BOARDS_DIR}/${BOARD}/Src/main.c
C_SRC_APP += ${BOARDS_DIR}/${BOARD}/Src/stm32h7xx_hal_msp.c
C_SRC_APP += ${BOARDS_DIR}/${BOARD}/Src/stm32h7xx_it.c
C_SRC_APP += ${BOARDS_DIR}/${BOARD}/Src/system_stm32h7xx.c

#C_SRC_APP += ${BOARDS_DIR}/${BOARD}/Src/sysmem.c
#C_SRC_APP += ${BOARDS_DIR}/${BOARD}/Src/syscalls.c

CFLAGS += -DUSE_HAL_DRIVER

# Startup file
ASM_SRC_APP += ${BOARDS_DIR}/${BOARD}/Asm/startup_stm32h747xx.s

# HAL

C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal.c
C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal_cortex.c
C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal_crc.c
C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal_crc_ex.c
# C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal_dma.c
# C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal_dma_ex.c
C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal_eth.c
C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal_exti.c

C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal_flash.c
C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal_flash_ex.c
C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal_gpio.c
# C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal_hsem.c

# C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal_i2c.c
# C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal_i2c_ex.c

C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal_mdma.c

C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal_pcd.c
C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal_pcd_ex.c

C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal_pwr.c
C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal_pwr_ex.c
C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal_qspi.c
C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal_rcc.c
C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal_rcc_ex.c
C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal_sdram.c
# C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal_tim.c
# C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal_tim_ex.c
C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal_uart.c
C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_hal_uart_ex.c
# C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_ll_usb.c
C_SRC_HAL +=  ${HAL_DIR_ROOT}/Src/stm32h7xx_ll_fmc.c

CFLAGS += -I${HAL_DIR_ROOT}/Inc

# BSP	

C_SRC_BSP +=  ${BSP_DIR_ROOT}/Components/is42s32800j/is42s32800j.c
C_SRC_BSP +=  ${BSP_DIR_ROOT}/Components/mfxstm32l152/mfxstm32l152.c
C_SRC_BSP +=  ${BSP_DIR_ROOT}/Components/mfxstm32l152/mfxstm32l152_reg.c
C_SRC_BSP +=  ${BSP_DIR_ROOT}/Components/mt25tl01g/mt25tl01g.c

C_SRC_BSP +=  ${BSP_DIR_ROOT}/STM32H747I-DISCO/stm32h747i_discovery.c
C_SRC_BSP +=  ${BSP_DIR_ROOT}/STM32H747I-DISCO/stm32h747i_discovery_sdram.c
C_SRC_BSP +=  ${BSP_DIR_ROOT}/STM32H747I-DISCO/stm32h747i_discovery_qspi.c

CFLAGS += -I${BSP_DIR_ROOT}/STM32H747I-DISCO
CFLAGS += -I${BSP_DIR_ROOT}/Components/mt25tl01g

C_SOURCES += ${C_SRC_APP} ${C_SRC_HAL} ${C_SRC_BSP}
ASM_SOURCES += ${ASM_SRC_APP}
