
ifeq ($(OS),Windows_NT)
#  SHELL := cmd
  tmp   :=
  SEP   := \$(tmp)
  RM    := del /f /q
  RMDIR := rd /s /q
  MKDIR := mkdir
  FixPath = $(subst /,\, s$1)
else
  SEP   := /
  RM    := rm -f
  RMDIR := rm -rf
  MKDIR := mkdir -p
  FixPath = $1
endif

CROSS_COMPILE ?= arm-none-eabi-

ifneq ($(GCC_BIN_DIR),)
_GCC_PREFIX := $(GCC_BIN_DIR)/
else
_GCC_PREFIX := ${ARM_PATH}/
endif

CC		= $(_GCC_PREFIX)$(CROSS_COMPILE)gcc
CPP		= $(_GCC_PREFIX)$(CROSS_COMPILE)cpp
LD		= $(_GCC_PREFIX)$(CROSS_COMPILE)ld
AR		= $(_GCC_PREFIX)$(CROSS_COMPILE)ar
NM		= $(_GCC_PREFIX)$(CROSS_COMPILE)nm
OBJCOPY	= $(_GCC_PREFIX)$(CROSS_COMPILE)objcopy
OBJDUMP	= $(_GCC_PREFIX)$(CROSS_COMPILE)objdump
READELF	= $(_GCC_PREFIX)$(CROSS_COMPILE)readelf
SIZE    = $(_GCC_PREFIX)$(CROSS_COMPILE)size
CXX		= $(_GCC_PREFIX)$(CROSS_COMPILE)g++


_TOOLS_VER = $(shell $(CC) -dumpversion)
_TOOLS_DIR = $(shell $(CC) -print-sysroot)


