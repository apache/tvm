
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

#
# GCC config
#

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


