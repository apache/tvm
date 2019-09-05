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

PATH_SETTINGS?=$(PWD)/settings.mk

# Driver configuration file
include $(PATH_SETTINGS)

ifeq ($(KERNELRELEASE),)
# kbuild

default:
	make ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) -C $(KSOURCE_DIR) M=`pwd` modules

clean:
	make -C $(KSOURCE_DIR) M=`pwd` clean

else
# run from Kernel Makefile

obj-m 	  := cma.o
ccflags-y := -DDRIVER_NODE_NAME="\"$(DRIVER_NODE_NAME)\"" \
			 -DCMA_DEBUG=$(CMA_DEBUG) \
			 -DCMA_IOC_MAGIC=$(CMA_IOC_MAGIC)
endif
