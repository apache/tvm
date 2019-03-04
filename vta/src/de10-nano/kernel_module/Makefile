PATH_SETTINGS?=$(PWD)/settings.mak
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
