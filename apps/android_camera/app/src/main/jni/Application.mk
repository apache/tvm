ifndef config
	ifneq ("$(wildcard ./config.mk)","")
	  config ?= config.mk
	else
	  config ?= make/config.mk
	endif
endif

include $(config)

APP_STL := c++_static

APP_CPPFLAGS += -DDMLC_LOG_STACK_TRACE=0 -DTVM4J_ANDROID=1 -std=c++11 -Oz -frtti
ifeq ($(USE_OPENCL), 1)                                                                                                                                             
	APP_CPPFLAGS += -DTVM_OPENCL_RUNTIME=1
endif
