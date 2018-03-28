LOCAL_PATH := $(call my-dir)
MY_PATH := $(LOCAL_PATH)

include $(CLEAR_VARS)

LOCAL_PATH := $(MY_PATH)
ROOT_PATH := $(MY_PATH)/../../../../../..

ifndef config
	ifneq ("$(wildcard ./config.mk)","")
	  config ?= config.mk
	else
	  config ?= make/config.mk
	endif
endif

include $(config)

LOCAL_SRC_FILES := ml_dmlc_tvm_native_c_api.cc
LOCAL_LDFLAGS := -L$(SYSROOT)/usr/lib/ -llog

LOCAL_C_INCLUDES := $(ROOT_PATH)/include \
                    $(ROOT_PATH)/dlpack/include \
                    $(ROOT_PATH)/dmlc-core/include \
                    $(ROOT_PATH)/HalideIR/src \
                    $(ROOT_PATH)/topi/include

LOCAL_MODULE = tvm4j_runtime_packed

LOCAL_CPP_FEATURES += exceptions
LOCAL_LDLIBS += -latomic
LOCAL_ARM_MODE := arm

ifdef ADD_C_INCLUDES
	LOCAL_C_INCLUDES += $(ADD_C_INCLUDES)
endif

ifdef ADD_LDLIBS
	LOCAL_LDLIBS += $(ADD_LDLIBS)
endif

include $(BUILD_SHARED_LIBRARY)
