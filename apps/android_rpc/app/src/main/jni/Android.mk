LOCAL_PATH := $(call my-dir)
MY_PATH := $(LOCAL_PATH)

include $(CLEAR_VARS)

LOCAL_PATH := $(MY_PATH)
ROOT_PATH := $(MY_PATH)/../../../../../..

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

include $(BUILD_SHARED_LIBRARY)
