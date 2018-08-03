ifndef config
    ifneq ("$(wildcard ./config.mk)","")
        config ?= config.mk
    else
        config ?= make/config.mk
    endif
endif

include $(config)

# We target every architecture except armeabi here, for two reasons:
# 1) armeabi is deprecated in NDK r16 and removed in r17
# 2) vulkan is not supported in armeabi
APP_ABI ?= armeabi-v7a arm64-v8a x86 x86_64 mips
APP_STL := c++_static

APP_CPPFLAGS += -DDMLC_LOG_STACK_TRACE=0 -DTVM4J_ANDROID=1 -std=c++11 -Oz -frtti
ifeq ($(USE_OPENCL), 1)
    APP_CPPFLAGS += -DTVM_OPENCL_RUNTIME=1
endif

ifeq ($(USE_VULKAN), 1)
    APP_CPPFLAGS += -DTVM_VULKAN_RUNTIME=1
    APP_LDFLAGS += -lvulkan
endif
