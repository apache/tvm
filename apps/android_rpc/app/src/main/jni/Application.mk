APP_ABI := all
APP_PLATFORM := android-8
APP_STL := gnustl_static

APP_CPPFLAGS += -DDMLC_LOG_STACK_TRACE=0 -DTVM4J_ANDROID -std=c++11 -Oz -frtti
