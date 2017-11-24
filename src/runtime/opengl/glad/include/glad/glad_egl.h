
#ifndef __glad_egl_h_

#ifdef __egl_h_
#error EGL header already included, remove this include, glad already provides it
#endif

#define __glad_egl_h_
#define __egl_h_

#if defined(_WIN32) && !defined(APIENTRY) && !defined(__CYGWIN__) && !defined(__SCITECH_SNAP__)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#include <windows.h>
#endif

#ifndef APIENTRY
#define APIENTRY
#endif
#ifndef APIENTRYP
#define APIENTRYP APIENTRY *
#endif
#ifndef GLAPI
#define GLAPI extern
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* (* GLADloadproc)(const char *name);
GLAPI int gladLoadEGLLoader(GLADloadproc);

GLAPI int gladLoadEGL(void);
GLAPI int gladLoadEGLLoader(GLADloadproc);

#include <KHR/khrplatform.h>
#include <EGL/eglplatform.h>
typedef unsigned int EGLBoolean;
typedef unsigned int EGLenum;
typedef intptr_t EGLAttribKHR;
typedef intptr_t EGLAttrib;
typedef void *EGLClientBuffer;
typedef void *EGLConfig;
typedef void *EGLContext;
typedef void *EGLDeviceEXT;
typedef void *EGLDisplay;
typedef void *EGLImage;
typedef void *EGLImageKHR;
typedef void *EGLOutputLayerEXT;
typedef void *EGLOutputPortEXT;
typedef void *EGLStreamKHR;
typedef void *EGLSurface;
typedef void *EGLSync;
typedef void *EGLSyncKHR;
typedef void *EGLSyncNV;
typedef void (*__eglMustCastToProperFunctionPointerType)(void);
typedef khronos_utime_nanoseconds_t EGLTimeKHR;
typedef khronos_utime_nanoseconds_t EGLTime;
typedef khronos_utime_nanoseconds_t EGLTimeNV;
typedef khronos_utime_nanoseconds_t EGLuint64NV;
typedef khronos_uint64_t EGLuint64KHR;
typedef int EGLNativeFileDescriptorKHR;
typedef khronos_ssize_t EGLsizeiANDROID;
typedef void (*EGLSetBlobFuncANDROID) (const void *key, EGLsizeiANDROID keySize, const void *value, EGLsizeiANDROID valueSize);
typedef EGLsizeiANDROID (*EGLGetBlobFuncANDROID) (const void *key, EGLsizeiANDROID keySize, void *value, EGLsizeiANDROID valueSize);
struct EGLClientPixmapHI {
    void  *pData;
    EGLint iWidth;
    EGLint iHeight;
    EGLint iStride;
};
EGLBoolean eglChooseConfig(EGLDisplay, const EGLint*, EGLConfig*, EGLint, EGLint*);
EGLBoolean eglCopyBuffers(EGLDisplay, EGLSurface, EGLNativePixmapType);
EGLContext eglCreateContext(EGLDisplay, EGLConfig, EGLContext, const EGLint*);
EGLSurface eglCreatePbufferSurface(EGLDisplay, EGLConfig, const EGLint*);
EGLSurface eglCreatePixmapSurface(EGLDisplay, EGLConfig, EGLNativePixmapType, const EGLint*);
EGLSurface eglCreateWindowSurface(EGLDisplay, EGLConfig, EGLNativeWindowType, const EGLint*);
EGLBoolean eglDestroyContext(EGLDisplay, EGLContext);
EGLBoolean eglDestroySurface(EGLDisplay, EGLSurface);
EGLBoolean eglGetConfigAttrib(EGLDisplay, EGLConfig, EGLint, EGLint*);
EGLBoolean eglGetConfigs(EGLDisplay, EGLConfig*, EGLint, EGLint*);
EGLDisplay eglGetCurrentDisplay();
EGLSurface eglGetCurrentSurface(EGLint);
EGLDisplay eglGetDisplay(EGLNativeDisplayType);
EGLint eglGetError();
__eglMustCastToProperFunctionPointerType eglGetProcAddress(const char*);
EGLBoolean eglInitialize(EGLDisplay, EGLint*, EGLint*);
EGLBoolean eglMakeCurrent(EGLDisplay, EGLSurface, EGLSurface, EGLContext);
EGLBoolean eglQueryContext(EGLDisplay, EGLContext, EGLint, EGLint*);
const char* eglQueryString(EGLDisplay, EGLint);
EGLBoolean eglQuerySurface(EGLDisplay, EGLSurface, EGLint, EGLint*);
EGLBoolean eglSwapBuffers(EGLDisplay, EGLSurface);
EGLBoolean eglTerminate(EGLDisplay);
EGLBoolean eglWaitGL();
EGLBoolean eglWaitNative(EGLint);
EGLBoolean eglBindTexImage(EGLDisplay, EGLSurface, EGLint);
EGLBoolean eglReleaseTexImage(EGLDisplay, EGLSurface, EGLint);
EGLBoolean eglSurfaceAttrib(EGLDisplay, EGLSurface, EGLint, EGLint);
EGLBoolean eglSwapInterval(EGLDisplay, EGLint);
EGLBoolean eglBindAPI(EGLenum);
EGLenum eglQueryAPI();
EGLSurface eglCreatePbufferFromClientBuffer(EGLDisplay, EGLenum, EGLClientBuffer, EGLConfig, const EGLint*);
EGLBoolean eglReleaseThread();
EGLBoolean eglWaitClient();
EGLContext eglGetCurrentContext();
EGLSync eglCreateSync(EGLDisplay, EGLenum, const EGLAttrib*);
EGLBoolean eglDestroySync(EGLDisplay, EGLSync);
EGLint eglClientWaitSync(EGLDisplay, EGLSync, EGLint, EGLTime);
EGLBoolean eglGetSyncAttrib(EGLDisplay, EGLSync, EGLint, EGLAttrib*);
EGLImage eglCreateImage(EGLDisplay, EGLContext, EGLenum, EGLClientBuffer, const EGLAttrib*);
EGLBoolean eglDestroyImage(EGLDisplay, EGLImage);
EGLDisplay eglGetPlatformDisplay(EGLenum, void*, const EGLAttrib*);
EGLSurface eglCreatePlatformWindowSurface(EGLDisplay, EGLConfig, void*, const EGLAttrib*);
EGLSurface eglCreatePlatformPixmapSurface(EGLDisplay, EGLConfig, void*, const EGLAttrib*);
EGLBoolean eglWaitSync(EGLDisplay, EGLSync, EGLint);
#define EGL_READ_SURFACE_BIT_KHR 0x0001
#define EGL_WRITE_SURFACE_BIT_KHR 0x0002
#define EGL_LOCK_SURFACE_BIT_KHR 0x0080
#define EGL_OPTIMAL_FORMAT_BIT_KHR 0x0100
#define EGL_MATCH_FORMAT_KHR 0x3043
#define EGL_FORMAT_RGB_565_EXACT_KHR 0x30C0
#define EGL_FORMAT_RGB_565_KHR 0x30C1
#define EGL_FORMAT_RGBA_8888_EXACT_KHR 0x30C2
#define EGL_FORMAT_RGBA_8888_KHR 0x30C3
#define EGL_MAP_PRESERVE_PIXELS_KHR 0x30C4
#define EGL_LOCK_USAGE_HINT_KHR 0x30C5
#define EGL_BITMAP_POINTER_KHR 0x30C6
#define EGL_BITMAP_PITCH_KHR 0x30C7
#define EGL_BITMAP_ORIGIN_KHR 0x30C8
#define EGL_BITMAP_PIXEL_RED_OFFSET_KHR 0x30C9
#define EGL_BITMAP_PIXEL_GREEN_OFFSET_KHR 0x30CA
#define EGL_BITMAP_PIXEL_BLUE_OFFSET_KHR 0x30CB
#define EGL_BITMAP_PIXEL_ALPHA_OFFSET_KHR 0x30CC
#define EGL_BITMAP_PIXEL_LUMINANCE_OFFSET_KHR 0x30CD
#define EGL_LOWER_LEFT_KHR 0x30CE
#define EGL_UPPER_LEFT_KHR 0x30CF
#define EGL_CUDA_DEVICE_NV 0x323A
#define EGL_DRM_BUFFER_FORMAT_MESA 0x31D0
#define EGL_DRM_BUFFER_USE_MESA 0x31D1
#define EGL_DRM_BUFFER_FORMAT_ARGB32_MESA 0x31D2
#define EGL_DRM_BUFFER_MESA 0x31D3
#define EGL_DRM_BUFFER_STRIDE_MESA 0x31D4
#define EGL_DRM_BUFFER_USE_SCANOUT_MESA 0x00000001
#define EGL_DRM_BUFFER_USE_SHARE_MESA 0x00000002
#define EGL_DRM_DEVICE_FILE_EXT 0x3233
#define EGL_STREAM_BIT_KHR 0x0800
#define EGL_PLATFORM_X11_EXT 0x31D5
#define EGL_PLATFORM_X11_SCREEN_EXT 0x31D6
#define EGL_CL_EVENT_HANDLE_KHR 0x309C
#define EGL_SYNC_CL_EVENT_KHR 0x30FE
#define EGL_SYNC_CL_EVENT_COMPLETE_KHR 0x30FF
#define EGL_NATIVE_BUFFER_TIZEN 0x32A0
#define EGL_NO_DEVICE_EXT ((EGLDeviceEXT)(0))
#define EGL_BAD_DEVICE_EXT 0x322B
#define EGL_DEVICE_EXT 0x322C
#define EGL_LINUX_DMA_BUF_EXT 0x3270
#define EGL_LINUX_DRM_FOURCC_EXT 0x3271
#define EGL_DMA_BUF_PLANE0_FD_EXT 0x3272
#define EGL_DMA_BUF_PLANE0_OFFSET_EXT 0x3273
#define EGL_DMA_BUF_PLANE0_PITCH_EXT 0x3274
#define EGL_DMA_BUF_PLANE1_FD_EXT 0x3275
#define EGL_DMA_BUF_PLANE1_OFFSET_EXT 0x3276
#define EGL_DMA_BUF_PLANE1_PITCH_EXT 0x3277
#define EGL_DMA_BUF_PLANE2_FD_EXT 0x3278
#define EGL_DMA_BUF_PLANE2_OFFSET_EXT 0x3279
#define EGL_DMA_BUF_PLANE2_PITCH_EXT 0x327A
#define EGL_YUV_COLOR_SPACE_HINT_EXT 0x327B
#define EGL_SAMPLE_RANGE_HINT_EXT 0x327C
#define EGL_YUV_CHROMA_HORIZONTAL_SITING_HINT_EXT 0x327D
#define EGL_YUV_CHROMA_VERTICAL_SITING_HINT_EXT 0x327E
#define EGL_ITU_REC601_EXT 0x327F
#define EGL_ITU_REC709_EXT 0x3280
#define EGL_ITU_REC2020_EXT 0x3281
#define EGL_YUV_FULL_RANGE_EXT 0x3282
#define EGL_YUV_NARROW_RANGE_EXT 0x3283
#define EGL_YUV_CHROMA_SITING_0_EXT 0x3284
#define EGL_YUV_CHROMA_SITING_0_5_EXT 0x3285
#define EGL_NO_OUTPUT_LAYER_EXT ((EGLOutputLayerEXT)0)
#define EGL_NO_OUTPUT_PORT_EXT ((EGLOutputPortEXT)0)
#define EGL_BAD_OUTPUT_LAYER_EXT 0x322D
#define EGL_BAD_OUTPUT_PORT_EXT 0x322E
#define EGL_SWAP_INTERVAL_EXT 0x322F
#define EGL_OPENWF_DEVICE_ID_EXT 0x3237
#define EGL_SYNC_STATUS_KHR 0x30F1
#define EGL_SIGNALED_KHR 0x30F2
#define EGL_UNSIGNALED_KHR 0x30F3
#define EGL_TIMEOUT_EXPIRED_KHR 0x30F5
#define EGL_CONDITION_SATISFIED_KHR 0x30F6
#define EGL_SYNC_TYPE_KHR 0x30F7
#define EGL_SYNC_REUSABLE_KHR 0x30FA
#define EGL_SYNC_FLUSH_COMMANDS_BIT_KHR 0x0001
#define EGL_FOREVER_KHR 0xFFFFFFFFFFFFFFFF
#define EGL_NO_SYNC_KHR ((EGLSyncKHR)0)
#define EGL_CONTEXT_PRIORITY_LEVEL_IMG 0x3100
#define EGL_CONTEXT_PRIORITY_HIGH_IMG 0x3101
#define EGL_CONTEXT_PRIORITY_MEDIUM_IMG 0x3102
#define EGL_CONTEXT_PRIORITY_LOW_IMG 0x3103
#define EGL_NO_FILE_DESCRIPTOR_KHR ((EGLNativeFileDescriptorKHR)(-1))
#define EGL_PROTECTED_CONTENT_EXT 0x32C0
#define EGL_COVERAGE_SAMPLE_RESOLVE_NV 0x3131
#define EGL_COVERAGE_SAMPLE_RESOLVE_DEFAULT_NV 0x3132
#define EGL_COVERAGE_SAMPLE_RESOLVE_NONE_NV 0x3133
#define EGL_GL_TEXTURE_3D_KHR 0x30B2
#define EGL_GL_TEXTURE_ZOFFSET_KHR 0x30BD
#define EGL_D3D_TEXTURE_2D_SHARE_HANDLE_ANGLE 0x3200
#define EGL_PLATFORM_DEVICE_EXT 0x313F
#define EGL_D3D9_DEVICE_ANGLE 0x33A0
#define EGL_D3D11_DEVICE_ANGLE 0x33A1
#define EGL_CUDA_EVENT_HANDLE_NV 0x323B
#define EGL_SYNC_CUDA_EVENT_NV 0x323C
#define EGL_SYNC_CUDA_EVENT_COMPLETE_NV 0x323D
#define EGL_SYNC_PRIOR_COMMANDS_COMPLETE_KHR 0x30F0
#define EGL_SYNC_CONDITION_KHR 0x30F8
#define EGL_SYNC_FENCE_KHR 0x30F9
#define EGL_DRM_CRTC_EXT 0x3234
#define EGL_DRM_PLANE_EXT 0x3235
#define EGL_DRM_CONNECTOR_EXT 0x3236
#define EGL_CONTEXT_OPENGL_ROBUST_ACCESS_EXT 0x30BF
#define EGL_CONTEXT_OPENGL_RESET_NOTIFICATION_STRATEGY_EXT 0x3138
#define EGL_NO_RESET_NOTIFICATION_EXT 0x31BE
#define EGL_LOSE_CONTEXT_ON_RESET_EXT 0x31BF
#define EGL_COVERAGE_BUFFERS_NV 0x30E0
#define EGL_COVERAGE_SAMPLES_NV 0x30E1
#define EGL_IMAGE_PRESERVED_KHR 0x30D2
#define EGL_NO_IMAGE_KHR ((EGLImageKHR)0)
#define EGL_SYNC_PRIOR_COMMANDS_COMPLETE_NV 0x30E6
#define EGL_SYNC_STATUS_NV 0x30E7
#define EGL_SIGNALED_NV 0x30E8
#define EGL_UNSIGNALED_NV 0x30E9
#define EGL_SYNC_FLUSH_COMMANDS_BIT_NV 0x0001
#define EGL_FOREVER_NV 0xFFFFFFFFFFFFFFFF
#define EGL_ALREADY_SIGNALED_NV 0x30EA
#define EGL_TIMEOUT_EXPIRED_NV 0x30EB
#define EGL_CONDITION_SATISFIED_NV 0x30EC
#define EGL_SYNC_TYPE_NV 0x30ED
#define EGL_SYNC_CONDITION_NV 0x30EE
#define EGL_SYNC_FENCE_NV 0x30EF
#define EGL_NO_SYNC_NV ((EGLSyncNV)0)
#define EGL_PLATFORM_X11_KHR 0x31D5
#define EGL_PLATFORM_X11_SCREEN_KHR 0x31D6
#define EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR 0x321E
#define EGL_GL_COLORSPACE_KHR 0x309D
#define EGL_GL_COLORSPACE_SRGB_KHR 0x3089
#define EGL_GL_COLORSPACE_LINEAR_KHR 0x308A
#define EGL_BUFFER_AGE_KHR 0x313D
#define EGL_CONTEXT_MAJOR_VERSION_KHR 0x3098
#define EGL_CONTEXT_MINOR_VERSION_KHR 0x30FB
#define EGL_CONTEXT_FLAGS_KHR 0x30FC
#define EGL_CONTEXT_OPENGL_PROFILE_MASK_KHR 0x30FD
#define EGL_CONTEXT_OPENGL_RESET_NOTIFICATION_STRATEGY_KHR 0x31BD
#define EGL_NO_RESET_NOTIFICATION_KHR 0x31BE
#define EGL_LOSE_CONTEXT_ON_RESET_KHR 0x31BF
#define EGL_CONTEXT_OPENGL_DEBUG_BIT_KHR 0x00000001
#define EGL_CONTEXT_OPENGL_FORWARD_COMPATIBLE_BIT_KHR 0x00000002
#define EGL_CONTEXT_OPENGL_ROBUST_ACCESS_BIT_KHR 0x00000004
#define EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT_KHR 0x00000001
#define EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT_KHR 0x00000002
#define EGL_OPENGL_ES3_BIT_KHR 0x00000040
#define EGL_NATIVE_SURFACE_TIZEN 0x32A1
#define EGL_GL_RENDERBUFFER_KHR 0x30B9
#define EGL_PLATFORM_WAYLAND_KHR 0x31D8
#define EGL_PLATFORM_GBM_KHR 0x31D7
#define EGL_DISCARD_SAMPLES_ARM 0x3286
#define EGL_RECORDABLE_ANDROID 0x3142
#define EGL_GL_TEXTURE_2D_KHR 0x30B1
#define EGL_GL_TEXTURE_LEVEL_KHR 0x30BC
#define EGL_SYNC_NATIVE_FENCE_ANDROID 0x3144
#define EGL_SYNC_NATIVE_FENCE_FD_ANDROID 0x3145
#define EGL_SYNC_NATIVE_FENCE_SIGNALED_ANDROID 0x3146
#define EGL_NO_NATIVE_FENCE_FD_ANDROID -1
#define EGL_OPENWF_PIPELINE_ID_EXT 0x3238
#define EGL_OPENWF_PORT_ID_EXT 0x3239
#define EGL_COLOR_FORMAT_HI 0x8F70
#define EGL_COLOR_RGB_HI 0x8F71
#define EGL_COLOR_RGBA_HI 0x8F72
#define EGL_COLOR_ARGB_HI 0x8F73
#define EGL_PLATFORM_ANDROID_KHR 0x3141
#define EGL_PLATFORM_GBM_MESA 0x31D7
#define EGL_MULTIVIEW_VIEW_COUNT_EXT 0x3134
#define EGL_NATIVE_BUFFER_ANDROID 0x3140
#define EGL_BUFFER_AGE_EXT 0x313D
#define EGL_STREAM_FIFO_LENGTH_KHR 0x31FC
#define EGL_STREAM_TIME_NOW_KHR 0x31FD
#define EGL_STREAM_TIME_CONSUMER_KHR 0x31FE
#define EGL_STREAM_TIME_PRODUCER_KHR 0x31FF
#define EGL_POST_SUB_BUFFER_SUPPORTED_NV 0x30BE
#define EGL_FIXED_SIZE_ANGLE 0x3201
#define EGL_GL_TEXTURE_CUBE_MAP_POSITIVE_X_KHR 0x30B3
#define EGL_GL_TEXTURE_CUBE_MAP_NEGATIVE_X_KHR 0x30B4
#define EGL_GL_TEXTURE_CUBE_MAP_POSITIVE_Y_KHR 0x30B5
#define EGL_GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_KHR 0x30B6
#define EGL_GL_TEXTURE_CUBE_MAP_POSITIVE_Z_KHR 0x30B7
#define EGL_GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_KHR 0x30B8
#define EGL_SYNC_NEW_FRAME_NV 0x321F
#define EGL_Y_INVERTED_NOK 0x307F
#define EGL_YUV_ORDER_EXT 0x3301
#define EGL_YUV_NUMBER_OF_PLANES_EXT 0x3311
#define EGL_YUV_SUBSAMPLE_EXT 0x3312
#define EGL_YUV_DEPTH_RANGE_EXT 0x3317
#define EGL_YUV_CSC_STANDARD_EXT 0x330A
#define EGL_YUV_PLANE_BPP_EXT 0x331A
#define EGL_YUV_BUFFER_EXT 0x3300
#define EGL_YUV_ORDER_YUV_EXT 0x3302
#define EGL_YUV_ORDER_YVU_EXT 0x3303
#define EGL_YUV_ORDER_YUYV_EXT 0x3304
#define EGL_YUV_ORDER_UYVY_EXT 0x3305
#define EGL_YUV_ORDER_YVYU_EXT 0x3306
#define EGL_YUV_ORDER_VYUY_EXT 0x3307
#define EGL_YUV_ORDER_AYUV_EXT 0x3308
#define EGL_YUV_SUBSAMPLE_4_2_0_EXT 0x3313
#define EGL_YUV_SUBSAMPLE_4_2_2_EXT 0x3314
#define EGL_YUV_SUBSAMPLE_4_4_4_EXT 0x3315
#define EGL_YUV_DEPTH_RANGE_LIMITED_EXT 0x3318
#define EGL_YUV_DEPTH_RANGE_FULL_EXT 0x3319
#define EGL_YUV_CSC_STANDARD_601_EXT 0x330B
#define EGL_YUV_CSC_STANDARD_709_EXT 0x330C
#define EGL_YUV_CSC_STANDARD_2020_EXT 0x330D
#define EGL_YUV_PLANE_BPP_0_EXT 0x331B
#define EGL_YUV_PLANE_BPP_8_EXT 0x331C
#define EGL_YUV_PLANE_BPP_10_EXT 0x331D
#define EGL_PLATFORM_WAYLAND_EXT 0x31D8
#define EGL_DEPTH_ENCODING_NV 0x30E2
#define EGL_DEPTH_ENCODING_NONE_NV 0
#define EGL_DEPTH_ENCODING_NONLINEAR_NV 0x30E3
#define EGL_VG_PARENT_IMAGE_KHR 0x30BA
#define EGL_CLIENT_PIXMAP_POINTER_HI 0x8F74
#define EGL_NO_STREAM_KHR ((EGLStreamKHR)0)
#define EGL_CONSUMER_LATENCY_USEC_KHR 0x3210
#define EGL_PRODUCER_FRAME_KHR 0x3212
#define EGL_CONSUMER_FRAME_KHR 0x3213
#define EGL_STREAM_STATE_KHR 0x3214
#define EGL_STREAM_STATE_CREATED_KHR 0x3215
#define EGL_STREAM_STATE_CONNECTING_KHR 0x3216
#define EGL_STREAM_STATE_EMPTY_KHR 0x3217
#define EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR 0x3218
#define EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR 0x3219
#define EGL_STREAM_STATE_DISCONNECTED_KHR 0x321A
#define EGL_BAD_STREAM_KHR 0x321B
#define EGL_BAD_STATE_KHR 0x321C
#define EGL_AUTO_STEREO_NV 0x3136
#define EGL_FRAMEBUFFER_TARGET_ANDROID 0x3147
#define EGL_NATIVE_PIXMAP_KHR 0x30B0
#define EGL_BITMAP_PIXEL_SIZE_KHR 0x3110
#define EGL_CONFORMANT_KHR 0x3042
#define EGL_VG_COLORSPACE_LINEAR_BIT_KHR 0x0020
#define EGL_VG_ALPHA_FORMAT_PRE_BIT_KHR 0x0040
#ifndef EGL_KHR_lock_surface
#define EGL_KHR_lock_surface 1
typedef EGLBoolean (APIENTRYP PFNEGLLOCKSURFACEKHRPROC)(EGLDisplay, EGLSurface, const EGLint*);
GLAPI PFNEGLLOCKSURFACEKHRPROC glad_eglLockSurfaceKHR;
#define eglLockSurfaceKHR glad_eglLockSurfaceKHR
typedef EGLBoolean (APIENTRYP PFNEGLUNLOCKSURFACEKHRPROC)(EGLDisplay, EGLSurface);
GLAPI PFNEGLUNLOCKSURFACEKHRPROC glad_eglUnlockSurfaceKHR;
#define eglUnlockSurfaceKHR glad_eglUnlockSurfaceKHR
#endif
#ifndef EGL_NV_device_cuda
#define EGL_NV_device_cuda 1
#endif
#ifndef EGL_KHR_surfaceless_context
#define EGL_KHR_surfaceless_context 1
#endif
#ifndef EGL_EXT_device_enumeration
#define EGL_EXT_device_enumeration 1
typedef EGLBoolean (APIENTRYP PFNEGLQUERYDEVICESEXTPROC)(EGLint, EGLDeviceEXT*, EGLint*);
GLAPI PFNEGLQUERYDEVICESEXTPROC glad_eglQueryDevicesEXT;
#define eglQueryDevicesEXT glad_eglQueryDevicesEXT
#endif
#ifndef EGL_MESA_drm_image
#define EGL_MESA_drm_image 1
typedef EGLImageKHR (APIENTRYP PFNEGLCREATEDRMIMAGEMESAPROC)(EGLDisplay, const EGLint*);
GLAPI PFNEGLCREATEDRMIMAGEMESAPROC glad_eglCreateDRMImageMESA;
#define eglCreateDRMImageMESA glad_eglCreateDRMImageMESA
typedef EGLBoolean (APIENTRYP PFNEGLEXPORTDRMIMAGEMESAPROC)(EGLDisplay, EGLImageKHR, EGLint*, EGLint*, EGLint*);
GLAPI PFNEGLEXPORTDRMIMAGEMESAPROC glad_eglExportDRMImageMESA;
#define eglExportDRMImageMESA glad_eglExportDRMImageMESA
#endif
#ifndef EGL_EXT_device_drm
#define EGL_EXT_device_drm 1
#endif
#ifndef EGL_KHR_stream_producer_eglsurface
#define EGL_KHR_stream_producer_eglsurface 1
typedef EGLSurface (APIENTRYP PFNEGLCREATESTREAMPRODUCERSURFACEKHRPROC)(EGLDisplay, EGLConfig, EGLStreamKHR, const EGLint*);
GLAPI PFNEGLCREATESTREAMPRODUCERSURFACEKHRPROC glad_eglCreateStreamProducerSurfaceKHR;
#define eglCreateStreamProducerSurfaceKHR glad_eglCreateStreamProducerSurfaceKHR
#endif
#ifndef EGL_EXT_platform_x11
#define EGL_EXT_platform_x11 1
#endif
#ifndef EGL_KHR_cl_event
#define EGL_KHR_cl_event 1
#endif
#ifndef EGL_TIZEN_image_native_buffer
#define EGL_TIZEN_image_native_buffer 1
#endif
#ifndef EGL_EXT_device_base
#define EGL_EXT_device_base 1
typedef EGLBoolean (APIENTRYP PFNEGLQUERYDEVICEATTRIBEXTPROC)(EGLDeviceEXT, EGLint, EGLAttrib*);
GLAPI PFNEGLQUERYDEVICEATTRIBEXTPROC glad_eglQueryDeviceAttribEXT;
#define eglQueryDeviceAttribEXT glad_eglQueryDeviceAttribEXT
typedef const char* (APIENTRYP PFNEGLQUERYDEVICESTRINGEXTPROC)(EGLDeviceEXT, EGLint);
GLAPI PFNEGLQUERYDEVICESTRINGEXTPROC glad_eglQueryDeviceStringEXT;
#define eglQueryDeviceStringEXT glad_eglQueryDeviceStringEXT
typedef EGLBoolean (APIENTRYP PFNEGLQUERYDISPLAYATTRIBEXTPROC)(EGLDisplay, EGLint, EGLAttrib*);
GLAPI PFNEGLQUERYDISPLAYATTRIBEXTPROC glad_eglQueryDisplayAttribEXT;
#define eglQueryDisplayAttribEXT glad_eglQueryDisplayAttribEXT
#endif
#ifndef EGL_MESA_image_dma_buf_export
#define EGL_MESA_image_dma_buf_export 1
typedef EGLBoolean (APIENTRYP PFNEGLEXPORTDMABUFIMAGEQUERYMESAPROC)(EGLDisplay, EGLImageKHR, int*, int*, EGLuint64KHR*);
GLAPI PFNEGLEXPORTDMABUFIMAGEQUERYMESAPROC glad_eglExportDMABUFImageQueryMESA;
#define eglExportDMABUFImageQueryMESA glad_eglExportDMABUFImageQueryMESA
typedef EGLBoolean (APIENTRYP PFNEGLEXPORTDMABUFIMAGEMESAPROC)(EGLDisplay, EGLImageKHR, int*, EGLint*, EGLint*);
GLAPI PFNEGLEXPORTDMABUFIMAGEMESAPROC glad_eglExportDMABUFImageMESA;
#define eglExportDMABUFImageMESA glad_eglExportDMABUFImageMESA
#endif
#ifndef EGL_EXT_image_dma_buf_import
#define EGL_EXT_image_dma_buf_import 1
#endif
#ifndef EGL_NV_system_time
#define EGL_NV_system_time 1
typedef EGLuint64NV (APIENTRYP PFNEGLGETSYSTEMTIMEFREQUENCYNVPROC)();
GLAPI PFNEGLGETSYSTEMTIMEFREQUENCYNVPROC glad_eglGetSystemTimeFrequencyNV;
#define eglGetSystemTimeFrequencyNV glad_eglGetSystemTimeFrequencyNV
typedef EGLuint64NV (APIENTRYP PFNEGLGETSYSTEMTIMENVPROC)();
GLAPI PFNEGLGETSYSTEMTIMENVPROC glad_eglGetSystemTimeNV;
#define eglGetSystemTimeNV glad_eglGetSystemTimeNV
#endif
#ifndef EGL_EXT_output_base
#define EGL_EXT_output_base 1
typedef EGLBoolean (APIENTRYP PFNEGLGETOUTPUTLAYERSEXTPROC)(EGLDisplay, const EGLAttrib*, EGLOutputLayerEXT*, EGLint, EGLint*);
GLAPI PFNEGLGETOUTPUTLAYERSEXTPROC glad_eglGetOutputLayersEXT;
#define eglGetOutputLayersEXT glad_eglGetOutputLayersEXT
typedef EGLBoolean (APIENTRYP PFNEGLGETOUTPUTPORTSEXTPROC)(EGLDisplay, const EGLAttrib*, EGLOutputPortEXT*, EGLint, EGLint*);
GLAPI PFNEGLGETOUTPUTPORTSEXTPROC glad_eglGetOutputPortsEXT;
#define eglGetOutputPortsEXT glad_eglGetOutputPortsEXT
typedef EGLBoolean (APIENTRYP PFNEGLOUTPUTLAYERATTRIBEXTPROC)(EGLDisplay, EGLOutputLayerEXT, EGLint, EGLAttrib);
GLAPI PFNEGLOUTPUTLAYERATTRIBEXTPROC glad_eglOutputLayerAttribEXT;
#define eglOutputLayerAttribEXT glad_eglOutputLayerAttribEXT
typedef EGLBoolean (APIENTRYP PFNEGLQUERYOUTPUTLAYERATTRIBEXTPROC)(EGLDisplay, EGLOutputLayerEXT, EGLint, EGLAttrib*);
GLAPI PFNEGLQUERYOUTPUTLAYERATTRIBEXTPROC glad_eglQueryOutputLayerAttribEXT;
#define eglQueryOutputLayerAttribEXT glad_eglQueryOutputLayerAttribEXT
typedef const char* (APIENTRYP PFNEGLQUERYOUTPUTLAYERSTRINGEXTPROC)(EGLDisplay, EGLOutputLayerEXT, EGLint);
GLAPI PFNEGLQUERYOUTPUTLAYERSTRINGEXTPROC glad_eglQueryOutputLayerStringEXT;
#define eglQueryOutputLayerStringEXT glad_eglQueryOutputLayerStringEXT
typedef EGLBoolean (APIENTRYP PFNEGLOUTPUTPORTATTRIBEXTPROC)(EGLDisplay, EGLOutputPortEXT, EGLint, EGLAttrib);
GLAPI PFNEGLOUTPUTPORTATTRIBEXTPROC glad_eglOutputPortAttribEXT;
#define eglOutputPortAttribEXT glad_eglOutputPortAttribEXT
typedef EGLBoolean (APIENTRYP PFNEGLQUERYOUTPUTPORTATTRIBEXTPROC)(EGLDisplay, EGLOutputPortEXT, EGLint, EGLAttrib*);
GLAPI PFNEGLQUERYOUTPUTPORTATTRIBEXTPROC glad_eglQueryOutputPortAttribEXT;
#define eglQueryOutputPortAttribEXT glad_eglQueryOutputPortAttribEXT
typedef const char* (APIENTRYP PFNEGLQUERYOUTPUTPORTSTRINGEXTPROC)(EGLDisplay, EGLOutputPortEXT, EGLint);
GLAPI PFNEGLQUERYOUTPUTPORTSTRINGEXTPROC glad_eglQueryOutputPortStringEXT;
#define eglQueryOutputPortStringEXT glad_eglQueryOutputPortStringEXT
#endif
#ifndef EGL_EXT_device_openwf
#define EGL_EXT_device_openwf 1
#endif
#ifndef EGL_KHR_reusable_sync
#define EGL_KHR_reusable_sync 1
typedef EGLSyncKHR (APIENTRYP PFNEGLCREATESYNCKHRPROC)(EGLDisplay, EGLenum, const EGLint*);
GLAPI PFNEGLCREATESYNCKHRPROC glad_eglCreateSyncKHR;
#define eglCreateSyncKHR glad_eglCreateSyncKHR
typedef EGLBoolean (APIENTRYP PFNEGLDESTROYSYNCKHRPROC)(EGLDisplay, EGLSyncKHR);
GLAPI PFNEGLDESTROYSYNCKHRPROC glad_eglDestroySyncKHR;
#define eglDestroySyncKHR glad_eglDestroySyncKHR
typedef EGLint (APIENTRYP PFNEGLCLIENTWAITSYNCKHRPROC)(EGLDisplay, EGLSyncKHR, EGLint, EGLTimeKHR);
GLAPI PFNEGLCLIENTWAITSYNCKHRPROC glad_eglClientWaitSyncKHR;
#define eglClientWaitSyncKHR glad_eglClientWaitSyncKHR
typedef EGLBoolean (APIENTRYP PFNEGLSIGNALSYNCKHRPROC)(EGLDisplay, EGLSyncKHR, EGLenum);
GLAPI PFNEGLSIGNALSYNCKHRPROC glad_eglSignalSyncKHR;
#define eglSignalSyncKHR glad_eglSignalSyncKHR
typedef EGLBoolean (APIENTRYP PFNEGLGETSYNCATTRIBKHRPROC)(EGLDisplay, EGLSyncKHR, EGLint, EGLint*);
GLAPI PFNEGLGETSYNCATTRIBKHRPROC glad_eglGetSyncAttribKHR;
#define eglGetSyncAttribKHR glad_eglGetSyncAttribKHR
#endif
#ifndef EGL_IMG_context_priority
#define EGL_IMG_context_priority 1
#endif
#ifndef EGL_KHR_stream_cross_process_fd
#define EGL_KHR_stream_cross_process_fd 1
typedef EGLNativeFileDescriptorKHR (APIENTRYP PFNEGLGETSTREAMFILEDESCRIPTORKHRPROC)(EGLDisplay, EGLStreamKHR);
GLAPI PFNEGLGETSTREAMFILEDESCRIPTORKHRPROC glad_eglGetStreamFileDescriptorKHR;
#define eglGetStreamFileDescriptorKHR glad_eglGetStreamFileDescriptorKHR
typedef EGLStreamKHR (APIENTRYP PFNEGLCREATESTREAMFROMFILEDESCRIPTORKHRPROC)(EGLDisplay, EGLNativeFileDescriptorKHR);
GLAPI PFNEGLCREATESTREAMFROMFILEDESCRIPTORKHRPROC glad_eglCreateStreamFromFileDescriptorKHR;
#define eglCreateStreamFromFileDescriptorKHR glad_eglCreateStreamFromFileDescriptorKHR
#endif
#ifndef EGL_EXT_protected_surface
#define EGL_EXT_protected_surface 1
#endif
#ifndef EGL_NV_coverage_sample_resolve
#define EGL_NV_coverage_sample_resolve 1
#endif
#ifndef EGL_EXT_device_query
#define EGL_EXT_device_query 1
#endif
#ifndef EGL_KHR_gl_texture_3D_image
#define EGL_KHR_gl_texture_3D_image 1
#endif
#ifndef EGL_EXT_stream_consumer_egloutput
#define EGL_EXT_stream_consumer_egloutput 1
typedef EGLBoolean (APIENTRYP PFNEGLSTREAMCONSUMEROUTPUTEXTPROC)(EGLDisplay, EGLStreamKHR, EGLOutputLayerEXT);
GLAPI PFNEGLSTREAMCONSUMEROUTPUTEXTPROC glad_eglStreamConsumerOutputEXT;
#define eglStreamConsumerOutputEXT glad_eglStreamConsumerOutputEXT
#endif
#ifndef EGL_ANGLE_surface_d3d_texture_2d_share_handle
#define EGL_ANGLE_surface_d3d_texture_2d_share_handle 1
#endif
#ifndef EGL_EXT_platform_device
#define EGL_EXT_platform_device 1
#endif
#ifndef EGL_ANGLE_device_d3d
#define EGL_ANGLE_device_d3d 1
#endif
#ifndef EGL_KHR_stream_producer_aldatalocator
#define EGL_KHR_stream_producer_aldatalocator 1
#endif
#ifndef EGL_NV_cuda_event
#define EGL_NV_cuda_event 1
#endif
#ifndef EGL_KHR_swap_buffers_with_damage
#define EGL_KHR_swap_buffers_with_damage 1
typedef EGLBoolean (APIENTRYP PFNEGLSWAPBUFFERSWITHDAMAGEKHRPROC)(EGLDisplay, EGLSurface, EGLint*, EGLint);
GLAPI PFNEGLSWAPBUFFERSWITHDAMAGEKHRPROC glad_eglSwapBuffersWithDamageKHR;
#define eglSwapBuffersWithDamageKHR glad_eglSwapBuffersWithDamageKHR
#endif
#ifndef EGL_KHR_fence_sync
#define EGL_KHR_fence_sync 1
#endif
#ifndef EGL_KHR_cl_event2
#define EGL_KHR_cl_event2 1
typedef EGLSyncKHR (APIENTRYP PFNEGLCREATESYNC64KHRPROC)(EGLDisplay, EGLenum, const EGLAttribKHR*);
GLAPI PFNEGLCREATESYNC64KHRPROC glad_eglCreateSync64KHR;
#define eglCreateSync64KHR glad_eglCreateSync64KHR
#endif
#ifndef EGL_EXT_output_drm
#define EGL_EXT_output_drm 1
#endif
#ifndef EGL_EXT_create_context_robustness
#define EGL_EXT_create_context_robustness 1
#endif
#ifndef EGL_NV_coverage_sample
#define EGL_NV_coverage_sample 1
#endif
#ifndef EGL_KHR_image_base
#define EGL_KHR_image_base 1
typedef EGLImageKHR (APIENTRYP PFNEGLCREATEIMAGEKHRPROC)(EGLDisplay, EGLContext, EGLenum, EGLClientBuffer, const EGLint*);
GLAPI PFNEGLCREATEIMAGEKHRPROC glad_eglCreateImageKHR;
#define eglCreateImageKHR glad_eglCreateImageKHR
typedef EGLBoolean (APIENTRYP PFNEGLDESTROYIMAGEKHRPROC)(EGLDisplay, EGLImageKHR);
GLAPI PFNEGLDESTROYIMAGEKHRPROC glad_eglDestroyImageKHR;
#define eglDestroyImageKHR glad_eglDestroyImageKHR
#endif
#ifndef EGL_ANDROID_blob_cache
#define EGL_ANDROID_blob_cache 1
typedef void (APIENTRYP PFNEGLSETBLOBCACHEFUNCSANDROIDPROC)(EGLDisplay, EGLSetBlobFuncANDROID, EGLGetBlobFuncANDROID);
GLAPI PFNEGLSETBLOBCACHEFUNCSANDROIDPROC glad_eglSetBlobCacheFuncsANDROID;
#define eglSetBlobCacheFuncsANDROID glad_eglSetBlobCacheFuncsANDROID
#endif
#ifndef EGL_NV_sync
#define EGL_NV_sync 1
typedef EGLSyncNV (APIENTRYP PFNEGLCREATEFENCESYNCNVPROC)(EGLDisplay, EGLenum, const EGLint*);
GLAPI PFNEGLCREATEFENCESYNCNVPROC glad_eglCreateFenceSyncNV;
#define eglCreateFenceSyncNV glad_eglCreateFenceSyncNV
typedef EGLBoolean (APIENTRYP PFNEGLDESTROYSYNCNVPROC)(EGLSyncNV);
GLAPI PFNEGLDESTROYSYNCNVPROC glad_eglDestroySyncNV;
#define eglDestroySyncNV glad_eglDestroySyncNV
typedef EGLBoolean (APIENTRYP PFNEGLFENCENVPROC)(EGLSyncNV);
GLAPI PFNEGLFENCENVPROC glad_eglFenceNV;
#define eglFenceNV glad_eglFenceNV
typedef EGLint (APIENTRYP PFNEGLCLIENTWAITSYNCNVPROC)(EGLSyncNV, EGLint, EGLTimeNV);
GLAPI PFNEGLCLIENTWAITSYNCNVPROC glad_eglClientWaitSyncNV;
#define eglClientWaitSyncNV glad_eglClientWaitSyncNV
typedef EGLBoolean (APIENTRYP PFNEGLSIGNALSYNCNVPROC)(EGLSyncNV, EGLenum);
GLAPI PFNEGLSIGNALSYNCNVPROC glad_eglSignalSyncNV;
#define eglSignalSyncNV glad_eglSignalSyncNV
typedef EGLBoolean (APIENTRYP PFNEGLGETSYNCATTRIBNVPROC)(EGLSyncNV, EGLint, EGLint*);
GLAPI PFNEGLGETSYNCATTRIBNVPROC glad_eglGetSyncAttribNV;
#define eglGetSyncAttribNV glad_eglGetSyncAttribNV
#endif
#ifndef EGL_KHR_platform_x11
#define EGL_KHR_platform_x11 1
#endif
#ifndef EGL_ANGLE_d3d_share_handle_client_buffer
#define EGL_ANGLE_d3d_share_handle_client_buffer 1
#endif
#ifndef EGL_NV_native_query
#define EGL_NV_native_query 1
typedef EGLBoolean (APIENTRYP PFNEGLQUERYNATIVEDISPLAYNVPROC)(EGLDisplay, EGLNativeDisplayType*);
GLAPI PFNEGLQUERYNATIVEDISPLAYNVPROC glad_eglQueryNativeDisplayNV;
#define eglQueryNativeDisplayNV glad_eglQueryNativeDisplayNV
typedef EGLBoolean (APIENTRYP PFNEGLQUERYNATIVEWINDOWNVPROC)(EGLDisplay, EGLSurface, EGLNativeWindowType*);
GLAPI PFNEGLQUERYNATIVEWINDOWNVPROC glad_eglQueryNativeWindowNV;
#define eglQueryNativeWindowNV glad_eglQueryNativeWindowNV
typedef EGLBoolean (APIENTRYP PFNEGLQUERYNATIVEPIXMAPNVPROC)(EGLDisplay, EGLSurface, EGLNativePixmapType*);
GLAPI PFNEGLQUERYNATIVEPIXMAPNVPROC glad_eglQueryNativePixmapNV;
#define eglQueryNativePixmapNV glad_eglQueryNativePixmapNV
#endif
#ifndef EGL_KHR_stream_consumer_gltexture
#define EGL_KHR_stream_consumer_gltexture 1
typedef EGLBoolean (APIENTRYP PFNEGLSTREAMCONSUMERGLTEXTUREEXTERNALKHRPROC)(EGLDisplay, EGLStreamKHR);
GLAPI PFNEGLSTREAMCONSUMERGLTEXTUREEXTERNALKHRPROC glad_eglStreamConsumerGLTextureExternalKHR;
#define eglStreamConsumerGLTextureExternalKHR glad_eglStreamConsumerGLTextureExternalKHR
typedef EGLBoolean (APIENTRYP PFNEGLSTREAMCONSUMERACQUIREKHRPROC)(EGLDisplay, EGLStreamKHR);
GLAPI PFNEGLSTREAMCONSUMERACQUIREKHRPROC glad_eglStreamConsumerAcquireKHR;
#define eglStreamConsumerAcquireKHR glad_eglStreamConsumerAcquireKHR
typedef EGLBoolean (APIENTRYP PFNEGLSTREAMCONSUMERRELEASEKHRPROC)(EGLDisplay, EGLStreamKHR);
GLAPI PFNEGLSTREAMCONSUMERRELEASEKHRPROC glad_eglStreamConsumerReleaseKHR;
#define eglStreamConsumerReleaseKHR glad_eglStreamConsumerReleaseKHR
#endif
#ifndef EGL_KHR_gl_colorspace
#define EGL_KHR_gl_colorspace 1
#endif
#ifndef EGL_KHR_partial_update
#define EGL_KHR_partial_update 1
typedef EGLBoolean (APIENTRYP PFNEGLSETDAMAGEREGIONKHRPROC)(EGLDisplay, EGLSurface, EGLint*, EGLint);
GLAPI PFNEGLSETDAMAGEREGIONKHRPROC glad_eglSetDamageRegionKHR;
#define eglSetDamageRegionKHR glad_eglSetDamageRegionKHR
#endif
#ifndef EGL_KHR_get_all_proc_addresses
#define EGL_KHR_get_all_proc_addresses 1
#endif
#ifndef EGL_KHR_create_context
#define EGL_KHR_create_context 1
#endif
#ifndef EGL_TIZEN_image_native_surface
#define EGL_TIZEN_image_native_surface 1
#endif
#ifndef EGL_KHR_gl_renderbuffer_image
#define EGL_KHR_gl_renderbuffer_image 1
#endif
#ifndef EGL_KHR_platform_wayland
#define EGL_KHR_platform_wayland 1
#endif
#ifndef EGL_KHR_platform_gbm
#define EGL_KHR_platform_gbm 1
#endif
#ifndef EGL_ARM_pixmap_multisample_discard
#define EGL_ARM_pixmap_multisample_discard 1
#endif
#ifndef EGL_KHR_wait_sync
#define EGL_KHR_wait_sync 1
typedef EGLint (APIENTRYP PFNEGLWAITSYNCKHRPROC)(EGLDisplay, EGLSyncKHR, EGLint);
GLAPI PFNEGLWAITSYNCKHRPROC glad_eglWaitSyncKHR;
#define eglWaitSyncKHR glad_eglWaitSyncKHR
#endif
#ifndef EGL_ANDROID_recordable
#define EGL_ANDROID_recordable 1
#endif
#ifndef EGL_KHR_gl_texture_2D_image
#define EGL_KHR_gl_texture_2D_image 1
#endif
#ifndef EGL_ANDROID_native_fence_sync
#define EGL_ANDROID_native_fence_sync 1
typedef EGLint (APIENTRYP PFNEGLDUPNATIVEFENCEFDANDROIDPROC)(EGLDisplay, EGLSyncKHR);
GLAPI PFNEGLDUPNATIVEFENCEFDANDROIDPROC glad_eglDupNativeFenceFDANDROID;
#define eglDupNativeFenceFDANDROID glad_eglDupNativeFenceFDANDROID
#endif
#ifndef EGL_EXT_output_openwf
#define EGL_EXT_output_openwf 1
#endif
#ifndef EGL_HI_colorformats
#define EGL_HI_colorformats 1
#endif
#ifndef EGL_KHR_platform_android
#define EGL_KHR_platform_android 1
#endif
#ifndef EGL_MESA_platform_gbm
#define EGL_MESA_platform_gbm 1
#endif
#ifndef EGL_EXT_multiview_window
#define EGL_EXT_multiview_window 1
#endif
#ifndef EGL_EXT_platform_base
#define EGL_EXT_platform_base 1
typedef EGLDisplay (APIENTRYP PFNEGLGETPLATFORMDISPLAYEXTPROC)(EGLenum, void*, const EGLint*);
GLAPI PFNEGLGETPLATFORMDISPLAYEXTPROC glad_eglGetPlatformDisplayEXT;
#define eglGetPlatformDisplayEXT glad_eglGetPlatformDisplayEXT
typedef EGLSurface (APIENTRYP PFNEGLCREATEPLATFORMWINDOWSURFACEEXTPROC)(EGLDisplay, EGLConfig, void*, const EGLint*);
GLAPI PFNEGLCREATEPLATFORMWINDOWSURFACEEXTPROC glad_eglCreatePlatformWindowSurfaceEXT;
#define eglCreatePlatformWindowSurfaceEXT glad_eglCreatePlatformWindowSurfaceEXT
typedef EGLSurface (APIENTRYP PFNEGLCREATEPLATFORMPIXMAPSURFACEEXTPROC)(EGLDisplay, EGLConfig, void*, const EGLint*);
GLAPI PFNEGLCREATEPLATFORMPIXMAPSURFACEEXTPROC glad_eglCreatePlatformPixmapSurfaceEXT;
#define eglCreatePlatformPixmapSurfaceEXT glad_eglCreatePlatformPixmapSurfaceEXT
#endif
#ifndef EGL_ANDROID_image_native_buffer
#define EGL_ANDROID_image_native_buffer 1
#endif
#ifndef EGL_EXT_buffer_age
#define EGL_EXT_buffer_age 1
#endif
#ifndef EGL_KHR_stream_fifo
#define EGL_KHR_stream_fifo 1
typedef EGLBoolean (APIENTRYP PFNEGLQUERYSTREAMTIMEKHRPROC)(EGLDisplay, EGLStreamKHR, EGLenum, EGLTimeKHR*);
GLAPI PFNEGLQUERYSTREAMTIMEKHRPROC glad_eglQueryStreamTimeKHR;
#define eglQueryStreamTimeKHR glad_eglQueryStreamTimeKHR
#endif
#ifndef EGL_EXT_client_extensions
#define EGL_EXT_client_extensions 1
#endif
#ifndef EGL_NV_post_sub_buffer
#define EGL_NV_post_sub_buffer 1
typedef EGLBoolean (APIENTRYP PFNEGLPOSTSUBBUFFERNVPROC)(EGLDisplay, EGLSurface, EGLint, EGLint, EGLint, EGLint);
GLAPI PFNEGLPOSTSUBBUFFERNVPROC glad_eglPostSubBufferNV;
#define eglPostSubBufferNV glad_eglPostSubBufferNV
#endif
#ifndef EGL_ANGLE_window_fixed_size
#define EGL_ANGLE_window_fixed_size 1
#endif
#ifndef EGL_NOK_swap_region2
#define EGL_NOK_swap_region2 1
typedef EGLBoolean (APIENTRYP PFNEGLSWAPBUFFERSREGION2NOKPROC)(EGLDisplay, EGLSurface, EGLint, const EGLint*);
GLAPI PFNEGLSWAPBUFFERSREGION2NOKPROC glad_eglSwapBuffersRegion2NOK;
#define eglSwapBuffersRegion2NOK glad_eglSwapBuffersRegion2NOK
#endif
#ifndef EGL_NV_post_convert_rounding
#define EGL_NV_post_convert_rounding 1
#endif
#ifndef EGL_KHR_gl_texture_cubemap_image
#define EGL_KHR_gl_texture_cubemap_image 1
#endif
#ifndef EGL_NV_stream_sync
#define EGL_NV_stream_sync 1
typedef EGLSyncKHR (APIENTRYP PFNEGLCREATESTREAMSYNCNVPROC)(EGLDisplay, EGLStreamKHR, EGLenum, const EGLint*);
GLAPI PFNEGLCREATESTREAMSYNCNVPROC glad_eglCreateStreamSyncNV;
#define eglCreateStreamSyncNV glad_eglCreateStreamSyncNV
#endif
#ifndef EGL_NOK_texture_from_pixmap
#define EGL_NOK_texture_from_pixmap 1
#endif
#ifndef EGL_EXT_yuv_surface
#define EGL_EXT_yuv_surface 1
#endif
#ifndef EGL_EXT_swap_buffers_with_damage
#define EGL_EXT_swap_buffers_with_damage 1
typedef EGLBoolean (APIENTRYP PFNEGLSWAPBUFFERSWITHDAMAGEEXTPROC)(EGLDisplay, EGLSurface, EGLint*, EGLint);
GLAPI PFNEGLSWAPBUFFERSWITHDAMAGEEXTPROC glad_eglSwapBuffersWithDamageEXT;
#define eglSwapBuffersWithDamageEXT glad_eglSwapBuffersWithDamageEXT
#endif
#ifndef EGL_EXT_platform_wayland
#define EGL_EXT_platform_wayland 1
#endif
#ifndef EGL_NV_depth_nonlinear
#define EGL_NV_depth_nonlinear 1
#endif
#ifndef EGL_KHR_vg_parent_image
#define EGL_KHR_vg_parent_image 1
#endif
#ifndef EGL_HI_clientpixmap
#define EGL_HI_clientpixmap 1
typedef EGLSurface (APIENTRYP PFNEGLCREATEPIXMAPSURFACEHIPROC)(EGLDisplay, EGLConfig, struct EGLClientPixmapHI*);
GLAPI PFNEGLCREATEPIXMAPSURFACEHIPROC glad_eglCreatePixmapSurfaceHI;
#define eglCreatePixmapSurfaceHI glad_eglCreatePixmapSurfaceHI
#endif
#ifndef EGL_KHR_stream
#define EGL_KHR_stream 1
typedef EGLStreamKHR (APIENTRYP PFNEGLCREATESTREAMKHRPROC)(EGLDisplay, const EGLint*);
GLAPI PFNEGLCREATESTREAMKHRPROC glad_eglCreateStreamKHR;
#define eglCreateStreamKHR glad_eglCreateStreamKHR
typedef EGLBoolean (APIENTRYP PFNEGLDESTROYSTREAMKHRPROC)(EGLDisplay, EGLStreamKHR);
GLAPI PFNEGLDESTROYSTREAMKHRPROC glad_eglDestroyStreamKHR;
#define eglDestroyStreamKHR glad_eglDestroyStreamKHR
typedef EGLBoolean (APIENTRYP PFNEGLSTREAMATTRIBKHRPROC)(EGLDisplay, EGLStreamKHR, EGLenum, EGLint);
GLAPI PFNEGLSTREAMATTRIBKHRPROC glad_eglStreamAttribKHR;
#define eglStreamAttribKHR glad_eglStreamAttribKHR
typedef EGLBoolean (APIENTRYP PFNEGLQUERYSTREAMKHRPROC)(EGLDisplay, EGLStreamKHR, EGLenum, EGLint*);
GLAPI PFNEGLQUERYSTREAMKHRPROC glad_eglQueryStreamKHR;
#define eglQueryStreamKHR glad_eglQueryStreamKHR
typedef EGLBoolean (APIENTRYP PFNEGLQUERYSTREAMU64KHRPROC)(EGLDisplay, EGLStreamKHR, EGLenum, EGLuint64KHR*);
GLAPI PFNEGLQUERYSTREAMU64KHRPROC glad_eglQueryStreamu64KHR;
#define eglQueryStreamu64KHR glad_eglQueryStreamu64KHR
#endif
#ifndef EGL_NV_3dvision_surface
#define EGL_NV_3dvision_surface 1
#endif
#ifndef EGL_ANDROID_framebuffer_target
#define EGL_ANDROID_framebuffer_target 1
#endif
#ifndef EGL_ANGLE_query_surface_pointer
#define EGL_ANGLE_query_surface_pointer 1
typedef EGLBoolean (APIENTRYP PFNEGLQUERYSURFACEPOINTERANGLEPROC)(EGLDisplay, EGLSurface, EGLint, void**);
GLAPI PFNEGLQUERYSURFACEPOINTERANGLEPROC glad_eglQuerySurfacePointerANGLE;
#define eglQuerySurfacePointerANGLE glad_eglQuerySurfacePointerANGLE
#endif
#ifndef EGL_KHR_image_pixmap
#define EGL_KHR_image_pixmap 1
#endif
#ifndef EGL_KHR_lock_surface3
#define EGL_KHR_lock_surface3 1
typedef EGLBoolean (APIENTRYP PFNEGLQUERYSURFACE64KHRPROC)(EGLDisplay, EGLSurface, EGLint, EGLAttribKHR*);
GLAPI PFNEGLQUERYSURFACE64KHRPROC glad_eglQuerySurface64KHR;
#define eglQuerySurface64KHR glad_eglQuerySurface64KHR
#endif
#ifndef EGL_KHR_lock_surface2
#define EGL_KHR_lock_surface2 1
#endif
#ifndef EGL_KHR_config_attribs
#define EGL_KHR_config_attribs 1
#endif
#ifndef EGL_KHR_image
#define EGL_KHR_image 1
#endif
#ifndef EGL_KHR_client_get_all_proc_addresses
#define EGL_KHR_client_get_all_proc_addresses 1
#endif
#ifndef EGL_NOK_swap_region
#define EGL_NOK_swap_region 1
typedef EGLBoolean (APIENTRYP PFNEGLSWAPBUFFERSREGIONNOKPROC)(EGLDisplay, EGLSurface, EGLint, const EGLint*);
GLAPI PFNEGLSWAPBUFFERSREGIONNOKPROC glad_eglSwapBuffersRegionNOK;
#define eglSwapBuffersRegionNOK glad_eglSwapBuffersRegionNOK
#endif

#ifdef __cplusplus
}
#endif

#endif
