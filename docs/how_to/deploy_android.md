How to deploy and use compiled model on Android
===============================================

This tutorial explain below aspects (Unlike the android_rpc approach we already have)

    1: Build a model for android target.
    2: TVM runtime building for android target.
    3: A sample native application to test the model on ADB before integrating into android application (apk).

Hope once we have working sample at android native, Android app developers know how to integrate with JNI.

As an example here is a reference block diagram.

![](https://github.com/dmlc/tvm/tree/master/docs/how_to/android_deploy.png)

Build model for Android Target
------------------------------

NNVM compilation of model for android target could follow same approach like android_rpc.

An reference exampe can be found at [chainer-nnvm-example](https://github.com/tkat0/chainer-nnvm-example)

Above example will directly run the compiled model on RPC target. Below modification at [rum_mobile.py](https://github.com/tkat0/chainer-nnvm-example/blob/5b97fd4d41aa4dde4b0aceb0be311054fb5de451/run_mobile.py#L64) will save the compilation output which is required on android target.

```
    lib.export_library("YOLOv2_tiny-aarch64.so", ndk.create_shared)

    with open("YOLOv2_tiny-aarch64.json", "w") as fo:
            fo.write(graph.json())
    with open("YOLOv2_tiny-aarch64.params", "wb") as fo:
            fo.write(nnvm.compiler.save_param_dict(params))
```

YOLOv2_tiny-aarch64.so, YOLOv2_tiny-aarch64.json, YOLOv2_tiny-aarch64.params will go to android target.


TVM Runtime for android Target
-------------------------------

This is a cross compilation process of libtvm_runtime.so for android with OpenCL support.


Prerequisites:
[Architecture and Android Standalone Toolchain](https://github.com/dmlc/tvm/tree/master/apps/android_rpc#cross-compile-and-run-on-android-devices)
Android OpenCL dependencies as shown below under OPENCL_PATH, libOpenCL.so is copied from target device (can be find under phone system/lib64/ or system/lib/ partition)

```
.
├── include
│   └── CL
│       ├── cl_d3d10.h
│       ├── cl_d3d11.h
│       ├── cl_dx9_media_sharing.h
│       ├── cl_dx9_media_sharing_intel.h
│       ├── cl_egl.h
│       ├── cl_ext.h
│       ├── cl_ext_intel.h
│       ├── cl_gl_ext.h
│       ├── cl_gl.h
│       ├── cl.h
│       ├── cl_platform.h
│       ├── cl_va_api_media_sharing_intel.h
│       └── opencl.h
└── lib
    └── libOpenCL.so
```

- Enable OPENCL in make/config.mk

- Below command can build libtvm_runtime.so for android.

```
    OPENCL_PATH=<path to OPENCL path> CXX=<ANDROID_NDK_PATH>/ndk/android-toolchain-arm64/bin/aarch64-linux-android-g++ make lib/libtvm_runtime.so
```

Result of this setup is the libtvm_runtime.so which is a dependency for Android native while building application.
The same can be deployed under /system/lib64/ as a system library too.


Android Native API Reference
----------------------------

From android native TVM API to load model & execute can be refered at this [native](https://github.com/dmlc/tvm/tree/master/tutorials/deployment) sample source.


