# How to deploy and use compiled model on Android


This tutorial explain below aspects (Unlike the android_rpc approach we already have)

  * Build a model for android target
  * TVM run on Android using Java API
  * TVM run on Android using Native API

As an example here is a reference block diagram.

![](http://www.tvmlang.org/images/release/tvm_flexible.png)

## Build model for Android Target

NNVM compilation of model for android target could follow same approach like android_rpc.

An reference exampe can be found at [chainer-nnvm-example](https://github.com/tkat0/chainer-nnvm-example)

Above example will directly run the compiled model on RPC target. Below modification at [rum_mobile.py](https://github.com/tkat0/chainer-nnvm-example/blob/5b97fd4d41aa4dde4b0aceb0be311054fb5de451/run_mobile.py#L64) will save the compilation output which is required on android target.

```
lib.export_library("deploy_lib.so", ndk.create_shared)
with open("deploy_graph.json", "w") as fo:
    fo.write(graph.json())
with open("deploy_param.params", "wb") as fo:
    fo.write(nnvm.compiler.save_param_dict(params))
```

deploy_lib.so, deploy_graph.json, deploy_param.params will go to android target.

## TVM run on Android using Java API
### TVM Runtime for android Target

Refer [here](https://github.com/dmlc/tvm/blob/master/apps/android_deploy/README.md#build-and-installation) to build CPU/OpenCL version flavor TVM runtime for android target.

### Android Native API Reference

From android java TVM API to load model & execute can be refered at this [java](https://github.com/dmlc/tvm/blob/master/apps/android_deploy/app/src/main/java/ml/dmlc/tvm/android/demo/MainActivity.java) sample source.


## TVM run on Android using Native API

### TVM Runtime for android Target

This is a cross compilation process of libtvm_runtime.so for android with OpenCL support.

Prerequisites:
- Android stand alone tool chain : ANDROID_NDK_PATH/ndk/android-toolchain-arm64/bin/aarch64-linux-android-g++.

  Please refer [Android NDK toolchain](https://developer.android.com/ndk/guides/standalone_toolchain.html) to generate standalone toolchain for your android device.

Android OpenCL dependencies as shown below under OPENCL_PATH. These can be picked from your android build which is compatible to the target.

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
OPENCL_PATH=<path to OPENCL> CXX=<ANDROID_NDK_PATH>/ndk/android-toolchain-arm64/bin/aarch64-linux-android-g++ make lib/libtvm_runtime.so
```

Result of this step is the libtvm_runtime.so which is a dependency for Android native while building application.
The same can be deployed under /system/lib64/ as a system library too.

### Android Native API Reference

From android native TVM API to load model & execute can be refered at this [native](https://github.com/dmlc/tvm/tree/master/tutorials/deployment) sample source.


