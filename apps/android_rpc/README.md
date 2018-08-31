# Android TVM RPC

This folder contains Android RPC app that allows us to launch an RPC server on a Android device and connect to it through python script and do testing on the python side as normal TVM RPC.

You will need JDK, [Android NDK](https://developer.android.com/ndk) and an Android device to use this.

## Build and Installation

### <a name="buildapk">Build APK</a>

We use [Gradle](https://gradle.org) to build. Please follow [the installation instruction](https://gradle.org/install) for your operating system.

Before you build the Android application, please refer to [TVM4J Installation Guide](https://github.com/dmlc/tvm/blob/master/jvm/README.md) and install tvm4j-core to your local maven repository. You can find tvm4j dependency declare in `app/build.gradle`. Modify it if it is necessary.

```
dependencies {
    compile fileTree(dir: 'libs', include: ['*.jar'])
    androidTestCompile('com.android.support.test.espresso:espresso-core:2.2.2', {
        exclude group: 'com.android.support', module: 'support-annotations'
    })
    compile 'com.android.support:appcompat-v7:26.0.1'
    compile 'com.android.support.constraint:constraint-layout:1.0.2'
    compile 'com.android.support:design:26.0.1'
    compile 'ml.dmlc.tvm:tvm4j-core:0.0.1-SNAPSHOT'
    testCompile 'junit:junit:4.12'
}
```

Now use Gradle to compile JNI, resolve Java dependencies and build the Android application together with tvm4j. Run following script to generate the apk file.

```bash
export ANDROID_HOME=[Path to your Android SDK, e.g., ~/Android/sdk]
cd apps/android_rpc
gradle clean build
```

In `app/build/outputs/apk` you'll find `app-release-unsigned.apk`, use `dev_tools/gen_keystore.sh` to generate a signature and use `dev_tools/sign_apk.sh` to get the signed apk file `app/build/outputs/apk/tvmrpc-release.apk`.

Upload `tvmrpc-release.apk` to your Android device and install it.

### Build with OpenCL

This application does not link any OpenCL library unless you configure it to. In `app/src/main/jni/make` you will find JNI Makefile config `config.mk`. Copy it to `app/src/main/jni` and modify it.

```bash
cd apps/android_rpc/app/src/main/jni
cp make/config.mk .
```

Here's a piece of example for `config.mk`.

```makefile
APP_ABI = arm64-v8a

APP_PLATFORM = android-17

# whether enable OpenCL during compile
USE_OPENCL = 1

# the additional include headers you want to add, e.g., SDK_PATH/adrenosdk/Development/Inc
ADD_C_INCLUDES = /opt/adrenosdk-osx/Development/Inc

# the additional link libs you want to add, e.g., ANDROID_LIB_PATH/libOpenCL.so
ADD_LDLIBS = libOpenCL.so
```

Note that you should specify the correct GPU development headers for your android device. Run `adb shell dumpsys | grep GLES` to find out what GPU your android device uses. It is very likely the library (libOpenCL.so) is already present on the mobile device. For instance, I found it under `/system/vendor/lib64`. You can do `adb pull /system/vendor/lib64/libOpenCL.so ./` to get the file to your desktop.

After you setup the `config.mk`, follow the instructions in [Build APK](#buildapk) to build the Android package.

## Cross Compile and Run on Android Devices

### Architecture and Android Standalone Toolchain

In order to cross compile a shared library (.so) for your android device, you have to know the target triple for the device. (Refer to [Cross-compilation using Clang](https://clang.llvm.org/docs/CrossCompilation.html) for more information). Run `adb shell cat /proc/cpuinfo` to list the device's CPU information.

Now use NDK to generate standalone toolchain for your device. For my test device, I use following command.

```bash
cd /opt/android-ndk/build/tools/
./make-standalone-toolchain.sh --platform=android-24 --use-llvm --arch=arm64 --install-dir=/opt/android-toolchain-arm64
```

If everything goes well, you will find compile tools in `/opt/android-toolchain-arm64/bin`. For example, `bin/aarch64-linux-android-g++` can be used to compile C++ source codes and create shared libraries for arm64 Android devices.

### Cross Compile and Upload to the Android Device

First start an RPC tracker using 

```python -m tvm.exec.rpc_tracker --port [PORT]``` 

and connect your Android device to this RPC tracker via the TVM RPC application. Open the app,
set the `Address` and `Port` fields to the address and port of the RPC tracker respectively.
The key should be set to "android" if you wish to avoid modifying the default test script.

After pushing "START RPC" button on the app, you can check the connect by run 

```python -m tvm.exec.query_rpc_tracker --port [PORT]``` 

on your host machine. 
You are supposed to find a free "android" in the queue status.

```
...

Queue Status
-------------------------------
key       total  free  pending
-------------------------------
android   1      1     0
-------------------------------
```


Then checkout [android\_rpc/tests/android\_rpc\_test.py](https://github.com/dmlc/tvm/blob/master/apps/android_rpc/tests/android_rpc_test.py) and run,

```bash
# Specify the RPC tracker
export TVM_TRACKER_HOST=0.0.0.0
export TVM_TRACKER_PORT=[PORT]
# Specify the standalone Android C++ compiler
export TVM_NDK_CC=/opt/android-toolchain-arm64/bin/aarch64-linux-android-g++
python android_rpc_test.py
```

This will compile TVM IR to shared libraries (CPU, OpenCL and Vulkan) and run vector addition on your Android device. To verify compiled TVM IR shared libraries on OpenCL target set [`'test_opencl = True'`](https://github.com/dmlc/tvm/blob/master/apps/android_rpc/tests/android_rpc_test.py#L25) and on Vulkan target set [`'test_vulkan = False'`](https://github.com/dmlc/tvm/blob/master/apps/android_rpc/tests/android_rpc_test.py#L27) in  [tests/android_rpc_test.py](https://github.com/dmlc/tvm/blob/master/apps/android_rpc/tests/android_rpc_test.py), by default on CPU target will execute.
On my test device, it gives following results.

```bash
Run CPU test ...
0.000962932 secs/op

Run GPU(OpenCL Flavor) test ...
0.000155807 secs/op

[23:29:34] /home/tvm/src/runtime/vulkan/vulkan_device_api.cc:674: Cannot initialize vulkan: [23:29:34] /home/tvm/src/runtime/vulkan/vulkan_device_api.cc:512: Check failed: __e == VK_SUCCESS Vulan Error, code=-9: VK_ERROR_INCOMPATIBLE_DRIVER

Stack trace returned 10 entries:
[bt] (0) /home/user/.local/lib/python3.6/site-packages/tvm-0.4.0-py3.6-linux-x86_64.egg/tvm/libtvm.so(dmlc::StackTrace[abi:cxx11]()+0x53) [0x7f477f5399f3]
.........

You can still compile vulkan module but cannot run locally
Run GPU(Vulkan Flavor) test ...
0.000225198 secs/op
```

You can define your own TVM operators and test via this RPC app on your Android device to find the most optimized TVM schedule.
