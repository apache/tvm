# Android TVM Demo

This folder contains Android Demo app that allows us to show how to deploy model using TVM runtime api on a Android phone.

You will need [JDK](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html), [Android SDK](https://developer.android.com/studio/index.html), [Android NDK](https://developer.android.com/ndk) and an Android device to use this.

## Build and Installation

### Build APK

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

Application default has CPU version TVM runtime flavor and follow below instruction to setup.
In `app/src/main/jni/make` you will find JNI Makefile config `config.mk` and copy it to `app/src/main/jni` and modify it.

```bash
cd apps/android_deploy/app/src/main/jni
cp make/config.mk .
```

Here's a piece of example for `config.mk`.

```makefile
APP_ABI = arm64-v8a

APP_PLATFORM = android-17

# whether enable OpenCL during compile
USE_OPENCL = 0
```

Now use Gradle to compile JNI, resolve Java dependencies and build the Android application together with tvm4j. Run following script to generate the apk file.

```bash
export ANDROID_HOME=[Path to your Android SDK, e.g., ~/Android/sdk]
cd apps/android_deploy
gradle clean build
```

In `app/build/outputs/apk` you'll find `app-release-unsigned.apk`, use `dev_tools/gen_keystore.sh` to generate a signature and use `dev_tools/sign_apk.sh` to get the signed apk file `app/build/outputs/apk/tvmdemo-release.apk`.

Upload `tvmdemo-release.apk` to your Android device and install it.

### Build with OpenCL

Application does not link with OpenCL library unless you configure it to. Modify JNI Makefile config `app/src/main/jni` with proper target OpenCL configuration.

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

After you setup the `config.mk`, follow the instructions in [Build APK](#buildapk) to build the Android package with OpenCL flavor.

## Cross Compile and Run on Android Devices

### Architecture and Android Standalone Toolchain

In order to cross compile a shared library (.so) for your android device, you have to know the target triple for the device. (Refer to [Cross-compilation using Clang](https://clang.llvm.org/docs/CrossCompilation.html) for more information). Run `adb shell cat /proc/cpuinfo` to list the device's CPU information.

Now use NDK to generate standalone toolchain for your device. For my test device, I use following command.

```bash
cd /opt/android-ndk/build/tools/
./make-standalone-toolchain.sh --platform=android-24 --use-llvm --arch=arm64 --install-dir=/opt/android-toolchain-arm64
```

If everything goes well, you will find compile tools in `/opt/android-toolchain-arm64/bin`. For example, `bin/aarch64-linux-android-g++` can be used to compile C++ source codes and create shared libraries for arm64 Android devices.

### Place compiled model on Android application assets folder

Follow instruction to get compiled version model for android target [here.](https://github.com/dmlc/tvm/blob/master/docs/how_to/deploy_android.md#build-model-for-android-target)

Copied these compiled model deploy_lib.so, deploy_graph.json and deploy_param.params to apps/android_deploy/app/src/main/assets/ and modify TVM flavor changes on [java](https://github.com/dmlc/tvm/blob/master/apps/android_deploy/app/src/main/java/ml/dmlc/tvm/android/demo/MainActivity.java#L81)

`CPU Verison flavor`
```
    private static final boolean EXE_GPU            = false;
```

`OpenCL Verison flavor`
```
    private static final boolean EXE_GPU            = true;
```


Install compiled android application on phone and enjoy the image classifier demo using extraction model

You can define your own TVM operators and deploy via this demo application on your Android device to find the most optimized TVM schedule.
