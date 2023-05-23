<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Android TVM Demo

This folder contains Android Demo app that allows us to show how to deploy model using TVM runtime api on a Android phone.

You will need [JDK](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html), [Android SDK](https://developer.android.com/studio/index.html), [Android NDK](https://developer.android.com/ndk) and an Android device to use this. Make sure the `ANDROID_HOME` variable already points to your Android SDK folder or set it using `export ANDROID_HOME=[Path to your Android SDK, e.g., ~/Android/sdk]`. We use [Gradle](https://gradle.org) to build. Please follow [the installation instruction](https://gradle.org/install) for your operating system.

Alternatively, you may execute Docker image we provide which contains the required packages. Use the command below to build the image and enter interactive session.

```bash
./docker/build.sh demo_android -it bash
(docker) $ echo $ANDROID_HOME
(docker) /opt/android-sdk-linux
```


## Build and Installation

### Build APK

Before you build the Android application, please refer to [TVM4J Installation Guide](https://github.com/apache/tvm/blob/main/jvm/README.md) and install tvm4j-core to your local maven repository. You can find tvm4j dependency declare in `app/build.gradle`. Modify it if it is necessary.

```
dependencies {
    implementation fileTree(dir: 'libs', include: ['*.jar'])
    androidTestImplementation('androidx.test.espresso:espresso-core:3.1.0', {
        exclude group: 'com.android.support', module: 'support-annotations'
    })
    implementation 'androidx.appcompat:appcompat:1.6.1'
    implementation 'androidx.constraintlayout:constraintlayout:2.1.4'
    implementation 'com.google.android.material:material:1.8.0'
    implementation files('../../../jvm/core/target/tvm4j-core-0.0.1-SNAPSHOT.jar')
    testImplementation 'junit:junit:4.13.2'
}
```

Application default has CPU and GPU (OpenCL) versions TVM runtime flavor and follow below instruction to setup.
In `app/src/main/jni/make` you will find JNI Makefile config `config.mk` and copy it to `app/src/main/jni` and modify it.

```bash
cd apps/android_deploy/app/src/main/jni
cp make/config.mk .
```

Here's a piece of example for `config.mk`.

```makefile
APP_ABI = arm64-v8a

APP_PLATFORM = android-17
```

Now use Gradle to compile JNI, resolve Java dependencies and build the Android application together with tvm4j. Run following script to generate the apk file.

```bash
cd apps/android_deploy
gradle clean build
```

In `app/build/outputs/apk` you'll find `app-release-unsigned.apk`, use `dev_tools/gen_keystore.sh` to generate a signature and use `dev_tools/sign_apk.sh` to get the signed apk file `app/build/outputs/apk/tvmdemo-release.apk`.

Upload `tvmdemo-release.apk` to your Android device and install it.

### Build with OpenCL

Application is building with OpenCL support by default.
[OpenCL-wrapper](../../src/runtime/opencl/opencl_wrapper) is used and will dynamically load OpenCL library on the device.
If the device doesn't have OpenCL library on it, then you'll see in the runtime that OpenCL library cannot be opened.
If you want to build this application without OpenCL then set `USE_OPENCL = 0`
in [config.mk](./app/src/main/jni/make/config.mk)

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

Follow instruction to get compiled version model for android target [here.](https://tvm.apache.org/docs/deploy/android.html)

Copied these compiled model deploy_lib.so, deploy_graph.json and deploy_param.params to apps/android_deploy/app/src/main/assets/ and modify TVM flavor changes on [java](https://github.com/apache/tvm/blob/main/apps/android_deploy/app/src/main/java/org/apache/tvm/android/demo/MainActivity.java#L81)

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

### Troubleshooting

If you build the application in Android Studio and see error similar to this one:
```
A problem occurred evaluating project ':app'.
> Failed to apply plugin 'com.android.internal.version-check'.
   > Minimum supported Gradle version is 7.5. Current version is 7.4. If using the gradle wrapper, try editing the distributionUrl in /Users/echuraev/Workspace/OctoML/tvm_android_test/apps/android_deploy/gradle/wrapper/gradle-wrapper.properties to gradle-7.5-all.zip
```
Run project syncing `File -> Sync Project with Gradle Files`. It should sync the
project and create gradle-wrapper files.
