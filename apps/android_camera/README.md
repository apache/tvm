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


# Android Camera Demo Sample App

The Android Camera Demo Sample App provides a basic implementation of an Android app that uses the tvm runtime to perform image classification in real time.

You will need JDK, [Android NDK](https://developer.android.com/ndk) and an Android device to use this.

## Build and Installation

### <a name="preparemodels">Prepare Models</a>

The `models/prepare_models.py` script provides a example flow for dumping model
parameter files for use by the app.

1. Set path to the NDK CC: `export TVM_NDK_CC=[Path to CC, e.g. /opt/android-toolchain-arm64/bin/aarch64-linux-android-g++]`
2. Switch to the script directory: `cd models`
3. Run script: `python3 prepare_model.py`

#### Sample output
```
mobilenet_v2
getting model...
building...
dumping lib...
dumping graph...
dumping params...
dumping labels...
resnet18_v1
getting model...
building...
dumping lib...
dumping graph...
dumping params...
dumping labels...
```

### <a name="buildapk">Build APK</a>

We use [Gradle](https://gradle.org) to build. Please follow [the installation instruction](https://gradle.org/install) for your operating system.

Before you build the Android application, please refer to [TVM4J Installation Guide](https://github.com/apache/tvm/blob/main/jvm/README.md) and install tvm4j-core to your local maven repository. You can find tvm4j dependency declare in `app/build.gradle`. Modify it if it is necessary.

```
dependencies {
    implementation fileTree(dir: 'libs', include: ['*.jar'])
    androidTestImplementation('androidx.test.espresso:espresso-core:3.2.0', {
        exclude group: 'com.android.support', module: 'support-annotations'
    })
    implementation 'androidx.appcompat:appcompat:1.4.0'
    implementation 'androidx.constraintlayout:constraintlayout:2.1.3'
    implementation 'com.google.android.material:material:1.5.0'
    implementation files('../../../jvm/core/target/tvm4j-core-0.0.1-SNAPSHOT.jar')
    testImplementation 'junit:junit:4.13.2'

    implementation "androidx.concurrent:concurrent-futures:1.0.0"
    implementation "androidx.camera:camera-core:1.0.0-beta01"
    implementation "androidx.camera:camera-camera2:1.0.0-beta01"
    // If you want to use the CameraX View class
    implementation "androidx.camera:camera-view:1.0.0-alpha08"
    // If you want to use the CameraX Extensions library
    implementation "androidx.camera:camera-extensions:1.0.0-alpha08"
    // If you want to use the CameraX Lifecycle library
    implementation "androidx.camera:camera-lifecycle:1.0.0-beta01"
}
```

Now use Gradle to compile JNI, resolve Java dependencies and build the Android application together with tvm4j. Run following script to generate the apk file.

```bash
export ANDROID_HOME=[Path to your Android SDK, e.g., ~/Android/sdk]
cd apps/android_camera
gradle clean build
```

In `app/build/outputs/apk` you'll find `app-release-unsigned.apk`, use `dev_tools/gen_keystore.sh` to generate a signature and use `dev_tools/sign_apk.sh` to get the signed apk file `app/build/outputs/apk/release/tv8mdemo-release.apk`.

Upload `tv8mdemo-release.apk` to your Android device and install it:

```bash
$ANDROID_HOME/platform-tools/adb install app/build/outputs/apk/release/tv8mdemo-release.apk
```

If you see error:

    adb: failed to install app/build/outputs/apk/release/tv8mdemo-release.apk:
      Failure [INSTALL_FAILED_UPDATE_INCOMPATIBLE:
      Package ml.apache.tvm.android.androidcamerademo signatures do not match the previously installed version; ignoring!]

Run uninstall first:

```bash
$ANDROID_HOME/platform-tools/adb uninstall ml.apache.tvm.android.androidcamerademo
```
### Troubleshooting

If you build the application in Android Studio and see error similar to this one:
```
A problem occurred evaluating project ':app'.
> Failed to apply plugin 'com.android.internal.version-check'.
   > Minimum supported Gradle version is 7.5. Current version is 7.4. If using the gradle wrapper, try editing the distributionUrl in /Users/echuraev/Workspace/OctoML/tvm_android_test/apps/android_deploy/gradle/wrapper/gradle-wrapper.properties to gradle-7.5-all.zip
```
Run project syncing `File -> Sync Project with Gradle Files`. It should sync the
project and create gradle-wrapper files.
