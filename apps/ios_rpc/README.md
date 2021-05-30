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

# iOS TVM RPC

This folder contains iOS RPC app that allows us to launch an rpc server on a iOS device(e.g. ipython)
and connect to it through python script and do testing on the python side as normal TVM RPC.
You will need XCode and an iOS device to use this.

## RPC proxy
Start the RPC proxy by running in a terminal:

    python -m tvm.exec.rpc_proxy

On success, you should see something like this:

    INFO:root:RPCProxy: client port bind to 0.0.0.0:9090
    INFO:root:RPCProxy: Websock port bind to 8888

IP-address of this machine will be used to initialize ```TVM_IOS_RPC_PROXY_HOST```
environment variable (see below).

## Building
Before start, please run ```init_proj.py``` to update XCode developer metadata. After this step, open
```tvmrpc.xcodeproj``` by using XCode, build the App and install the App on the phone. Usually, we
**do not** use the iOS App directly.

To test an App, you can fill ``Address`` field with IP-address of RPC proxy
(see above), and press ``Connect to Proxy``.

On success, "Disconnected" will change to "Connected".
On RPC proxy side you can see the next message in a log:

    INFO:root:Handler ready TCPSocketProxy:<iPhone IP address>:server:iphone

Now App can be closed by pressing the home button (or even removed from a device).

## Workflow
Due to security restriction of iOS10. We cannot upload dynamic libraries to the App and load it from sandbox.
Instead, we need to build a list of libraries, pack them into the app bundle, launch the RPC server and
connect to test the bundled libraries. We use ```xcodebuild test``` to automate this process. There is also
one more approach to workaround this limitation, for more details please take a look into section
[Custom DSO loader integration](#custom-dso-loader-plugin).

The test script [tests/ios_rpc_test.py](tests/ios_rpc_test.py) is a good template for the workflow. With this
script, we don't need to manually operate the iOS App, this script will build the app, run it and collect the results 
automatically.

 To run the script,  you need to configure the following environment variables

- ```TVM_IOS_CODESIGN``` The signature you use to codesign the app and libraries (e.g. ```iPhone Developer: Name (XXXX)```)
- ```TVM_IOS_TEAM_ID``` The developer Team ID available at https://developer.apple.com/account/#/membership     
- ```TVM_IOS_RPC_ROOT``` The root directory of the iOS rpc project
- ```TVM_IOS_RPC_PROXY_HOST``` The RPC proxy address (see above)
- ```TVM_IOS_RPC_DESTINATION``` The Xcode target device (e.g. ```platform=iOS,id=xxxx```)

See instructions of how to find UUID of the iOS device:

- https://www.innerfence.com/howto/find-iphone-unique-device-identifier-udid

## How it works
Let us explain how it works, the project look for ```rpc_config.txt``` file in the project root folder.
The ```rpc_config.txt``` file should be in the following format:
```
<url> <port> <key>
[path to dylib1]
[path to dylib2]
...
```
The build script will copy all the dynamic libraries into bundle ```tvmrpc.app/Frameworks/tvm```,
which you will be able to load via RPC using ```remote.load_module```.
It will also create an ```tvmrpc.app/Frameworks/tvm/rpc_config.txt``` containing the first line.

When we run the testcase, the testcase read the configuration from ```tvmrpc.app/Frameworks/tvm/rpc_config.txt```
and connect to the specified RPC proxy, start serving loop.

So if we want to start the RPC from XCode IDE, simply manually modify ```rpc_config.txt``` file and click test.
Then connect to the proxy via the python script.

We can also use the RPC App directly, by typing in the address and press connect to connect to the proxy.
However, the restriction is we can only load the modules that are bundled to the App.

## Custom DSO loader plugin
While iOS platform itself doesn't allow us to run an unsigned binary, where is a partial ability to run JIT code
on real iOS devices. While application is running under debug session, system allows allocating memory with write
and execute permissions (requirements of debugger). So we can use this feature to load binary on RPC side. For this
purpose we use custom version of `dlopen` function which doesn't check signature and permissions for module loading.
This custom `dlopen` mechanic is integrated into TVM RPC as plugin and registered to execution only inside iOS RPC
application.

The custom implementation of `dlopen` and other functions from `dlfcn.h` header are placed in separate repository,
and will be downloaded automatically during cmake build for iOS. To run cmake build you may use next flags:
```shell
export DEVELOPER_DIR=/Applications/Xcode.app  # iOS SDK is part of Xcode bundle. Have to set it as default Dev Env
cmake ..
  -DCMAKE_BUILD_TYPE=Debug
  -DCMAKE_SYSTEM_NAME=iOS
  -DCMAKE_SYSTEM_VERSION=14.0
  -DCMAKE_OSX_SYSROOT=iphoneos
  -DCMAKE_OSX_ARCHITECTURES=arm64
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
  -DCMAKE_BUILD_WITH_INSTALL_NAME_DIR=ON
  -DCMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM=XXXXXXXXXX  # insert your Team ID
  -DUSE_IOS_RPC=ON  # to enable build iOS RPC application from TVM project tree
cmake --build . --target custom_dso_loader ios_rpc  # Will use custom DSO loader by default
# Resulting iOS RPC app bundle will be placed in:
# apps/ios_rpc/ios_rpc/src/ios_rpc-build/[CONFIG]-iphoneos/tvmrpc.app
```

To enable using of Custom DSO Plugin during xcode build outsde of Cmake you should specify two additional variables.
You can do it manually inside Xcode IDE or via command line args for `xcodebuild`. Make sure that `custom_dso_loader`
target from previous step is already built.
* TVM_BUILD_DIR=path-to-tvm-ios-build-dir
* USE_CUSTOM_DSO_LOADER=1

iOS RPC application with enabled custom DSO loader is able to process modules passed via regular
`remote.upload("my_module.dylib")` mechanics. For example take a look inside `test_rpc_module_with_upload` test case
of file [ios_rpc_test.py](tests/ios_rpc_test.py).
