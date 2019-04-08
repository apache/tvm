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
connect to test the bundled libraries. We use ```xcodebuild test``` to automate this process.

The test script [tests/ios_rpc_test.py](tests/ios_rpc_test.py) is a good template for the workflow. With this
script, we don't need to manually operate the iOS App, this script will build the app, run it and collect the results automatically.

 To run the script,  you need to configure the following environment variables

- ```TVM_IOS_CODESIGN``` The signature you use to codesign the app and libraries (e.g. ```iPhone Developer: Name (XXXX)```)
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
