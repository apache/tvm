# iOS TVM RPC

This folder contains iOS RPC app that allows us to launch an rpc server on a iOS device(e.g. ipython)
and connect to it through python script and do testing on the python side as normal TVM RPC.
You will need XCode and an iOS device to use this.


## Building
Before start, please run ```init_proj.py``` to update XCode developer metadata. After this step, open
```tvmrpc.xcodeproj``` by using XCode, build the App and install the App on the phone. Usually, we
**do not** use the iOS App directly. 

## Workflow
Due to security restriction of iOS10. We cannot upload dynamic libraries to the App and load it from sandbox.
Instead, we need to build a list of libraries, pack them into the app bundle, launch the RPC server and
connect to test the bundled libraries. We use ```xcodebuild test``` to automate this process.

The test script [tests/ios_rpc_test.py](tests/ios_rpc_test.py) is a good template for the workflow. With this 
script, we don't need to manually operate the iOS App, this script will build the app, run it and collect the results automatically. 

 To run the script,  you need to configure the following environment variables

- ```TVM_IOS_CODESIGN``` The signature you use to codesign the app and libraries (e.g. ```iPhone Developer: Name (XXXX)```)
- ```TVM_IOS_RPC_ROOT``` The root directory of the iOS rpc project
- ```TVM_IOS_RPC_PROXY_HOST``` The RPC proxy address
- ```TVM_IOS_RPC_DESTINATION``` The Xcode target device(e.g. ```platform=iOS,name=xxxx```)

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
It will also create an ```tvmrpc.app/Frameworks/tvm/rpc_config.txt``` contaiing the first line.

When we run the testcase, the testcase read the configuration from ```tvmrpc.app/Frameworks/tvm/rpc_config.txt```
and connect to the specified RPC proxy, start serving loop.

So if we want to start the RPC from XCode IDE, simply manually modify ```rpc_config.txt``` file and click test.
Then connect to the proxy via the python script.

We can also use the RPC App directly, by typing in the address and press connect to connect to the proxy.
However, the restriction is we can only load the modules that are bundled to the App.
