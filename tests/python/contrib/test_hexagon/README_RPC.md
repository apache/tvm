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


# A life of a Hexagon API call

The goal is to understand what exactly is happening during `A_data.copyfrom(np.array([2, 3]))`, where `A_data` lives in Hexagon.

## Overview
The diagram below describes the sequence of calls and components involved when memcpy over the Hexagon device is invoked.

![Overview of RPC](https://github.com/tlc-pack/web-data/raw/main/images/design/tvm-hex-rpc.png)

The communication between x86 and Android is done via the standard TVM RPC protocol implemented mostly in `src/runtime/rpc/rpc_endpoint.cc`.

A packet between Android and Hexagon is proxy-ed by the Hexagon FastRPC mechanism. FastRPC depends on the auto-generated implementations of client- and server- side API. During the build time, the Android side API (”stub”) and the Hexagon side API (”skel”) is generated from `src/runtime/hexagon/rpc/hexagon_rpc.idl` (see `cmake/modules/Hexagon.cmake`).

When TVM’s RPC server on Android, `tvm_rpc_android_server`, invokes `hexagon_rpc_send(...)`, it actually calls into the same-name function defined in the stub with the exact same arguments (which includes the URI for the `*skel.so` library to use on Hexagon, which in our case is `libhexagon_rpc_skel.so`). Similarly, on the Hexagon side, `hexagon_rpc_send(...)` call is first intercepted by the “skel” API, which in tern calls the actual implementation defined in `src/runtime/hexagon/rpc/rpc_server.cc`.

## Initialization: Setting up Android and establishing connection between x86 host and android

What’s happening during the launcher initialization at [https://github.com/apache/tvm/blob/7cfaa88e6c18edc0a41e1a984d3cb9d8659a1c2c/tests/python/contrib/test_hexagon/test_launcher.py#L71-L73](https://github.com/apache/tvm/blob/7cfaa88e6c18edc0a41e1a984d3cb9d8659a1c2c/tests/python/contrib/test_hexagon/test_launcher.py#L71-L73) ?

```python
launcher = HexagonLauncher(serial_number=android_serial_number, rpc_info=rpc_info)
launcher.upload(dso_binary_path, dso_binary)
launcher.start_server()
```

Here, we send various files over android via `adb`, and initialize a RPC server via `tvm_rpc_android` binary (built from [https://github.com/apache/tvm/tree/main/apps/cpp_rpc](https://github.com/apache/tvm/tree/main/apps/cpp_rpc)):

[https://github.com/apache/tvm/blob/0c0245ae2230fa07d3e4b8be490fc9c88965730c/python/tvm/contrib/hexagon/build.py#L373-L378](https://github.com/apache/tvm/blob/0c0245ae2230fa07d3e4b8be490fc9c88965730c/python/tvm/contrib/hexagon/build.py#L373-L378)

```python
subprocess.Popen(
    self._adb_device_sub_cmd + ["shell", f"cd {self._workspace} && ./android_bash.sh"],
    stdout=subprocess.PIPE,
    stdin=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
```

[https://github.com/apache/tvm/blob/cd2fa69677516048e165e84a88c774dfb0ee65d1/src/runtime/hexagon/rpc/android_bash.sh.template#L20](https://github.com/apache/tvm/blob/cd2fa69677516048e165e84a88c774dfb0ee65d1/src/runtime/hexagon/rpc/android_bash.sh.template#L20)

```
./tvm_rpc_android server --port=<RPC_SERVER_PORT> --tracker=<RPC_TRACKER_HOST>:<RPC_TRACKER_PORT> --key=<HEXAGON_REMOTE_DEVICE_KEY>&
```

When we do `launcher.create_session()` , a remote RPC session between x86 and android is established via this line:

[https://github.com/apache/tvm/blob/0c0245ae2230fa07d3e4b8be490fc9c88965730c/python/tvm/contrib/hexagon/session.py#L57-L67](https://github.com/apache/tvm/blob/0c0245ae2230fa07d3e4b8be490fc9c88965730c/python/tvm/contrib/hexagon/session.py#L57-L67)

```python
self._rpc = tracker.request(
    ...
    session_constructor_args=[
        "tvm.contrib.hexagon.create_hexagon_session",
        self._session_name,
        self._remote_stack_size_bytes,
    ],
)
```

Which eventually jumps to the following line in C++, which creates a RPC client session on an x86 host and run a server initialization function `tvm.contrib.hexagon.create_hexagon_session` on android:

[https://github.com/apache/tvm/blob/2cca934aad1635e3a83b712958ea83ff65704316/src/runtime/rpc/rpc_socket_impl.cc#L123-L129](https://github.com/apache/tvm/blob/2cca934aad1635e3a83b712958ea83ff65704316/src/runtime/rpc/rpc_socket_impl.cc#L123-L129)

```cpp
TVM_REGISTER_GLOBAL("rpc.Connect").set_body([](TVMArgs args, TVMRetValue* rv) {
  std::string url = args[0];
  int port = args[1];
  std::string key = args[2];
  *rv = RPCClientConnect(url, port, key,
                         TVMArgs(args.values + 3, args.type_codes + 3, args.size() - 3));
});
```

`tvm.contrib.hexagon.create_hexagon_session` is defined here. It establishes a link between android and hexagon, this code runs on android.

[https://github.com/apache/tvm/blob/cd2fa69677516048e165e84a88c774dfb0ee65d1/src/runtime/hexagon/rpc/android/session.cc#L106](https://github.com/apache/tvm/blob/cd2fa69677516048e165e84a88c774dfb0ee65d1/src/runtime/hexagon/rpc/android/session.cc#L106)

```cpp
TVM_REGISTER_GLOBAL("tvm.contrib.hexagon.create_hexagon_session")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      std::string session_name = args[0];
      int remote_stack_size_bytes = args[1];
      HexagonTransportChannel* hexagon_channel =
          new HexagonTransportChannel(hexagon_rpc_URI CDSP_DOMAIN, remote_stack_size_bytes);
      std::unique_ptr<RPCChannel> channel(hexagon_channel);
      auto ep = RPCEndpoint::Create(std::move(channel), session_name, "", NULL);
      auto sess = CreateClientSession(ep);
      *rv = CreateRPCSessionModule(sess);
    });
```

`HexagonTransportChannel` is the one that actually knows how to talk to Hexagon. It uses functions such as `hexagon_rpc_send`, `hexagon_rpc_receive` defined in

[https://github.com/apache/tvm/blob/cd2fa69677516048e165e84a88c774dfb0ee65d1/src/runtime/hexagon/rpc/hexagon/rpc_server.cc](https://github.com/apache/tvm/blob/cd2fa69677516048e165e84a88c774dfb0ee65d1/src/runtime/hexagon/rpc/hexagon/rpc_server.cc)

## x86 host → Android

`A_data.copyfrom(np.array([2, 3]))` reaches this line. This is the boundary between Python and C++ land in TVM FFI:

[https://github.com/apache/tvm/blob/b2757817af7ba3aefe16ea3ccb6d4982dd7fd531/python/tvm/runtime/ndarray.py#L183](https://github.com/apache/tvm/blob/b2757817af7ba3aefe16ea3ccb6d4982dd7fd531/python/tvm/runtime/ndarray.py#L183)

```python
check_call(_LIB.TVMArrayCopyFromBytes(self.handle, data, nbytes))
```

[https://github.com/apache/tvm/blob/37cd9837ff302e4490696ca57a9fbba6404c7046/src/runtime/ndarray.cc#L322](https://github.com/apache/tvm/blob/37cd9837ff302e4490696ca57a9fbba6404c7046/src/runtime/ndarray.cc#L322)

```cpp
int TVMArrayCopyFromBytes(TVMArrayHandle handle, void* data, size_t nbytes) {
  API_BEGIN();
  ArrayCopyFromBytes(handle, data, nbytes);
  API_END();
}
```

Now we come to `ArrayCopyFromBytes` function. The first non-obvious question is, which `DeviceAPI` is selected by `DeviceAPI::Get(handle->device)`?

```cpp
void ArrayCopyFromBytes(DLTensor* handle, const void* data, size_t nbytes) {
  ...
  DLTensor from;
  ...
  DeviceAPI::Get(handle->device)->CopyDataFromTo(&from, handle, nullptr);
  // Synchronize in case data become unavailable later.
  DeviceAPI::Get(handle->device)->StreamSync(handle->device, nullptr);
}
```

The answer: `RPCDeviceAPI` defined below, not `HexagonDeviceAPI`.

[https://github.com/apache/tvm/blob/899bc064e1bf8df915bcadc979a6f37210cdce33/src/runtime/rpc/rpc_device_api.cc#L34](https://github.com/apache/tvm/blob/899bc064e1bf8df915bcadc979a6f37210cdce33/src/runtime/rpc/rpc_device_api.cc#L34)

```cpp
class RPCDeviceAPI final : public DeviceAPI {
   ...
```

This is due to the fact that `sess.device`, used in `test_launcher.py` below, encodes two pieces of information: (1) The device is RPC and (2) it wraps the underlying “real” device Hexagon.

[https://github.com/apache/tvm/blob/2b35cfd6ddb73afecd3f550f33881e1fdc7c3267/tests/python/contrib/test_hexagon/rpc/test_launcher.py#L112](https://github.com/apache/tvm/blob/2b35cfd6ddb73afecd3f550f33881e1fdc7c3267/tests/python/contrib/test_hexagon/rpc/test_launcher.py#L112)

See below for how `sess.device` is created during `HexagonLauncher` initialization.

 `self.device = self._rpc.hexagon(0)`.

[https://github.com/apache/tvm/blob/cd2fa69677516048e165e84a88c774dfb0ee65d1/python/tvm/contrib/hexagon/session.py#L64](https://github.com/apache/tvm/blob/cd2fa69677516048e165e84a88c774dfb0ee65d1/python/tvm/contrib/hexagon/session.py#L64)

`RPCDeviceAPI::CopyDataFromTo` is defined in [https://github.com/apache/tvm/blob/899bc064e1bf8df915bcadc979a6f37210cdce33/src/runtime/rpc/rpc_device_api.cc#L80](https://github.com/apache/tvm/blob/899bc064e1bf8df915bcadc979a6f37210cdce33/src/runtime/rpc/rpc_device_api.cc#L80)

Here, we meet another `GetAPI` call:

```cpp
GetSess(dev_from)->GetDeviceAPI(remote_dev)->CopyDataFromTo(&from_tensor, &to_tensor, stream);
```

[https://github.com/apache/tvm/blob/899bc064e1bf8df915bcadc979a6f37210cdce33/src/runtime/rpc/rpc_device_api.cc#L94](https://github.com/apache/tvm/blob/899bc064e1bf8df915bcadc979a6f37210cdce33/src/runtime/rpc/rpc_device_api.cc#L94)

At first, it is not obvious where this `CopyDataFromTo` jumps to (initially I thought it would jump to `HexagonDeviceAPI`). Since `GetSess(dev_from)` returns the client RPC connection between x86 and android, created during initialization in

[https://github.com/apache/tvm/blob/2cca934aad1635e3a83b712958ea83ff65704316/src/runtime/rpc/rpc_socket_impl.cc#L107](https://github.com/apache/tvm/blob/2cca934aad1635e3a83b712958ea83ff65704316/src/runtime/rpc/rpc_socket_impl.cc#L107)

```cpp
Module RPCClientConnect(std::string url, int port, std::string key, TVMArgs init_seq) {
  auto endpt = RPCConnect(url, port, "client:" + key, init_seq);
  return CreateRPCSessionModule(CreateClientSession(endpt));
}
```

, this jumps to `RPCClientSession` class defined in [https://github.com/apache/tvm/blob/899bc064e1bf8df915bcadc979a6f37210cdce33/src/runtime/rpc/rpc_endpoint.cc#L994](https://github.com/apache/tvm/blob/899bc064e1bf8df915bcadc979a6f37210cdce33/src/runtime/rpc/rpc_endpoint.cc#L994)

```cpp
class RPCClientSession : public RPCSession, public DeviceAPI {
  ...
```

`rpc_endpoint.cc` is a very important file. It contains the core RPC protocol logic. `CopyDataFromTo` in `rpc_device_api.cc` jumps to

[https://github.com/apache/tvm/blob/899bc064e1bf8df915bcadc979a6f37210cdce33/src/runtime/rpc/rpc_endpoint.cc#L1060-L1062](https://github.com/apache/tvm/blob/899bc064e1bf8df915bcadc979a6f37210cdce33/src/runtime/rpc/rpc_endpoint.cc#L1060-L1062)

```cpp
void CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) final {
    endpoint_->SysCallRemote(RPCCode::kCopyAmongRemote, from, to, stream);
}
```

from which things transfer to the Android side.

Here is where `RPCCode::kCopyAmongRemote` is handled:

[https://github.com/apache/tvm/blob/899bc064e1bf8df915bcadc979a6f37210cdce33/src/runtime/rpc/rpc_endpoint.cc#L979-L981](https://github.com/apache/tvm/blob/899bc064e1bf8df915bcadc979a6f37210cdce33/src/runtime/rpc/rpc_endpoint.cc#L979-L981)

```cpp
case RPCCode::kCopyAmongRemote:
  SysCallHandler(RPCCopyAmongRemote);
  break;
```

The handler is represented by `serving_session_`, which is initialized during server initialization at

[https://github.com/apache/tvm/blob/899bc064e1bf8df915bcadc979a6f37210cdce33/src/runtime/rpc/rpc_endpoint.cc#L541](https://github.com/apache/tvm/blob/899bc064e1bf8df915bcadc979a6f37210cdce33/src/runtime/rpc/rpc_endpoint.cc#L541)

```cpp
serving_session_ = RPCModuleGetSession(mod);
```

which corresponds to the Hexagon session created before in [https://github.com/apache/tvm/blob/cd2fa69677516048e165e84a88c774dfb0ee65d1/src/runtime/hexagon/rpc/android/session.cc#L106](https://github.com/apache/tvm/blob/cd2fa69677516048e165e84a88c774dfb0ee65d1/src/runtime/hexagon/rpc/android/session.cc#L106).

The handler is passed to the following function

[https://github.com/apache/tvm/blob/899bc064e1bf8df915bcadc979a6f37210cdce33/src/runtime/rpc/rpc_endpoint.cc#L909-L922](https://github.com/apache/tvm/blob/899bc064e1bf8df915bcadc979a6f37210cdce33/src/runtime/rpc/rpc_endpoint.cc#L909-L922)

```cpp
void RPCCopyAmongRemote(RPCSession* handler, TVMArgs args, TVMRetValue* rv) {
  DLTensor* from = args[0];
  DLTensor* to = args[1];
  ...
  handler->GetDeviceAPI(dev)->CopyDataFromTo(from, to, stream);
}
```

This is an interesting function. Here, `handler` is again `RPCClientSession` due to the line in

[https://github.com/apache/tvm/blob/cd2fa69677516048e165e84a88c774dfb0ee65d1/src/runtime/hexagon/rpc/android/session.cc#L114](https://github.com/apache/tvm/blob/cd2fa69677516048e165e84a88c774dfb0ee65d1/src/runtime/hexagon/rpc/android/session.cc#L114)

```cpp
auto sess = CreateClientSession(ep);
```

so apparently, things might look like it is looping back to `RPCClientSession::CopyDataFromTo`:

```cpp
void CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) final {
    endpoint_->SysCallRemote(RPCCode::kCopyAmongRemote, from, to, stream);
  }
```

But this time, `endpoint_` is different. Previously, this `endpoint_` represented the connection between x86 and android (created in [https://github.com/apache/tvm/blob/2cca934aad1635e3a83b712958ea83ff65704316/src/runtime/rpc/rpc_socket_impl.cc#L99-L100](https://github.com/apache/tvm/blob/2cca934aad1635e3a83b712958ea83ff65704316/src/runtime/rpc/rpc_socket_impl.cc#L99-L100)), but this `endpoint_` belongs to the Hexagon session created in [https://github.com/apache/tvm/blob/cd2fa69677516048e165e84a88c774dfb0ee65d1/src/runtime/hexagon/rpc/android/session.cc#L113](https://github.com/apache/tvm/blob/cd2fa69677516048e165e84a88c774dfb0ee65d1/src/runtime/hexagon/rpc/android/session.cc#L113). So this is where the RPC communication between Android and Hexagon starts.

## Android → Hexagon

Recall that the `endpoint_` owned by the Hexagon session is created via `tvm.contrib.hexagon.create_hexagon_session` when the Android RPC server is being initialized. The `endpoint_` is represented by the following class:

[https://github.com/apache/tvm/blob/c20cbc55c03f9f048b151a1221469b9888123608/src/runtime/hexagon/rpc/android/session.cc#L46](https://github.com/apache/tvm/blob/c20cbc55c03f9f048b151a1221469b9888123608/src/runtime/hexagon/rpc/android/session.cc#L46)

```cpp
class HexagonTransportChannel : public RPCChannel {
 public:
  explicit HexagonTransportChannel(const std::string& uri, int remote_stack_size_bytes) {
    ...
    hexagon_rpc_open(uri.c_str(), &_handle);
    ...
  }

  size_t Send(const void* data, size_t size) override {
    hexagon_rpc_send(_handle, static_cast<const unsigned char*>(data), static_cast<int>(size));
    ...
  }
```

On construction, `hexagon_rpc_open` is called, which will initialize the TVM MinRPC server on Hexagon and overwrites `device_api.hexagon` registry to point to the call to `HexagonDeviceAPI`. [https://github.com/apache/tvm/blob/c20cbc55c03f9f048b151a1221469b9888123608/src/runtime/hexagon/rpc/hexagon/rpc_server.cc#L210-L213](https://github.com/apache/tvm/blob/c20cbc55c03f9f048b151a1221469b9888123608/src/runtime/hexagon/rpc/hexagon/rpc_server.cc#L210-L213)

The endpoint routes each RPC packet by `Send` function, which in turn calls `hexagon_rpc_send(...)` defined in:

[https://github.com/apache/tvm/blob/c20cbc55c03f9f048b151a1221469b9888123608/src/runtime/hexagon/rpc/hexagon/rpc_server.cc#L243](https://github.com/apache/tvm/blob/c20cbc55c03f9f048b151a1221469b9888123608/src/runtime/hexagon/rpc/hexagon/rpc_server.cc#L243)

```cpp
AEEResult hexagon_rpc_send(remote_handle64 _handle, const unsigned char* data,
                           int dataLen) {
  get_hexagon_rpc_server()->Write(reinterpret_cast<const uint8_t*>(data),
                                  static_cast<size_t>(dataLen));
  ...
}
```

This is where FastRPC comes into play and things get very confusing. The endpoint lives in Android, so `hexagon_rpc_send` call (also `hexagon_rpc_open`) happens at Android. But the implementations of these functions in `rpc_server.cc` describe the behavior on the Hexagon side... What’s happening is that FastRPC “stub” and “skel” (see the overview at the top) API intercept those calls and play some magic behind the scene to make RPC call look transparent from the client (Android) perspective.

So when the control comes to the point of definition of `hexagon_rpc_send` in `rpc_server.cc`, FastRPC has already finished its job and so we are really on the Hexagon side now. We come to `HexagonRPCServer::Write(...)` function, which in tern calls into TVM MinRPC server instance `rpc_server_` to process the incoming packet:

[https://github.com/apache/tvm/blob/c20cbc55c03f9f048b151a1221469b9888123608/src/runtime/hexagon/rpc/hexagon/rpc_server.cc#L167](https://github.com/apache/tvm/blob/c20cbc55c03f9f048b151a1221469b9888123608/src/runtime/hexagon/rpc/hexagon/rpc_server.cc#L167)

```cpp
int64_t Write(const uint8_t* data, size_t data_size_bytes) {
  if (io_.SetReadBuffer(data, data_size_bytes) != AEE_SUCCESS) {
    return -1;
  }
  rpc_server_.ProcessOnePacket();
  return (int64_t)data_size_bytes;
}
```

`MinRPCServer::ProcessOnePacket()` function dispatches to `HandleCopyFromRemote()` upon receiving `kCopyFromRemote` request:

[https://github.com/apache/tvm/blob/8c125ca6090a29f38a66d26138b056b7de27cb0b/src/runtime/minrpc/minrpc_server.h#L87](https://github.com/apache/tvm/blob/8c125ca6090a29f38a66d26138b056b7de27cb0b/src/runtime/minrpc/minrpc_server.h#L87)

```cpp
bool ProcessOnePacket() {
  ...

  if (...) {
    ...
  } else {
    switch (code) {
      ...
      case RPCCode::kCopyFromRemote: {
        HandleCopyFromRemote();
        break;
      }
      ...
```

[https://github.com/apache/tvm/blob/8c125ca6090a29f38a66d26138b056b7de27cb0b/src/runtime/minrpc/minrpc_server.h#L178](https://github.com/apache/tvm/blob/8c125ca6090a29f38a66d26138b056b7de27cb0b/src/runtime/minrpc/minrpc_server.h#L178)

```cpp
void HandleCopyFromRemote() {
  DLTensor* arr = this->ArenaAlloc<DLTensor>(1);
  uint64_t data_handle;
  this->Read(&data_handle);
  arr->data = reinterpret_cast<void*>(data_handle);
  ...
  this->ReadArray(arr->shape, arr->ndim);

  if (...) {
    ...
  } else {
    data_ptr = this->ArenaAlloc<uint8_t>(num_bytes);
    DLTensor temp;
    ...
    call_ecode = TVMDeviceCopyDataFromTo(arr, &temp, nullptr);
    // need sync to make sure that the copy is completed.
    if (call_ecode == 0) {
      call_ecode = TVMSynchronize(arr->device.device_type, arr->device.device_id, nullptr);
    }
  }
```

And finally we see a call to `DeviceAPIManager::Get(dev)->CopyDataFromTo` which translates to `HexagonDeviceAPI::CopyDataFromTo` .

[https://github.com/apache/tvm/blob/f929b0fc8e7a600978c9ac0418469bd70d046446/src/runtime/c_runtime_api.cc#L623-L630](https://github.com/apache/tvm/blob/f929b0fc8e7a600978c9ac0418469bd70d046446/src/runtime/c_runtime_api.cc#L623-L630)

```cpp
int TVMDeviceCopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) {
  ...
  DeviceAPIManager::Get(dev)->CopyDataFromTo(from, to, stream);
  ...
}
```
