# TVM RPC Server
This folder contains a simple recipe to make RPC server in c++.

## Usage
- Build tvm
- Make the rpc executable [Makefile](Makefile)
- Use `./tvm_rpc server` to start the RPC server

## How it works
- The tvm dll is linked along with this executable and when the RPC server starts it will load the tvm library.

```
Command line usage
 server       - Start the server
--host        - The hostname of the server, Default=0.0.0.0
--port        - The port of the RPC, Default=9090
--port-end    - The end search port of the RPC, Default=9199
--tracker     - The RPC tracker address in host:port format e.g. 10.1.1.2:9190 Default=""
--key         - The key used to identify the device type in tracker. Default=""
--custom-addr - Custom IP Address to Report to RPC Tracker. Default=""
--silent      - Whether run in silent mode. Default=True
--proxy       - Whether to run in proxy mode. Default=False

  Example
  ./tvm_rpc server --host=0.0.0.0 --port=9000 --port-end=9090 --tracker=127.0.0.1:9190 --key=rasp
```

## Note
Currently support is only there for linux environment.
