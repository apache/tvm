# TVM4J - Java Frontend for TVM Runtime

This folder contains the Java interface for TVM runtime. It brings TVM runtime to Java virtual machine.

- It enables you to construct NDArray from Java native array and vice versa.
- You can register and convert Java native functions to TVM functions.
- It enables you to load shared libraries created by Python and C++.
- It provides a simple interface for RPC server and client.

## Installation

### Requirements

- JDK 1.6+. Oracle JDK and OpenJDK are well tested.
- Maven 3 for build.

### Modules

TVM4J contains three modules:

- core
    * It contains all the Java interfaces.
- native
    * The JNI native library is compiled in this module. It does not link TVM runtime library (libtvm\_runtime.so for Linux and libtvm\_runtime.dylib for OSX). Instead, you have to specify `libtvm.so.path` which contains the TVM runtime library as Java system property.
- assembly
    * It assembles Java interfaces (core), JNI library (native) and TVM runtime library together. The simplest way to integrate tvm4j in your project is to rely on this module. It automatically extracts the native library to a tempfile and load it.

### Build

First please refer to [Installation Guide](http://docs.tvm.ai/install/) and build runtime shared library from the C++ codes (libtvm\_runtime.so for Linux and libtvm\_runtime.dylib for OSX).

Then you can compile tvm4j by

```bash
make jvmpkg
```

(Optional) run unit test by

```bash
make jvmpkg JVM_TEST_ARGS="-DskipTests=false"
```

After it is compiled and packaged, you can install tvm4j in your local maven repository,

```bash
make jvminstall
```

## Convert and Register Java Function as TVM Function

It is easy to define a Java function and call it from TVM. The following snippet demonstrate how to concatenate Java strings.

```java
Function func = Function.convertFunc(new Function.Callback() {
      @Override public Object invoke(TVMValue... args) {
        StringBuilder res = new StringBuilder();
        for (TVMValue arg : args) {
          res.append(arg.asString());
        }
        return res.toString();
      }
    });
TVMValue res = func.pushArg("Hello").pushArg(" ").pushArg("World!").invoke();
assertEquals("Hello World!", res.asString());
res.release();
func.release();
```

It is your job to verify the types of callback arguments, as well as the type of returned result.

You can register the Java function by `Function.register` and use `Function.getFunction` to get the registered function later.

## Use TVM to Generate Shared Library

There's nothing special for this part. The following Python snippet generate add_cpu.so which add two vectors on CPU.

```python
import os
import tvm
from tvm.contrib import cc, util

def test_add(target_dir):
    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
    s = tvm.create_schedule(C.op)
    fadd = tvm.build(s, [A, B, C], "llvm", target_host="llvm", name="myadd")

    fadd.save(os.path.join(target_dir, "add_cpu.o"))
    cc.create_shared(os.path.join(target_dir, "add_cpu.so"),
            [os.path.join(target_dir, "add_cpu.o")])

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit(-1)
    test_add(sys.argv[1])
```

## Run the Generated Shared Library

The following code snippet demonstrate how to load generated shared library (add_cpu.so).

```java
import ml.dmlc.tvm.Module;
import ml.dmlc.tvm.NDArray;
import ml.dmlc.tvm.TVMContext;

import java.io.File;
import java.util.Arrays;

public class LoadAddFunc {
  public static void main(String[] args) {
    String loadingDir = args[0];
    Module fadd = Module.load(loadingDir + File.separator + "add_cpu.so");

    TVMContext ctx = TVMContext.cpu();

    long[] shape = new long[]{2};
    NDArray arr = NDArray.empty(shape, ctx);
    arr.copyFrom(new float[]{3f, 4f});
    NDArray res = NDArray.empty(shape, ctx);

    fadd.entryFunc().pushArg(arr).pushArg(arr).pushArg(res).invoke();
    System.out.println(Arrays.toString(res.asFloatArray()));

    arr.release();
    res.release();
    fadd.release();
  }
}
```

## RPC Server

There are two ways to start an RPC server on JVM. A standalone server can be started by

```java
Server server = new Server(port);
server.start();
```

This will open a socket and wait for remote requests. You can use Java, Python, or any other frontend to make an RPC call. Here's an example for calling remote function `test.rpc.strcat` in Java.

```java
RPCSession client = Client.connect("localhost", port.value);
Function func = client.getFunction("test.rpc.strcat");
String result = func.call("abc", 11L).asString();
```

Another way is to start a proxy, make server and client communicate with each other through the proxy. The following snippet shows how to start a server which connects to a proxy.

```java
Server server = new Server(proxyHost, proxyPort, "key");
server.start();
```

You can also use `StandaloneServerProcessor` and `ConnectProxyServerProcessor` to build your own RPC server. Refer to [Android RPC Server](https://github.com/dmlc/tvm/blob/master/apps/android_rpc/app/src/main/java/ml/dmlc/tvm/tvmrpc/RPCProcessor.java) for more details.