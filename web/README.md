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

# TVM WebAssembly and Javascript Backend

This folder contains TVM WebAssembly and Javascript backend through Emscripten.

## Installation
While the LLVM main branch support webassembly as a target. We still need a good runtime with libc and other
system library support. Emscripten toolchain offers that nicely. The general idea is to build TVM against
the fastcomp LLVM backend in the Emscripten project and allow us to generate ```asmjs-unknown-emscripten```
as a backend target.

### Setup Emscripten
Checkout [Emscripten Portable SDK Downloads](https://kripken.github.io/emscripten-site/docs/getting_started/downloads.html)
to download emsdk-portable and unzip it on a local folder. Follow the installation guide from emscripten document.

```bash
./emsdk update
./emsdk install latest
./emsdk activate latest
```

Because we need to compile against the LLVM backend of emscripten, we will need the source and llvm library.
Which can be installed via following command.

```bash
./emsdk install clang-incoming-64bit
./emsdk activate clang-incoming-64bit
```

### Setup Environment Variable

In normal setting, we can setup the necessary environment variable with the following command.
```bash
source /path-to-emsdk-portable/emsdk_env.sh
```
However, this will put emscripten's clang and llvm path ahead of the current system path.
What you can do is to set the path manually, by putting emscripten's path after the PATH like the following ones.
You can get the detailed path by type ```./emsdk activate```

```bash
export PATH=${PATH}:/emsdk-related-path-here

```

### Build TVM with Fastcomp LLVM

To build TVM with Emscripten's Fastcomp LLVM, we can modify the LLVM_CONFIG in ```config.mk```
to point to fastcomp's llvm-config and build TVM normally.

```bash
LLVM_CONFIG = /path/to/emsdk-portable/clang/fastcomp/build_incoming_64/bin/llvm-config
```

### Build TVM Web Runtime

The above command gives us the TVM compiling environment. Now we need to build runtime,
to do so, make sure we set the environment correctly as in previous section and type

```bash
make web
```

This will create ```build/libtvm_web_runtime.bc``` and ```build/libtvm_web_runtime.js```.

## Use TVM to Generate Javascript Library

The general idea is to use TVM as normally and set target to be ```llvm -target=asmjs-unknown-emscripten -system-lib```.

The following code snippet from [tests/web/prepare_test_libs.py](https://github.com/apache/incubator-tvm/tree/master/tests/web/prepare_test_libs.py) demonstrate
the compilation process.

```python
import tvm
from tvm import te
from tvm.contrib import emscripten
import os
def prepare_test_libs(base_path):
    target = "llvm -target=asmjs-unknown-emscripten -system-lib"
    if not tvm.runtime.enabled(target):
        raise RuntimeError("Target %s is not enbaled" % target)
    n = te.var("n")
    A = te.placeholder((n,), name='A')
    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
    s = te.create_schedule(B.op)
    fadd1 = tvm.build(s, [A, B], target, name="add_one")
    obj_path = os.path.join(base_path, "test_add_one.bc")
    fadd1.save(obj_path)
    emscripten.create_js(os.path.join(base_path, "test_module.js"), obj_path)

if __name__ == "__main__":
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    prepare_test_libs(os.path.join(curr_path, "../../build"))
```

In this workflow, we use TVM to generate a ```.bc``` file and statically link
that with the  ```build/libtvm_web_runtime.bc```(emscripten.create_js will help you do that).
The result js library is a library that contains both TVM runtime and the compiled function.


## Run the Generated Library

The following code snippet from [tests/web/test_module_load.js](https://github.com/apache/incubator-tvm/tree/master/tests/web/test_module_load.js) demonstrate
how to run the compiled library.

```js
// Load Emscripten Module, need to change path to root/build
const path = require("path");
process.chdir(path.join(__dirname, "../../build"));
var Module = require("../../build/test_module.js");
// Bootstrap TVMruntime with emscripten module.
const tvm_runtime = require("../../web/tvm_runtime.js");
const tvm = tvm_runtime.create(Module);

// Load system library, the compiled function is registered in sysLib.
var sysLib = tvm.systemLib();

function randomArray(length, max) {
  return Array.apply(null, Array(length)).map(function() {
    return Math.random() * max;
  });
}

function testAddOne() {
  // grab pre-loaded function
  var faddOne = sysLib.getFunction("add_one");
  var assert = require('assert');
  tvm.assert(tvm.isPackedFunc(faddOne));
  var n = 124;
  var A = tvm.empty(n).copyFrom(randomArray(n, 1));
  var B = tvm.empty(n);
  // call the function.
  faddOne(A, B);
  AA = A.asArray();  // retrieve values in js array
  BB = B.asArray();  // retrieve values in js array
  // verify
  for (var i = 0; i < BB.length; ++i) {
    assert(Math.abs(BB[i] - (AA[i] + 1)) < 1e-5);
  }
  faddOne.release();
}

testAddOne();
sysLib.release();
console.log("Finish verifying test_module_load");
```

Current example supports static linking, which is the preferred way to get more efficiency
in javascript backend.

## Proxy based RPC

We can now use javascript end to start an RPC server and connect to it from python side,
making the testing flow easier.

The following is an example to reproduce this. This requires everything to be in the git source and setup PYTHONPATH(instead of use setup.py install)
- run "python -m tvm.exec.rpc_proxy --example-rpc=1" to start proxy.
- Open broswer, goto the server webpage click Connect to proxy.
  - Alternatively run "node web/example_rpc_node.js"
- run "python tests/web/websock_rpc_test.py" to run the rpc client.

The general idea is to use Emscripten's dynamic linking to dynamically load modules.
