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

AOCL Backend Example
====================

TVM supports Intel FPGA SDK for OpenCL also known as AOCL.  Here is a tutorial for how to use TVM with AOCL.

***Note***: This feature is still experimental.  We cannot use AOCL to deploy an end to end neural networks for now.  In addition, we only tested compilation for emulation mode of AOCL.

We use two python scripts for this tutorial.

- build.py - a script to synthesize FPGA bitstream.
```
import tvm
from tvm import te
tgt_host="llvm"
tgt="aocl_sw_emu"

n = te.var("n")
A = te.placeholder((n,), name='A')
B = te.placeholder((n,), name='B')
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

s = te.create_schedule(C.op)
px, x = s[C].split(C.op.axis[0], nparts=1)

s[C].bind(px, tvm.thread_axis("pipeline"))

fadd = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="myadd")

fadd.save("myadd.o")
fadd.imported_modules[0].save("myadd.aocx")

tvm.contrib.cc.create_shared("myadd.so", ["myadd.o"])
```

- run.py - a script to use FPGA as an accelerator.
```
import tvm
import numpy as np
import os

tgt="aocl_sw_emu"

fadd = tvm.runtime.load("myadd.so")
fadd_dev = tvm.runtime.load("myadd.aocx")
fadd.import_module(fadd_dev)

ctx = tvm.context(tgt, 0)

n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype("float32"), ctx)
b = tvm.nd.array(np.random.uniform(size=n).astype("float32"), ctx)
c = tvm.nd.array(np.zeros(n, dtype="float32"), ctx)

fadd(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())
```

Setup
-----

- Install AOCL 17.1 on Ubuntu 16.04.4 LTS.
- Install BSP for your FPGA device.
- Install FPGA device driver.
- Create an ICD file at /etc/OpenCL/vendors/Altera.icd so that the OpenCL platform can be found.
```
/opt/intelFPGA/17.1/hld/linux64/lib/libalteracl.so
```
- Create an FCD file for example at /opt/Intel/OpenCL/Boards/s5_ref.fcd so that your FPGA device can be found.
```
/opt/intelFPGA/17.1/hld/board/s5_ref/linux64/lib/libaltera_s5_ref_mmd.so
```
- Setup TVM with AOCL and OpenCL enabled.

Emulation
---------

- Run software emulation
```
export CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1

python build.py
python run.py
```

- Run on FPGA devices (not tested)
    - Change tgt value to "aocl -device=s5_ref" on build.py and run.py
```
unset CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA

python build.py
python run.py
```
