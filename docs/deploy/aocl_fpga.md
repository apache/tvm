AOCL Backend Example
====================

TVM supports Intel FPGA SDK for OpenCL also known as AOCL.  Here is a tutorial for how to use TVM with AOCL.

***Note***: This feature is still experimental.  We cannot use AOCL to deploy an end to end neural networks for now.  In addition, we can only use AOCL's emulation mode for now.

We use a python scripts for this tutorial.

- emu-aocl-fpga.py
```# -*- coding: utf-8 -*-
import tvm
import numpy as np

tgt_host = 'llvm'
tgt = 'opencl'

# Define a computation.
n = tvm.var('n')
a = tvm.placeholder((n,), name='a')
b = tvm.placeholder((n,), name='b')
c = tvm.compute(a.shape, lambda i: a[i] + b[i], name='c')

# Make a schedule.
s = tvm.create_schedule(c.op)
px, x = s[c].split(c.op.axis[0], nparts=1)
s[c].bind(px, tvm.thread_axis("pipeline"))

# Make a executable code.
fadd = tvm.build(s, [a, b, c], tgt, target_host=tgt_host, name='myadd')

# Run.
ctx = tvm.context(tgt, 0)
n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(a.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=n).astype(b.dtype), ctx)
c = tvm.nd.array(np.zeros(n, dtype=c.dtype), ctx)
fadd(a, b, c)```

Setup
-----

- Install AOCL 17.1 on Ubuntu 16.04.4 LTS.
- Install FPGA device driver.
- Make ICD file.
- Make FCD file.
- Setup TVM with AOCL and OpenCL enabled.

Emulation
---------

- Set environment variable.
```export CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1```

- Run software emulation
```python emu-aocl-fpga.py```
