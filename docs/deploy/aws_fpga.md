HLS Backend Example
===================

TVM supports Xilinx FPGA board with SDAccel.  Here is a tutorial for how to deploy TVM to AWS F1 FPGA instance.

***Note***: This feature is still experimental.  We cannot use SDAccel to deploy an end to end neural networks for now.

We use two python scripts for this tutorial.

- build.py - a script to synthesize FPGA bitstream.
```python
import tvm

tgt_host="llvm"
tgt="sdaccel"

n = tvm.var("n")
A = tvm.placeholder((n,), name='A')
B = tvm.placeholder((n,), name='B')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")

s = tvm.create_schedule(C.op)
px, x = s[C].split(C.op.axis[0], nparts=1)

s[C].bind(px, tvm.thread_axis("pipeline"))

fadd = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="myadd")

fadd.save("myadd.o")
fadd.imported_modules[0].save("myadd.xclbin")

tvm.contrib.cc.create_shared("myadd.so", ["myadd.o"])
```

- run.py - a script to use FPGA as an accelerator.
```python
import tvm
import numpy as np
import os

tgt="sdaccel"

fadd = tvm.module.load("myadd.so")
if os.environ.get("XCL_EMULATION_MODE"):
    fadd_dev = tvm.module.load("myadd.xclbin")
else:
    fadd_dev = tvm.module.load("myadd.awsxclbin")
fadd.import_module(fadd_dev)

ctx = tvm.context(tgt, 0)

n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype("float32"), ctx)
b = tvm.nd.array(np.random.uniform(size=n).astype("float32"), ctx)
c = tvm.nd.array(np.zeros(n, dtype="float32"), ctx)

fadd(a, b, c)
np.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())
```

Setup
-----

- Launch an instance using the FPGA Developer AMI.  We don't need an F1 instance for emulation and synthesis, so it is recommended to use a lower cost instance for them.

- Setup AWS FPGA development kit.
```bash
git clone https://github.com/aws/aws-fpga.git
cd aws-fpga
source sdaccel_setup.sh
source ${XILINX_SDX}/settings64.sh
```

- Setup TVM with OpenCL enabled.

Emulation
---------

- Create emconfig.json for emulation.
```bash
emconfigutil --platform ${AWS_PLATFORM} --nd 1
```

- Copy emconfig.json to the python binary directory.  It is because the current Xilinx toolkit assumes that both host binary and the emconfig.json file are in the same path.
```bash
cp emconfig.json $(dirname $(which python))
```

- Run software emulation
```bash
export XCL_EMULATION_MODE=1
export XCL_TARGET=sw_emu

python build.py
python run.py
```

- Run hardware emulation
```bash
export XCL_EMULATION_MODE=1
export XCL_TARGET=hw_emu

python build.py
python run.py
```


Synthesis
---------

- Run synthesis with the following script.

```bash
unset XCL_EMULATION_MODE
export XCL_TARGET=hw

python build.py
```

- Create AWS FPGA image and upload it to AWS S3.
```
${SDACCEL_DIR}/tools/create_sdaccel_afi.sh -xclbin=myadd.xclbin -o=myadd \
    -s3_bucket=<bucket-name> -s3_dcp_key=<dcp-folder-name> -s3_logs_key=<logs-folder-name>
```
This also generates an awsxclbin file, which is necessary to use the AWS FPGA image on F1 instances.

Run
---

- Launch Amazon EC2 F1 instance.

- Copy `myadd.so`, `myadd.awsxclbin`, and `run.py` to the F1 instance.

- Setup AWS FPGA development kit.
```bash
git clone https://github.com/aws/aws-fpga.git
cd aws-fpga
source sdaccel_setup.sh
```

- Setup TVM with OpenCL enabled.

- Become root and setup environment variables.
```bash
sudo sh
source ${INSTALL_ROOT}/setup.sh
```

- Run
```bash
python run.py
```
