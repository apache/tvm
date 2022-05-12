..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.


HLS Backend Example
===================

TVM supports Xilinx FPGA board with SDAccel.  Here is a tutorial for how to deploy TVM to AWS F1 FPGA instance.

.. note::

    This feature is still experimental.  We cannot use SDAccel to deploy an end to end neural networks for now.

We use two python scripts for this tutorial.

- build.py - a script to synthesize FPGA bitstream.

  .. code:: python

      import tvm
      from tvm import te

      tgt= tvm.target.Target("sdaccel", host="llvm")

      n = te.var("n")
      A = te.placeholder((n,), name='A')
      B = te.placeholder((n,), name='B')
      C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

      s = te.create_schedule(C.op)
      px, x = s[C].split(C.op.axis[0], nparts=1)

      s[C].bind(px, tvm.te.thread_axis("pipeline"))

      fadd = tvm.build(s, [A, B, C], tgt, name="myadd")

      fadd.save("myadd.o")
      fadd.imported_modules[0].save("myadd.xclbin")

      tvm.contrib.cc.create_shared("myadd.so", ["myadd.o"])

- run.py - a script to use FPGA as an accelerator.

  .. code:: python

      import tvm
      import numpy as np
      import os

      tgt="sdaccel"

      fadd = tvm.runtime.load_module("myadd.so")
      if os.environ.get("XCL_EMULATION_MODE"):
          fadd_dev = tvm.runtime.load_module("myadd.xclbin")
      else:
          fadd_dev = tvm.runtime.load_module("myadd.awsxclbin")
      fadd.import_module(fadd_dev)

      dev = tvm.device(tgt, 0)

      n = 1024
      a = tvm.nd.array(np.random.uniform(size=n).astype("float32"), dev)
      b = tvm.nd.array(np.random.uniform(size=n).astype("float32"), dev)
      c = tvm.nd.array(np.zeros(n, dtype="float32"), dev)

      fadd(a, b, c)
      tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())


Setup
-----

- Launch an instance using the FPGA Developer AMI.  We don't need an F1 instance for emulation and synthesis, so it is recommended to use a lower cost instance for them.
- Setup AWS FPGA development kit.

  .. code:: bash

      git clone https://github.com/aws/aws-fpga.git
      cd aws-fpga
      source sdaccel_setup.sh
      source ${XILINX_SDX}/settings64.sh

- Setup TVM with OpenCL enabled.

Emulation
---------

- Create emconfig.json for emulation.

  .. code:: bash

      emconfigutil --platform ${AWS_PLATFORM} --nd 1

- Copy emconfig.json to the python binary directory.  It is because the current Xilinx toolkit assumes that both host binary and the emconfig.json file are in the same path.

  .. code:: bash

      cp emconfig.json $(dirname $(which python))

- Run software emulation

  .. code:: bash

      export XCL_EMULATION_MODE=1
      export XCL_TARGET=sw_emu

      python build.py
      python run.py

- Run hardware emulation

  .. code:: bash

      export XCL_EMULATION_MODE=1
      export XCL_TARGET=hw_emu

      python build.py
      python run.py

Synthesis
---------

- Run synthesis with the following script.

  .. code:: bash

      unset XCL_EMULATION_MODE
      export XCL_TARGET=hw

      python build.py

- Create AWS FPGA image and upload it to AWS S3.

  .. code:: bash

      ${SDACCEL_DIR}/tools/create_sdaccel_afi.sh \
          -xclbin=myadd.xclbin -o=myadd \
          -s3_bucket=<bucket-name> -s3_dcp_key=<dcp-folder-name> \
          -s3_logs_key=<logs-folder-name>

  This also generates an awsxclbin file, which is necessary to use the AWS FPGA image on F1 instances.

Run
---

- Launch Amazon EC2 F1 instance.
- Copy ``myadd.so``, ``myadd.awsxclbin``, and ``run.py`` to the F1 instance.
- Setup AWS FPGA development kit.

  .. code:: bash

      git clone https://github.com/aws/aws-fpga.git
      cd aws-fpga
      source sdaccel_setup.sh

- Setup TVM with OpenCL enabled.
- Become root and setup environment variables.

  .. code:: bash

      sudo sh
      source ${INSTALL_ROOT}/setup.sh

- Run

  .. code:: bash

      python run.py
