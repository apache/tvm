VTA: Open, Modular, Deep Learning Accelerator Stack
===================================================
[![Build Status](http://mode-gpu.cs.washington.edu:8080/buildStatus/icon?job=uwsaml/vta/master)](http://mode-gpu.cs.washington.edu:8080/job/uwsaml/job/vta/job/master/)
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)

VTA(versatile tensor accelerator) is an open-source deep learning accelerator stack.
It is not just an open-source hardware, but is an end to end solution that includes
the entire software stack on top of VTA open-source hardware.


The key features include:

- Generic, modular open-source hardware
  - Streamlined workflow to deploy to FPGAs.
  - Simulator support
- Driver and JIT runtime for both simulated backend and FPGA.
- End to end TVM stack integration
  - Direct optimization and deploy models from deep learning frameworks via TVM stack.
  - Customized and extendible TVM compiler backend
  - Flexible RPC support to ease the deployment, you can program it with python :)

VTA is part of our effort on [TVM Stack](http://www.tvmlang.org/).

License
-------
© Contributors, 2018. Licensed under an [Apache-2.0](https://github.com/tmoreau89/vta/blob/master/LICENSE) license.
