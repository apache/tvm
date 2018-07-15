VTA: Open, Modular, Deep Learning Accelerator Stack
===================================================
VTA(versatile tensor accelerator) is an open-source deep learning accelerator stack.
It is not just an open-source hardware, but is an end to end solution that includes
the entire software stack on top of VTA open-source hardware.

The key features include:

- Generic, modular open-source hardware
  - Streamlined workflow to deploy to FPGAs.
  - Simulator support to protoype compilation passes on regular workstations.
- Driver and JIT runtime for both simulated and FPGA hardware backend.
- End to end TVM stack integration
  - Direct optimization and deploy models from deep learning frameworks via TVM stack.
  - Customized and extendible TVM compiler backend.
  - Flexible RPC support to ease the deployment, and program FPGAs with Python

VTA is part of our effort on TVM Stack.
