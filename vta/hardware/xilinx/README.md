# Hardware Compilation Guide

**This hardware compilation guide aims to provide guidance on generating VTA bitstreams with the Xilinx Vivado toolchains.**

As of writing this guide, we recommend using `Vivado 2017.1` since our scripts have been tested to work on this version of the Xilinx toolchains.

# Vivado Toolchains Installation for Pynq Board

## Ubuntu instructions

You’ll need to install Xilinx’ FPGA compilation toolchain, [Vivado HL WebPACK 2017.1](https://www.xilinx.com/products/design-tools/vivado.html), which a license-free version of the Vivado HLx toolchain.

### Obtaining and launching the installation binary
 
1. Go to the [download webpage](https://www.xilinx.com/support/download.html), and download the Linux Self Extracting Web Installer for Vivado HL 2017.1 WebPACK and Editions.
2. You’ll have to sign in with a Xilinx account. This requires a Xilinx account creation that will take 2 minutes.
3. Complete the Name and Address Verification by clicking “Next”, and you will get the opportunity to download a binary file, called `Xilinx_Vivado_SDK_2017.1_0415_1_Lin64.bin`.
4. Now that the file is downloaded, go to your `Downloads` directory, and change the file permissions so it can be executed:
```bash
chmod u+x Xilinx_Vivado_SDK_2017.1_0415_1_Lin64.bin
```
5. Now you can execute the binary: 
```bash
./Xilinx_Vivado_SDK_2017.1_0415_1_Lin64.bin
```

### Installation Steps

At this point you've launched the Vivado 2017.1 Installer GUI program.

1. Click “Next” on the **Welcome** screen.
2. Enter your Xilinx User Credentials under “User Authentication” and select the “Download and Install Now” before clicking “Next” on the **Select Install Type** screen.
3. Accept all terms before clicking on “Next” on the **Accept License Agreements** screen.
4. Select the “Vivado HL WebPACK” before clicking on “Next” on the **Select Edition to Install** screen.
5. Under the **Vivado HL WebPACK** screen, before hitting “Next", check the following options (the rest should be unchecked):
   * Design Tools -> Vivado Design Suite -> Vivado
   * Design Tools -> Vivado Design Suite -> Vivado High Level Synthesis
   * Devices -> Production Services -> SoCs -> Zynq-7000 Series
6. Your total download size should be about 3GB and the amount of Disk Space Required 13GB.
7. Set the installation directory before clicking “Next” on the **Select Destination Directory** screen. It might highlight some paths as red - that’s because the installer doesn’t have the permission to write to that directory. In that case select a path that doesn’t require special write permissions (e.g. in your home directory).
8. Hit “Install” under the **Installation Summary** screen.
9. An **Installation Progress Window** will pop-up to track progress of the download and the installation.
10. This process will take about 20-30 minutes depending on your connection speed.
11. A pop-up window will inform you that the installation completed successfully. Click "OK".
12. Finally the **Vivado License Manager** will launch. Select "Get Free ISE WebPACK, ISE/Vivado IP or PetaLinux License" and click "Connect Now" to complete the license registration process. 

### Environment Setup

The last step is to update your `~/.bashrc` with the following line:
```bash
# Xilinx Vivado 2017.1 environment
source <install_path>/Vivado/2017.1/settings64.sh
```

This will include all of the Xilinx binary paths so you can launch compilation scripts from the command line.

Note that this will overwrite the paths to GCC required to build TVM, or NNVM. Therefore, when attempting to build TVM and NNVM, please comment this line from your `~/.bashrc` before re-sourcing it.

# Bitstream compilation

High-level parameters are listed under `<vta root>/make/config.mk` and can be customized by the user.

Bitstream generation is driven by a makefile. All it takes is to enter the following command:
```bash
make
```

The local `Makefile` containts several variables that can be tweaked by the user:
* `VTA_HW_COMP_THREADS`: determines the number of threads used for the Vivado compilation job (default 8 threads).

Once the compilation completes, the generated bitstream can be found under `<vta root>/build/hardware/xilinx/vivado/<design name>/export/vta.bit`. 