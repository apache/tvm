VTA Installation Guide
======================

We present three installation guides, each extending on the previous one:
1. [Simulator installation](#vta-simulator-installation)
2. [Hardware test setup](#vta-pynq-based-test-setup)
3. [FPGA toolchain installation](#vta-fpga-toolchain-installation)

## VTA Simulator Installation

You need [TVM installed](https://docs.tvm.ai/install/index.html) on your machine.
For a quick and easy start, use the pre-built [TVM Docker image](https://docs.tvm.ai/install/docker.html).

The VTA simulator library is built by default with TVM.
Add the VTA library to your python path to run the VTA examples.

```bash
export PYTHONPATH=/path/to/vta/python:${PYTHONPATH}
```

### Testing your VTA Simulation Setup

To ensure that you've properly installed the VTA python package, run the following 2D convolution testbench.

```bash
python <tvm root>/vta/tests/python/integration/test_benchmark_topi_conv2d.py
```

> Note: You'll notice that for every convolution layer, the throughput gets reported in GOPS. These numbers are actually the computational throughput that the simulator achieves, by evaluating the convolutions in software.

You are invited to try out our [VTA programming tutorials](https://docs.tvm.ai/vta/tutorials/index.html).


### Advanced Configuration (optional)

VTA is a generic configurable deep learning accelerator.
The configuration is specified by `vta_config.json` under the TVM root folder.
This file provides an architectural specification of the VTA accelerator to parameterize the TVM compiler stack and the VTA hardware stack.

The VTA configuration file also specifies the TVM compiler target.
When `TARGET` is set to `sim`, all TVM workloads execute on the VTA simulator.
You can modify the content of the configuration file to rebuild VTA to a different parameterization.
To do so,

```bash
cd <tvm root>
cp vta/config/vta_config.json vta_config.json
# edit vta_config.json
make vta
```

## VTA Pynq-Based Test Setup

This second guide extends the *VTA Simulator Installation* guide above to run FPGA hardware tests of the complete TVM and VTA software-hardware stack.
In terms of hardware components you'll need:
* The [Pynq](http://www.pynq.io/) FPGA development board which can be acquired for $200, or $150 for academics from [Digilent](https://store.digilentinc.com/pynq-z1-python-productivity-for-zynq/).
* An Ethernet-to-USB adapter to connect the Pynq board to your development machine.
* An 8+GB micro SD card.
* An AC to DC 12V 3A power adapter.

This guide covers the following themes:
1. Pynq board setup instructions.
2. Pynq-side RPC server build and deployment.
3. Revisiting the test examples from the *VTA Simulator Installation* guide, this time executing on the Pynq board.

### Pynq Board Setup

Setup your Pynq board based on the [Pynq board getting started tutorial](http://pynq.readthedocs.io/en/latest/getting_started.html).
You should follow the instructions up to and including the *Turning On the PYNQ-Z1* step (no need to pursue the tutorial beyond this point).
* Make sure that you've downloaded the latest Pynq image, [PYNQ-Z1 v2.1](http://pynq-testing.readthedocs.io/en/image_v2.2/getting_started/pynq_image.html) (released 21 Feb 2018), and have imaged your SD card with it (we recommend the free [Etcher](https://etcher.io/) program).
* For this test setup, follow the ["Connect to a Computer"](http://pynq.readthedocs.io/en/latest/getting_started.html#connect-to-a-computer) Ethernet setup instructions. To be able to talk to the board, make sure to [assign your computer a static IP address](http://pynq.readthedocs.io/en/latest/appendix.html#assign-your-computer-a-static-ip)

Once the board is powered on and connected to your development machine, try connecting to it to make sure you've properly set up your Pynq board:
```bash
# To connect to the Pynq board use the [username, password] combo: [xilinx, xilinx]
ssh xilinx@192.168.2.99
```

### Pynq-Side RPC Server Build & Deployment

Because the direct board-to-computer connection prevents the board from directly accessing the internet, we'll need to mount the Pynq's file system to your development machine's file system with [sshfs](https://www.digitalocean.com/community/tutorials/how-to-use-sshfs-to-mount-remote-file-systems-over-ssh). Next we directly clone the TVM repository into the sshfs mountpoint on your development machine.

```bash
# On the Host-side
mkdir <mountpoint>
sshfs xilinx@192.168.2.99:/home/xilinx <mountpoint>
cd <mountpoint>
git clone --recursive https://github.com/dmlc/tvm
# When finished, you can leave the moutpoint and unmount the directory
cd ~
sudo umount <mountpoint>
```

Now that we've cloned the VTA repository in the Pynq's file system, we can ssh into it and launch the build of the TVM-based RPC server.
The build process should take roughly 5 minutes.

```bash
ssh xilinx@192.168.2.99
# Build TVM runtime library (takes 5 mins)
cd /home/xilinx/tvm
mkdir build
cp cmake/config.cmake build/.
# Copy pynq specific configuration
cp vta/config/pynq_sample.json build/vta_config.json
cd build
cmake ..
make runtime vta -j2
# Build VTA RPC server (takes 1 min)
cd ..
sudo ./apps/pynq_rpc/start_rpc_server.sh # pw is 'xilinx'
```

You should see the following being displayed when starting the RPC server. In order to run the next examples, you'll need to leave the RPC server running in an `ssh` session.
```
INFO:root:RPCServer: bind to 0.0.0.0:9091
```

Tips regarding the Pynq RPC Server:
* The RPC server should be listening on port `9091`. If not, an earlier process might have terminated unexpectedly and it's recommended in this case to just reboot the Pynq, and re-run the RPC server.
* To kill the RPC server, just send the `Ctrl + c` command. You can re-run it with `sudo ./apps/pynq_rpc/start_rpc_server.sh`.
* If unresponsive, the board can be rebooted by power-cycling it with the physical power switch.

### Testing your Pynq-based Hardware Setup

Before running the examples on your development machine, you'll need to configure your host environment as follows:
```bash
# On the Host-side
export VTA_PYNQ_RPC_HOST=192.168.2.99
export VTA_PYNQ_RPC_PORT=9091
```

In addition, you'll need to edit the `vta_config.json` file on the host to indicate that we are targeting the Pynq platform, by setting the `TARGET` field to `"pynq"`.
Alternatively, you can copy the default `vta/config/pynq_sample.json` into the TVM root as `vta_config.json`.
> Note: in contrast to our simulation setup, there are no libraries to compile on the host side since the host offloads all of the computation to the Pynq board.

```bash
# On the Host-side
cd <tvm root>
cp vta/config/pynq_sample.json vta_config.json
```

This time again, we will run the 2D convolution testbench.
Beforehand, we need to program the Pynq board FPGA with a VTA bitstream, and build the VTA runtime via RPC.
The following `test_program_rpc.py` script will perform two operations:
* FPGA programming, by downloading a pre-compiled bitstream from a [VTA bitstream repository](https://github.com/uwsaml/vta-distro) that matches the default `vta_config.json` configuration set by the host, and sending it over to the Pynq via RPC to program the Pynq's FPGA.
* Runtime building on the Pynq, which needs to be run every time the `vta_config.json` configuration is modified. This ensures that the VTA software runtime that generates the accelerator's executable via just-in-time (JIT) compilation matches the specifications of the VTA design that is programmed on the FPGA. The build process takes about 30 seconds to complete so be patient!

```bash
# On the Host-side
python <tvm root>/vta/tests/python/pynq/test_program_rpc.py
```

> Tip: You can track progress of the FPGA programming and the runtime rebuilding steps by looking at the RPC server's logging messages in your Pynq `ssh` session.

We are now ready to run the 2D convolution testbench in hardware.

```bash
# On the Host-side
python <tvm root>/vta/tests/python/integration/test_benchmark_topi_conv2d.py
```

The performance metrics measured on the Pynq board will be reported for each convolutional layer.

You can also try out our [VTA programming tutorials](https://docs.tvm.ai/vta/tutorials/index.html).


## VTA FPGA Toolchain Installation

This third and last guide allows users to generate custom VTA bitstreams using free-to-use Xilinx compilation toolchains.

### Xilinx Toolchain Installation

We recommend using `Vivado 2018.2` since our scripts have been tested to work on this version of the Xilinx toolchains.
Our guide is written for Linux (Ubuntu) installation.

You’ll need to install Xilinx’ FPGA compilation toolchain, [Vivado HL WebPACK 2018.2](https://www.xilinx.com/products/design-tools/vivado.html), which a license-free version of the Vivado HLx toolchain.

#### Obtaining and Launching the Vivado GUI Installer

1. Go to the [download webpage](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vivado-design-tools/2018-2.html), and download the Linux Self Extracting Web Installer for Vivado HLx 2018.2: WebPACK and Editions.
2. You’ll have to sign in with a Xilinx account. This requires a Xilinx account creation that will take 2 minutes.
3. Complete the Name and Address Verification by clicking “Next”, and you will get the opportunity to download a binary file, called `Xilinx_Vivado_SDK_Web_2018.2_0614_1954_Lin64.bin`.
4. Now that the file is downloaded, go to your `Downloads` directory, and change the file permissions so it can be executed:
```bash
chmod u+x Xilinx_Vivado_SDK_Web_2018.2_0614_1954_Lin64.bin
```
5. Now you can execute the binary:
```bash
./Xilinx_Vivado_SDK_Web_2018.2_0614_1954_Lin64.bin
```

#### Xilinx Vivado GUI Installer Steps

At this point you've launched the Vivado 2017.1 Installer GUI program.

1. Click “Next” on the *Welcome* screen.
2. On the *Select Install Type* screen, enter your Xilinx user credentials under the “User Authentication” box and select the “Download and Install Now” option before clicking “Next” .
3. On the *Accept License Agreements* screen, accept all terms before clicking “Next”.
4. On the *Select Edition to Install* screen, select the “Vivado HL WebPACK” before clicking “Next” .
5. Under the *Vivado HL WebPACK* screen, before hitting “Next", check the following options (the rest should be unchecked):
   * Design Tools -> Vivado Design Suite -> Vivado
   * Devices -> Production Devices -> SoCs -> Zynq-7000 (if you are targeting the Pynq board)
   * Devices -> Production Devices -> SoCs -> UltraScale+ MPSoC (if you are targeting the Ultra-96 board)
6. Your total download size should be about 5GB and the amount of Disk Space Required 23GB.
7. On the *Select Destination Directory* screen, set the installation directory before clicking “Next”. It might highlight some paths as red - that’s because the installer doesn’t have the permission to write to the directory. In that case select a path that doesn’t require special write permissions (e.g. your home directory).
8. On the *Installation Summary* screen, hit “Install”.
9. An *Installation Progress* window will pop-up to track progress of the download and the installation.
10. This process will take about 20-30 minutes depending on your connection speed.
11. A pop-up window will inform you that the installation completed successfully. Click "OK".
12. Finally the *Vivado License Manager* will launch. Select "Get Free ISE WebPACK, ISE/Vivado IP or PetaLinux License" and click "Connect Now" to complete the license registration process.

#### Environment Setup

The last step is to update your `~/.bashrc` with the following lines. This will include all of the Xilinx binary paths so you can launch compilation scripts from the command line.
```bash
# Xilinx Vivado 2018.2 environment
export XILINX_VIVADO=${XILINX_PATH}/Vivado/2018.2
export PATH=${XILINX_VIVADO}/bin:${PATH}
```

### Custom VTA Bitstream Compilation

High-level hardware parameters are listed in the VTA configuration file and can be customized by the user.
For this custom VTA bitstream compilation exercise, we'll change the frequency of our design, so it can be clocked a little faster.
* Set the `HW_FREQ` field to `142`. The Pynq board supports 100, 142, 167 and 200MHz clocks. Note that the higher the frequency, the harder it will be to close timing. Increasing the frequency can lead to timing violation and thus faulty hardware execution.
* Set the `HW_CLK_TARGET` to `6`. This parameters refers to the target clock period in nano seconds for HLS - a lower clock period leads to more aggressive pipelining to achieve timing closure at higher frequencies. Technically a 142MHz clock would require a 7ns target, but we intentionally lower the clock target to 6ns to more aggressively pipeline our design.

Bitstream generation is driven by a top-level `Makefile` under `<tvm root>/vta/hardware/xilinx/`.

If you just want to simulate the VTA design in software emulation to make sure that it is functional, enter:
```bash
cd <tvm root>/vta/hardware/xilinx
make ip MODE=sim
```

If you just want to generate the HLS-based VTA IP cores without launching the entire design place and route, enter:
```bash
make ip
```
You'll be able to view the HLS synthesis reports under `<tvm root>/vta/build/hardware/xilinx/hls/` `<configuration>/<block>/solution0/syn/report/<block>_csynth.rpt`
> Note: The `<configuration>` name is a string that summarizes the VTA configuration parameters listed in the `vta_config.json`. The `<block>` name refers to the specific module (or HLS function) that compose the high-level VTA pipeline.

Finally to run the full hardware compilation and generate the VTA bitstream, run:

```bash
make
```

This process is lengthy, and can take around up to an hour to complete depending on your machine's specs.
We recommend setting the `VTA_HW_COMP_THREADS` variable in the Makefile to take full advantage of all the cores on your development machine.

Once the compilation completes, the generated bitstream can be found under `<tvm root>/vta/build/hardware/xilinx/vivado/<configuration>/export/vta.bit`.

### Use the Custom Bitstream

We can program the new VTA FPGA bitstream by setting the bitstream path of the `vta.program_fpga()` function in the tutorial examples, or in the `test_program_rpc.py` script.

```python
vta.program_fpga(remote, bitstream="<tvm root>/vta/build/hardware/xilinx/vivado/<configuration>/export/vta.bit")
```

Instead of downloading a pre-built bitstream from the VTA bitstream repository, TVM will instead use the new bitstream you just generated, which is a VTA design clocked at a higher frequency.
Do you observe a noticeable performance increase on the ImageNet classification example?
