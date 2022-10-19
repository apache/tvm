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

VTA Installation Guide
======================

We present three installation guides, each extending on the previous one:

1. `VTA Simulator Installation`_
2. `Xilinx Pynq FPGA Setup`_
3. `Intel DE10 FPGA Setup`_
4. `Bitstream Generation with Xilinx Toolchains`_
5. `Bitstream Generation with Intel Toolchains`_


VTA Simulator Installation
--------------------------

You need :ref:`TVM installed <installation>` on your machine.  For a quick and
easy start, checkout the :ref:`Docker Guide <docker-images>`.

You'll need to set the following paths to use VTA:

.. code:: bash

   export TVM_PATH=<path to TVM root>
   export VTA_HW_PATH=$TVM_PATH/3rdparty/vta-hw

The VTA functional simulation library needs to be enabled when building TVM.

.. code:: bash

   cd <tvm-root>
   mkdir build
   cp cmake/config.cmake build/.
   echo 'set(USE_VTA_FSIM ON)' >> build/config.cmake
   cd build && cmake .. && make -j4

Add the VTA python library to your python path to run the VTA examples.

.. code:: bash

   export PYTHONPATH=/path/to/vta/python:${PYTHONPATH}

Testing your VTA Simulation Setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To ensure that you've properly installed the VTA python package, run the following 2D convolution testbench.

.. code:: bash

   python <tvm root>/vta/tests/python/integration/test_benchmark_topi_conv2d.py

You are invited to try out our :ref:`VTA programming tutorials <vta-tutorials>`.

   **Note**: You'll notice that for every convolution layer, the throughput gets reported in GOPS. These numbers are actually the computational throughput that the simulator achieves, by evaluating the convolutions in software.

Advanced Configuration (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

VTA is a generic configurable deep learning accelerator.
The configuration is specified by ``vta_config.json`` under ``3rdparty/vta-hw/config``.
This file provides an architectural specification of the VTA accelerator to parameterize the TVM compiler stack and the VTA hardware stack.

The VTA configuration file also specifies the TVM compiler target.
When ``TARGET`` is set to ``sim``, all TVM workloads execute on the VTA simulator.
You can modify the content of the configuration file to rebuild VTA to a different parameterization.
To do so,

.. code:: bash

   cd <tvm root>
   vim 3rdparty/vta-hw/config/vta_config.json
   # edit vta_config.json
   make



Xilinx Pynq FPGA Setup
----------------------

This second guide extends the *VTA Simulator Installation* guide above to run FPGA hardware tests of the complete TVM and VTA software-hardware stack.
In terms of hardware components you'll need:

* The `Pynq <http://www.pynq.io/>`_ FPGA development board which can be acquired for $200, or $150 for academics from `Digilent <https://store.digilentinc.com/pynq-z1-python-productivity-for-zynq/>`_.
* An Ethernet-to-USB adapter to connect the Pynq board to your development machine.
* An 8+GB micro SD card.
* An AC to DC 12V 3A power adapter.

This guide covers the following themes:

1. Pynq board setup instructions.
2. Pynq-side RPC server build and deployment.
3. Revisiting the test examples from the *VTA Simulator Installation* guide, this time executing on the Pynq board.

Pynq Board Setup
^^^^^^^^^^^^^^^^

Setup your Pynq board based on the `Pynq board getting started tutorial <http://pynq.readthedocs.io/en/latest/getting_started.html>`_.

You should follow the instructions up to and including the *Turning On the PYNQ-Z1* step (no need to pursue the tutorial beyond this point).

* Make sure that you've downloaded the latest Pynq image, `PYNQ-Z1 v2.5 <http://www.pynq.io/board.html>`_, and have imaged your SD card with it (we recommend the free `Etcher <https://etcher.io/>`_ program).
* For this test setup, follow the `"Connect to a Computer" <https://pynq.readthedocs.io/en/latest/getting_started/pynq_z1_setup.html>`_ Ethernet setup instructions. To be able to talk to the board, make sure to `assign your computer a static IP address <https://pynq.readthedocs.io/en/latest/appendix.html#assign-your-computer-a-static-ip>`_

Once the board is powered on and connected to your development machine, try connecting to it to make sure you've properly set up your Pynq board:

.. code:: bash

   # To connect to the Pynq board use the <username, password> combo: <xilinx, xilinx>
   ssh xilinx@192.168.2.99

Pynq-Side RPC Server Build & Deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because the direct board-to-computer connection prevents the board from directly accessing the internet, we'll need to mount the Pynq's file system to your development machine's file system with `sshfs <https://www.digitalocean.com/community/tutorials/how-to-use-sshfs-to-mount-remote-file-systems-over-ssh>`_. Next we directly clone the TVM repository into the sshfs mountpoint on your development machine.

.. code:: bash

   # On the Host-side
   mkdir <mountpoint>
   sshfs xilinx@192.168.2.99:/home/xilinx <mountpoint>
   cd <mountpoint>
   git clone --recursive https://github.com/apache/tvm tvm
   # When finished, you can leave the moutpoint and unmount the directory
   cd ~
   sudo umount <mountpoint>

Now that we've cloned the VTA repository in the Pynq's file system, we can ssh into it and launch the build of the TVM-based RPC server.
The build process should take roughly 5 minutes.

.. code:: bash

   ssh xilinx@192.168.2.99
   # Build TVM runtime library (takes 5 mins)
   cd /home/xilinx/tvm
   mkdir build
   cp cmake/config.cmake build/.
   echo 'set(USE_VTA_FPGA ON)' >> build/config.cmake
   # Copy pynq specific configuration
   cp 3rdparty/vta-hw/config/pynq_sample.json 3rdparty/vta-hw/config/vta_config.json
   cd build
   cmake ..
   make runtime vta -j2
   # FIXME (tmoreau89): remove this step by fixing the cmake build
   make clean; make runtime vta -j2
   # Build VTA RPC server (takes 1 min)
   cd ..
   sudo ./apps/vta_rpc/start_rpc_server.sh # pw is 'xilinx'


You should see the following being displayed when starting the RPC server. In order to run the next examples, you'll need to leave the RPC server running in an ``ssh`` session.

.. code:: bash

   INFO:root:RPCServer: bind to 0.0.0.0:9091


Tips regarding the Pynq RPC Server:

* The RPC server should be listening on port ``9091``. If not, an earlier process might have terminated unexpectedly and it's recommended in this case to just reboot the Pynq, and re-run the RPC server.
* To kill the RPC server, just send the ``Ctrl + c`` command. You can re-run it with ``sudo ./apps/pynq_rpc/start_rpc_server.sh``.
* If unresponsive, the board can be rebooted by power-cycling it with the physical power switch.

Testing your Pynq-based Hardware Setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before running the examples on your development machine, you'll need to configure your host environment as follows:

.. code:: bash

   # On the Host-side
   export VTA_RPC_HOST=192.168.2.99
   export VTA_RPC_PORT=9091


In addition, you'll need to edit the ``vta_config.json`` file on the host to indicate that we are targeting the Pynq platform, by setting the ``TARGET`` field to ``"pynq"``.
> Note: in contrast to our simulation setup, there are no libraries to compile on the host side since the host offloads all of the computation to the Pynq board.

.. code:: bash

   # On the Host-side
   cd <tvm root>
   cp 3rdparty/vta-hw/config/pynq_sample.json 3rdparty/vta-hw/config/vta_config.json


This time again, we will run the 2D convolution testbench.
Beforehand, we need to program the Pynq board FPGA with a VTA bitstream, and build the VTA runtime via RPC.
The following ``test_program_rpc.py`` script will perform two operations:

* FPGA programming, by downloading a pre-compiled bitstream from a `VTA bitstream repository <https://github.com/uwsampl/vta-distro>`_ that matches the default ``vta_config.json`` configuration set by the host, and sending it over to the Pynq via RPC to program the Pynq's FPGA.
* Runtime building on the Pynq, which needs to be run every time the ``vta_config.json`` configuration is modified. This ensures that the VTA software runtime that generates the accelerator's executable via just-in-time (JIT) compilation matches the specifications of the VTA design that is programmed on the FPGA. The build process takes about 30 seconds to complete so be patient!

.. code:: bash

   # On the Host-side
   python <tvm root>/vta/tests/python/pynq/test_program_rpc.py


We are now ready to run the 2D convolution testbench in hardware.

.. code:: bash

   # On the Host-side
   python <tvm root>/vta/tests/python/integration/test_benchmark_topi_conv2d.py

The performance metrics measured on the Pynq board will be reported for each convolutional layer.

**Tip**: You can track progress of the FPGA programming and the runtime rebuilding steps by looking at the RPC server's logging messages in your Pynq ``ssh`` session.

You can also try out our :ref:`VTA programming tutorials <vta-tutorials>`.

Intel DE10 FPGA Setup
---------------------

Similar to the Pynq-side setup steps, this third guide bring us the details on how can we setup up the Linux environment for Intel FPGA boards like DE10-Nano.

In terms of hardware components, you would need the `DE10-Nano Development Kit <https://www.terasic.com.tw/cgi-bin/page/archive.pl?Language=English&No=1046>`_, which can be acquired for $130, or $100 for academics from `Terasic <https://www.terasic.com.tw/>`_. A microSD card would be delivered the kit. Power cables and USB cables would be included as well. However, an additional Ethernet cable would be needed to connect the board to LAN.

The rest part of this guide would provide the steps to

* Flash the microSD card with latest Angstrom Linux image
* Cross-compilation setup
* Device-side RPC server setup and deployment

DE10-Nano Board Setup
^^^^^^^^^^^^^^^^^^^^^

Before powering up the device, we need to flash the microSD card image with latest Angstrom Linux image.

Flash SD Card and Boot Angstrom Linux
"""""""""""""""""""""""""""""""""""""

To flash SD card and boot Linux on DE10-Nano, it is recommended to navigate to the `Resource <https://www.terasic.com.tw/cgi-bin/page/archive.pl?Language=English&CategoryNo=167&No=1046&PartNo=4>`_ tab of the DE10-Nano product page from Terasic Inc.
After registration and login on the webpage, the prebuilt Angstrom Linux image would be available for downloading and flashing.
Specifically, to flash the downloaded Linux SD card image into your physical SD card:

First, extract the gzipped archive file.

.. code:: bash

   tar xf de10-nano-image-Angstrom-v2016.12.socfpga-sdimg.2017.03.31.tgz

This would produce a single SD card image named ``de10-nano-image-Angstrom-v2016.12.socfpga-sdimg`` (approx. 2.4 GB), it contains all the file systems to boot Angstrom Linux.

Second, plugin a SD card that is ready to flash in your PC, and identify the device id for the disk with ``fdisk -l``, or ``gparted`` if you feel better to use GUI. The typical device id for your disk would likely to be ``/dev/sdb``.

Then, flash the disk image into your physical SD card with the following command:

.. code:: bash

   # NOTE: root privilege is typically required to run the following command.
   dd if=de10-nano-image-Angstrom-v2016.12.socfpga-sdimg of=/dev/sdb status=progress

This would take a few minutes for your PC to write the whole file systems into the SD card.
After this process completes, you are ready to unmount the SD card and insert it into your DE10-Nano board.
Now you can connect the power cable and serial port to boot the Angstrom Linux.

   **Note**: When boot up from the microSD card, you might notice the incompatibility of the Linux kernel ``zImage`` in the microSD card.
   In this case, you might need to build the ``zImage`` file of your own from `socfpga-4.9.78-ltsi <https://github.com/altera-opensource/linux-socfpga/tree/socfpga-4.9.78-ltsi>`_ branch of the `linux-socfpga <https://github.com/altera-opensource/linux-socfpga>`_ repository.
   For a quick fix, you can also download a prebuilt version of the ``zImage`` file `from this link <https://raw.githubusercontent.com/liangfu/de10-nano-supplement/master/zImage>`_.

After connecting the usb cables to the DE10-Nano board, power on the board by connecting the power cable. You may then connect to the serial port of the device by using ``minicom`` on your host PC:

.. code:: bash

   # NOTE: root privilege is typically required to run the following command.
   minicom -D /dev/ttyUSB0

The default user name for the device would be ``root``, and the password is empty for the default user.

You may now start to install supporting Python3 packages (TVM has dropped the support for Python2), specifically, they are ``numpy``, ``attrs`` and ``decorator``.

   **Note**: You might fail to install ``numpy`` by using ``pip3`` on the DE10-Nano device.
   In that case, you have the option to either build your own filesystem image for the board from `meta-de10-nano <https://github.com/intel/meta-de10-nano>`_ repository;
   an alternative option is to download prebuilt packages from existing Linux distributions, e.g. Debian.
   For a quick fix, we have concatenated the supplementary binary files `here <https://raw.githubusercontent.com/liangfu/de10-nano-supplement/master/rootfs_supplement.tgz>`_, and you can extract the files into the root filesystem.

Install Required Python Packages
""""""""""""""""""""""""""""""""

After accessing bash terminal from the serial port, we need to install required Python packages before building and installing TVM and VTA programs.

Build Additional Components to Use VTA Bitstream
""""""""""""""""""""""""""""""""""""""""""""""""

To use the above built bitstream on DE10-Nano hardware, several additional components need to be compiled for the system.
Specifically, to compile application executables for the system, you need to download and install `SoCEDS <http://fpgasoftware.intel.com/soceds/18.1/?edition=standard&download_manager=dlm3&platform=linux>`_ (recommended), or alternatively install the ``g++-arm-linux-gnueabihf`` package on your host machine. You would also need a ``cma`` kernel module to allocate contigous memory, and a driver for communicating with the VTA subsystem.


Bitstream Generation with Xilinx Toolchains
-------------------------------------------

If you're interested in generating the Xilinx FPGA bitstream on your own instead of using the pre-built VTA bitstreams, follow the instructions below.

Xilinx Toolchain Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We recommend using Vivado 2020.1 since our scripts have been tested to work on this version of the Xilinx toolchains.
Our guide is written for Linux (Ubuntu) installation.

You’ll need to install Xilinx’ FPGA compilation toolchain, `Vivado HL WebPACK 2020.1 <https://www.xilinx.com/products/design-tools/vivado.html>`_, which a license-free version of the Vivado HLx toolchain.

Obtaining and Launching the Vivado GUI Installer
""""""""""""""""""""""""""""""""""""""""""""""""

1. Go to the `download webpage <https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vivado-design-tools/2020-1.html>`_, and download the Linux Self Extracting Web Installer for Vivado HLx 2020.1: WebPACK and Editions.
2. You’ll have to sign in with a Xilinx account. This requires a Xilinx account creation that will take 2 minutes.
3. Complete the Name and Address Verification by clicking “Next”, and you will get the opportunity to download a binary file, called ``Xilinx_Unified_2020.1_0602_1208_Lin64.bin``.
4. Now that the file is downloaded, go to your ``Downloads`` directory, and change the file permissions so it can be executed:

.. code:: bash

   chmod u+x Xilinx_Unified_2020.1_0602_1208_Lin64.bin

5. Now you can execute the binary:

.. code:: bash

   ./Xilinx_Unified_2020.1_0602_1208_Lin64.bin

Xilinx Vivado GUI Installer Steps
"""""""""""""""""""""""""""""""""

At this point you've launched the Vivado 2020.1 Installer GUI program.

1. Click “Next” on the "Welcome" screen.
2. On the "Select Install Type" screen, enter your Xilinx user credentials under the “User Authentication” box and select the “Download and Install Now” option before clicking “Next”.
3. On the "Accept License Agreements" screen, accept all terms before clicking “Next”.
4. On the "Select Edition to Install" screen, select the “Vivado HL WebPACK” before clicking “Next”.
5. Under the "Vivado HL WebPACK" screen, before hitting “Next", check the following options (the rest should be unchecked):
   * Design Tools -> Vivado Design Suite -> Vivado
   * Devices -> Production Devices -> SoCs -> Zynq-7000 (if you are targeting the Pynq board)
   * Devices -> Production Devices -> SoCs -> UltraScale+ MPSoC (if you are targeting the Ultra-96 board)
6. Your total download size should be about 5GB and the amount of Disk Space Required 23GB.
7. On the "Select Destination Directory" screen, set the installation directory before clicking “Next”. It might highlight some paths as red - that’s because the installer doesn’t have the permission to write to the directory. In that case select a path that doesn’t require special write permissions (e.g. your home directory).
8. On the "Installation Summary" screen, hit “Install”.
9. An "Installation Progress" window will pop-up to track progress of the download and the installation.
10. This process will take about 20-30 minutes depending on your connection speed.
11. A pop-up window will inform you that the installation completed successfully. Click "OK".
12. Finally the "Vivado License Manager" will launch. Select "Get Free ISE WebPACK, ISE/Vivado IP or PetaLinux License" and click "Connect Now" to complete the license registration process.

Environment Setup
"""""""""""""""""

The last step is to update your ``~/.bashrc`` with the following lines. This will include all of the Xilinx binary paths so you can launch compilation scripts from the command line.

.. code:: bash

   # Xilinx Vivado 2020.1 environment
   export XILINX_VIVADO=${XILINX_PATH}/Vivado/2020.1
   export PATH=${XILINX_VIVADO}/bin:${PATH}

HLS-based Custom VTA Bitstream Compilation for Pynq
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

High-level hardware parameters are listed in the VTA configuration file and can be customized by the user.
For this custom VTA bitstream compilation exercise, we'll change the frequency of our design, so it can be clocked a little faster.

* Set the ``HW_FREQ`` field to ``142``. The Pynq board supports 100, 142, 167 and 200MHz clocks. Note that the higher the frequency, the harder it will be to close timing. Increasing the frequency can lead to timing violation and thus faulty hardware execution.
* Set the ``HW_CLK_TARGET`` to ``6``. This parameters refers to the target clock period in nano seconds for HLS - a lower clock period leads to more aggressive pipelining to achieve timing closure at higher frequencies. Technically a 142MHz clock would require a 7ns target, but we intentionally lower the clock target to 6ns to more aggressively pipeline our design.

Bitstream generation is driven by a top-level ``Makefile`` under ``<tvm root>/3rdparty/vta-hw/hardware/xilinx/``.

If you just want to simulate the VTA design in software emulation to make sure that it is functional, enter:

.. code:: bash

   cd <tvm root>/3rdparty/vta-hw/hardware/xilinx
   make ip MODE=sim


If you just want to generate the HLS-based VTA IP cores without launching the entire design place and route, enter:

.. code:: bash

   make ip

You'll be able to view the HLS synthesis reports under ``<tvm root>/3rdparty/vta-hw/build/hardware/xilinx/hls/<configuration>/<block>/solution0/syn/report/<block>_csynth.rpt``

   **Note**: The ``<configuration>`` name is a string that summarizes the VTA configuration parameters listed in the ``vta_config.json``. The ``<block>`` name refers to the specific module (or HLS function) that compose the high-level VTA pipeline.

Finally to run the full hardware compilation and generate the VTA bitstream, run ``make``.

This process is lengthy, and can take around up to an hour to complete depending on your machine's specs.
We recommend setting the ``VTA_HW_COMP_THREADS`` variable in the Makefile to take full advantage of all the cores on your development machine.

Once the compilation completes, the generated bitstream can be found under ``<tvm root>/3rdparty/vta-hw/build/hardware/xilinx/vivado/<configuration>/export/vta.bit``.

Using A Custom Bitstream
^^^^^^^^^^^^^^^^^^^^^^^^

We can program the new VTA FPGA bitstream by setting the bitstream path of the ``vta.program_fpga()`` function in the tutorial examples, or in the ``test_program_rpc.py`` script.

.. code:: python

   vta.program_fpga(remote, bitstream="<tvm root>/3rdparty/vta-hw/build/hardware/xilinx/vivado/<configuration>/export/vta.bit")

Instead of downloading a pre-built bitstream from the VTA bitstream repository, TVM will instead use the new bitstream you just generated, which is a VTA design clocked at a higher frequency.
Do you observe a noticeable performance increase on the ImageNet classification example?



Bitstream Generation with Intel Toolchains
-------------------------------------------

If you're interested in generating the Xilinx FPGA bitstream on your own instead of using the pre-built VTA bistreams, follow the instructions below.

Intel Toolchain Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is recommended to use ``Intel Quartus Prime 18.1``, since the test scripts contained in this document have been tested on this version.

You would need to install Intel's FPGA compilation toolchain, `Quartus Prime Lite <http://fpgasoftware.intel.com/?edition=lite>`_, which is a license-free version of the Intel Quartus Prime software.

Obtaining and Launching the Quartus GUI Installer
"""""""""""""""""""""""""""""""""""""""""""""""""

1. Go to the `download center <http://fpgasoftware.intel.com/?edition=lite>`_, and download the linux version of "Quartus Prime (include Nios II EDS)" and "Cyclone V device support" files in the "Separate file" tab. This avoid downloading unused device support files.
2. Sign in the form if you have an account, or register on the right side of the web page to create an account.
3. After signed in, you are able to download the installer and the device support files.
4. Now that the files are downloaded, go to your ``Downloads`` directory, and change the file permissions:

.. code:: bash

   chmod u+x QuartusLiteSetup-18.1.0.625-linux.run

5. Now ensure both the installer and device support files are in the same directory, and you can run the install with:

.. code:: bash

   ./QuartusLiteSetup-18.1.0.625-linux.run

6. Follow the instructions on the pop-up GUI form, and install all the content in the ``/usr/local`` directory. After installation, ``/usr/local/intelFPGA_lite/18.1`` would be created and the Quartus program along with other programs would be available in the folder.

Environment Setup
"""""""""""""""""

Similar to what should be done for Xilinx toolchain, the following line should be added to your ``~/.bashrc``.

.. code:: bash

   # Intel Quartus 18.1 environment
   export QUARTUS_ROOTDIR="/usr/local/intelFPGA_lite/18.1/quartus"
   export PATH=${QUARTUS_ROOTDIR}/bin:${PATH}
   export PATH=${QUARTUS_ROOTDIR}/sopc_builder/bin:${PATH}

This would add quartus binary path into your ``PATH`` environment variable, so you can launch compilation scripts from the command line.

Chisel-based Custom VTA Bitstream Compilation for DE10-Nano
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Similar to the HLS-based design, high-level hardware parameters in Chisel-based design are listed in the VTA configuration file `Configs.scala <https://github.com/apache/tvm/blob/main/3rdparty/vta-hw/hardware/chisel/src/main/scala/core/Configs.scala>`_, and they can be customized by the user.

For Intel FPGA, bitstream generation is driven by a top-level ``Makefile`` under ``<tvm root>/3rdparty/vta-hw/hardware/intel``.

If you just want to generate the Chisel-based VTA IP core for the DE10-Nano board without compiling the design for the FPGA hardware, enter:

.. code:: bash

   cd <tvm root>/3rdparty/vta-hw/hardware/intel
   make ip

Then you'll be able to locate the generated verilog file at ``<tvm root>/3rdparty/vta-hw/build/hardware/intel/chisel/<configuration>/VTA.DefaultDe10Config.v``.

If you would like to run the full hardware compilation for the ``de10nano`` board:

.. code:: bash

   make

This process might be a bit lengthy, and might take up to half an hour to complete depending on the performance of your PC. The Quartus Prime software would automatically detect the number of cores available on your PC and try to utilize all of them to perform such process.

Once the compilation completes, the generated bistream can be found under ``<tvm root>/3rdparty/vta-hw/build/hardware/intel/quartus/<configuration>/export/vta.rbf``. You can also open the Quartus project file (.qpf) available at ``<tvm root>/3rdparty/vta-hw/build/hardware/intel/quartus/<configuration>/de10_nano_top.qpf`` to look around the generated reports.
