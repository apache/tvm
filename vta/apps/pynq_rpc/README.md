# PYNQ RPC Server for VTA

This guide describes how to setup a Pynq-based RPC server to accelerate deep learning workloads with VTA.

## Pynq Setup

Follow the getting started tutorial for the [Pynq board](http://pynq.readthedocs.io/en/latest/getting_started.html).
* This assumes that you've downloaded the latest Pynq image, PYNQ-Z1 v2.1 (released 21 Feb 2018).
* For this RPC setup, follow the ["Connect to a Computer"](http://pynq.readthedocs.io/en/latest/getting_started.html#connect-to-a-computer) Pynq setup instructions.
* To be able to talk to the board, you'll need to make sure that you've followed the steps to [assign a static IP address](http://pynq.readthedocs.io/en/latest/appendix.html#assign-your-computer-a-static-ip)

Make sure that you can talk to your Pynq board successfully:
```bash
ping 192.168.2.99
```

When ssh-ing onto the board, the password for the `xilinx` username is `xilinx`.

For convenience let's go ahead and mount the Pynq board's file system to easily access it (this will require sshfs to be installed):
```bash
mkdir <mountpoint>
sshfs xilinx@192.168.2.99:/home/xilinx <mountpoint>
```

## Pynq TVM & VTA installation

On your **host PC**, go to the `<mountpoint>` directory of your Pynq board file system.
```bash
cd <mountpoint>
```

From there, clone the VTA repository:
```bash
git clone git@github.com:uwsaml/vta.git --recursive
```

Now, ssh into your **Pynq board** to build the TVM runtime with the following commands. This build should take about 5 minutes.
```bash
ssh xilinx@192.168.2.99 # ssh if you haven't done so
cd ~/vta/nnvm/tvm
cp make/config.mk .
echo USE_RPC=1 >> config.mk
make runtime -j2
```

We're now ready to build the Pynq RPC server on the Pynq board, which should take less than 30 seconds.
```bash
ssh xilinx@192.168.2.99 # ssh if you haven't done so
cd ~/vta
make -j2
```

The last stage will build the `vta/lib/libvta.so` library file. We are now ready to launch the RPC server on the Pynq. In order to enable the FPGA drivers, we need to run the RPC server with `sudo` privileges.
```bash
ssh xilinx@192.168.2.99 # ssh if you haven't done so
cd ~/vta
sudo ./apps/pynq_rpc/start_rpc_server.sh # pw is xilinx
```

You should see the following being displayed when starting the RPC server:
```
INFO:root:Load additional library /home/xilinx/vta/lib/libvta.so
INFO:root:RPCServer: bind to 0.0.0.0:9091
```

Note that it should be listening on port `9091`.

To kill the RPC server, just enter the `Ctrl + c` command.
