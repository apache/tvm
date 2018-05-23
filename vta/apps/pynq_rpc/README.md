### PYNQ RPC Server for VTA

This guide describes how to setup a Pynq-based RPC server to accelerate deep learning workloads with VTA.

## Pynq Setup

Follow the getting started tutorial for the [Pynq board](http://pynq.readthedocs.io/en/latest/getting_started.html).
* For this RPC setup make sure to go with the *Connect to a Computer* Ethernet setup.

Make sure that you can ssh into your Pynq board successfully:
```bash
ssh xilinx@192.168.2.99
```

When ssh-ing onto the board, the default password for the `xilinx` account is `xilinx`.

For convenience let's go ahead and mount the Pynq board's file system to easily access it and maintain it:
```bash
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

Now, ssh into your **Pynq board** to build the TVM runtime with the following commands:
```bash
ssh xilinx@192.168.2.99 # ssh if you haven't done so
cd ~/vta/nnvm/tvm
cp make/config.mk .
echo USE_RPC=1 >> config.mk
make runtime -j2
```

## Pynq RPC server setup

We're now ready to build the Pynq RPC server on the Pynq board.
```bash
ssh xilinx@192.168.2.99 # ssh if you haven't done so
cd ~/vta
make
```

The last stage will build the `192.168.2.99:home/xilinx/vta/lib/libvta.so` library file. We are now ready to launch the RPC server on the Pynq. In order to enable the FPGA drivers, we need to run the RPC server with administrator privileges (using `su`, account: `xilinx`, pwd: `xilinx`).
```bash
ssh xilinx@192.168.2.99 # ssh if you haven't done so
cd ~/vta
su
./apps/pynq_rpc/start_rpc_server.sh
```

You should see the following being displayed when starting the RPC server:
```
INFO:root:Load additional library /home/xilinx/vta/lib/libvta.so
INFO:root:RPCServer: bind to 0.0.0.0:9091
```

Note that it should be listening on port `9091`.

To kill the RPC server, just enter the `Ctrl + c` command.