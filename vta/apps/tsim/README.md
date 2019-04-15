# tsim-steps

*TSIM* can be used in both OSX and Linux. `verilator` is used as simulation backend and `sbt` for Chisel3.

## OSX dependencies

```bash
brew install verilator sbt
```

## Linux dependencies

### Add sbt to package manager (Debian-based)

```bash
echo "deb https://dl.bintray.com/sbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823
sudo apt-get update
```

### Install both sbt and verilator

```bash
sudo apt install verilator sbt
```

## Setup steps

1. Download repo and build tvm (build instructions [here](https://docs.tvm.ai/install/from_source.html))

```bash
git clone git@github.com:vegaluisjose/tvm.git
git checkout tsim
```

2. Add-by-one Verilog example

```
cd tvm/vta/apps/tsim
make
```

3. Add-by-one Chisel3 example
    * Open `tvm/vta/apps/tsim/python/tsim/config.json`
    * Change `TARGET` from `verilog` to `chisel`
    * Go to `tvm/vta/apps/tsim`
    * Run `make`
