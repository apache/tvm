#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/home/xilinx/tvm/python:/home/xilinx/vta/python
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/python3.6/lib/python3.6/site-packages/pynq/drivers/
python -m vta.exec.rpc_server
