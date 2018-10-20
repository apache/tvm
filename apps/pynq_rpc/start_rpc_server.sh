#!/bin/bash
PROJROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../" && pwd )"

export PYTHONPATH=${PYTHONPATH}:${PROJROOT}/python:${PROJROOT}/vta/python
export PYTHONPATH=${PYTHONPATH}:/home/xilinx/pynq
python3 -m vta.exec.rpc_server
