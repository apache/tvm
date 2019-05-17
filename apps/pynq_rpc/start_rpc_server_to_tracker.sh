#!/bin/bash
PROJROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../" && pwd )"

export PYTHONPATH=${PYTHONPATH}:${PROJROOT}/python:${PROJROOT}/vta/python
python3.6 -m vta.exec.rpc_server --tracker fleet:9190 --key pynq
