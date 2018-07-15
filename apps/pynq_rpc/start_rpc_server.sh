#!/bin/bash
PROJROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../" && pwd )"

export PYTHONPATH=${PYTHONPATH}:${PROJROOT}/python:${PROJROOT}/vta/python
python -m vta.exec.rpc_server
