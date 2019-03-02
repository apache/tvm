#!/bin/bash

set -e
set -u
set -o pipefail

pip2 install keras tensorflow h5py
pip3 install keras tensorflow h5py
