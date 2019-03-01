#!/bin/bash

set -e
set -u
set -o pipefail

#install the necessary dependancies, cffi, opencv
wget -q 'https://github.com/siju-samuel/darknet/blob/master/lib/libdarknet.so?raw=true' -O libdarknet.so
pip2 install opencv-python cffi
pip3 install opencv-python cffi
