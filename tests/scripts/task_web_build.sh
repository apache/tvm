#!/bin/bash
cp /emsdk-portable/.emscripten ~/.emscripten
source /emsdk-portable/emsdk_env.sh
make -j4
