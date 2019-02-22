#!/bin/bash

set -e
set -u
set -o pipefail

apt-get update --fix-missing

apt-get install -y --no-install-recommends \
        libgl1-mesa-dev libglfw3-dev
