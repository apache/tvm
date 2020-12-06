#!/bin/bash
set -e
set -u
set -o pipefail

service ssh start
service ssh status

mkdir -p /workspace
jupyter lab --ip 0.0.0.0 --allow-root --notebook-dir /workspace