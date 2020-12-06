#!/bin/bash

set -e
set -u
set -o pipefail

wget -q https://github.com/cdr/code-server/releases/download/v3.7.4/code-server_3.7.4_amd64.deb -O /tmp/code-server.deb
apt-get install -y /tmp/code-server.deb
rm /tmp/code-server.deb

pip3 install jupyter-vscode-proxy