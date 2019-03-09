#!/bin/bash

set -e
set -u
set -o pipefail

apt-get update
apt-get install -y curl

# The node install script fetched and executed here will update the
# apt source list, hence the second apt-get update is necessary.
curl -s -S -L https://deb.nodesource.com/setup_6.x | bash -
apt-get update
apt-get install -y nodejs

npm install eslint jsdoc ws
