#!/bin/bash

set -e
set -u
set -o pipefail

#install the necessary dependancies for golang build
apt-get update
apt-get install -y golang-1.10-go
apt-get install -y golang-1.10-doc
apt-get install -y golint
