#!/bin/bash

set -e
set -u
set -o pipefail

echo deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-4.0 main\
     >> /etc/apt/sources.list.d/llvm.list
echo deb-src http://apt.llvm.org/xenial/ llvm-toolchain-xenial-4.0 main\
     >> /etc/apt/sources.list.d/llvm.list

echo deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-5.0 main\
     >> /etc/apt/sources.list.d/llvm.list
echo deb-src http://apt.llvm.org/xenial/ llvm-toolchain-xenial-5.0 main\
     >> /etc/apt/sources.list.d/llvm.list

echo deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-6.0 main\
     >> /etc/apt/sources.list.d/llvm.list
echo deb-src http://apt.llvm.org/xenial/ llvm-toolchain-xenial-6.0 main\
     >> /etc/apt/sources.list.d/llvm.list

echo deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial main\
     >> /etc/apt/sources.list.d/llvm.list
echo deb-src http://apt.llvm.org/xenial/ llvm-toolchain-xenial main\
     >> /etc/apt/sources.list.d/llvm.list

wget -q -O - http://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add -
apt-get update && apt-get install -y llvm-4.0 llvm-5.0 llvm-6.0 clang-6.0
