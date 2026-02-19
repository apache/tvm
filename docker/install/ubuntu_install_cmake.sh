#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

set -e
set -u
set -o pipefail

CMAKE_VERSION="3.31.11"
CMAKE_SHA256_X86_64="d815c10cf54e8e122088b3bb25ea6b4010fb96b7ad6e1ad3fdef75be3d996b0b"
CMAKE_SHA256_AARCH64="faef5420c3853d0e10a3862a0d3c71b40ffc2de36fc0ef0bcc5c896cf18f240f"

case "$(uname -m)" in
x86_64)
  CMAKE_ARCH="x86_64"
  CMAKE_SHA256="$CMAKE_SHA256_X86_64"
  ;;
aarch64|arm64)
  CMAKE_ARCH="aarch64"
  CMAKE_SHA256="$CMAKE_SHA256_AARCH64"
  ;;
*)
  echo "Unsupported architecture: $(uname -m)"
  exit 1
  ;;
esac

v=$(echo "$CMAKE_VERSION" | sed 's/\(.*\)\..*/\1/g')
CMAKE_PKG="cmake-${CMAKE_VERSION}-linux-${CMAKE_ARCH}.tar.gz"
CMAKE_URL="https://cmake.org/files/v${v}/${CMAKE_PKG}"

TMP_DIR=$(mktemp -d)
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup 0

echo "Installing cmake ${CMAKE_VERSION} (${CMAKE_ARCH})"
pushd "$TMP_DIR"
  download-and-verify "$CMAKE_URL" "$CMAKE_PKG" sha256 "$CMAKE_SHA256"

  mkdir -p /opt/cmake
  tar -xzf "$CMAKE_PKG" -C /opt/cmake

  CMAKE_DIR="/opt/cmake/cmake-${CMAKE_VERSION}-linux-${CMAKE_ARCH}"
  ln -sf "${CMAKE_DIR}/bin/cmake" /usr/local/bin/cmake
  ln -sf "${CMAKE_DIR}/bin/ctest" /usr/local/bin/ctest
  ln -sf "${CMAKE_DIR}/bin/cpack" /usr/local/bin/cpack
popd
