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

workdir="$(mktemp -d)"

cleanup()
{
    rm -rf "$workdir"
}

trap cleanup 0

SCCACHE_VERSION="v0.14.0"
SCCACHE_BASE_URL="https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}"

arch="$(uname -m)"
case "${arch}" in
    x86_64)
        target="x86_64-unknown-linux-musl"
        expected_sha256="8424b38cda4ecce616a1557d81328f3d7c96503a171eab79942fad618b42af44"
        ;;
    aarch64|arm64)
        target="aarch64-unknown-linux-musl"
        expected_sha256="62a6c942c47c93333bc0174704800cef7edfa0416d08e1356c1d3e39f0b462f2"
        ;;
    *)
        echo "Unsupported architecture '${arch}' for prebuilt sccache ${SCCACHE_VERSION}" >&2
        exit 1
        ;;
esac

archive="sccache-${SCCACHE_VERSION}-${target}.tar.gz"

download-and-verify "${SCCACHE_BASE_URL}/${archive}" "${workdir}/${archive}" sha256 "${expected_sha256}"

tar -xzf "${workdir}/${archive}" -C "${workdir}"

mkdir -p /opt/sccache
install -m 0755 "${workdir}/sccache-${SCCACHE_VERSION}-${target}/sccache" /opt/sccache/sccache

# The docs specifically recommend hard links: https://github.com/mozilla/sccache#known-caveats
ln -f /opt/sccache/sccache /opt/sccache/cc
ln -f /opt/sccache/sccache /opt/sccache/c++

# Only add clang if it's on the PATH
if command -v clang >/dev/null 2>&1
then
    ln -f /opt/sccache/sccache /opt/sccache/clang
    ln -f /opt/sccache/sccache /opt/sccache/clang++
fi

# make sccache usable by all users after install during container build
chmod -R a+rw /opt/sccache
