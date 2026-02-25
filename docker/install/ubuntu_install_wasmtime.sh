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

set -euxo pipefail

WASMTIME_VERSION="v41.0.3"
WASMTIME_HOME=/opt/wasmtime

case "$(uname -m)" in
	x86_64)
		WASMTIME_ARCH="x86_64-linux"
		WASMTIME_SHA256="797d0a4f790e79c33ccaf43bfe413f077fff951e3a35145afe7b5a8324f14644"
		;;
	aarch64|arm64)
		WASMTIME_ARCH="aarch64-linux"
		WASMTIME_SHA256="1dd1f69089eeefc3826f38463f8375d6ff2e59684a2a85b44a6622516d0a5677"
		;;
	*)
		echo "Unsupported architecture: $(uname -m)"
		exit 1
		;;
esac

WASMTIME_TARBALL="wasmtime-${WASMTIME_VERSION}-${WASMTIME_ARCH}.tar.xz"
WASMTIME_URL="https://github.com/bytecodealliance/wasmtime/releases/download/${WASMTIME_VERSION}/${WASMTIME_TARBALL}"

TMP_DIR="$(mktemp -d)"
cleanup() {
	rm -rf "${TMP_DIR}"
}
trap cleanup 0

rm -rf "${WASMTIME_HOME}"
mkdir -p "${WASMTIME_HOME}"
download-and-verify "${WASMTIME_URL}" "${TMP_DIR}/${WASMTIME_TARBALL}" sha256 "${WASMTIME_SHA256}"

tar -xJf "${TMP_DIR}/${WASMTIME_TARBALL}" -C "${WASMTIME_HOME}" --strip-components=1
ln -sf "${WASMTIME_HOME}/wasmtime" /usr/local/bin/wasmtime
