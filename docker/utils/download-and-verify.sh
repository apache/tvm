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

# Download a file with curl and verify its checksum (sha256 or sha512).
if [[ "$#" -lt 4 || "$#" -gt 5 ]]; then
    echo "Usage: $0 URL OUTPUT_PATH CHECKSUM_ALGO EXPECTED_CHECKSUM [RETRIES]" >&2
    exit 2
fi

url="$1"
output_path="$2"
checksum_algo="$3"
expected_checksum="$4"
retries="${5:-0}"

case "${checksum_algo}" in
    sha256)
        checksum_bin="sha256sum"
        ;;
    sha512)
        checksum_bin="sha512sum"
        ;;
    *)
        echo "Unsupported checksum algorithm: ${checksum_algo}" >&2
        exit 2
        ;;
esac

if [ "${retries}" -gt 0 ]; then
    curl --retry "${retries}" -fsSL "${url}" -o "${output_path}"
else
    curl -fsSL "${url}" -o "${output_path}"
fi

printf "%s  %s\n" "${expected_checksum}" "${output_path}" | "${checksum_bin}" -c -
