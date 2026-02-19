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

WORK_DIR="$(mktemp -d)"
cleanup() {
	rm -rf "${WORK_DIR}"
}
trap cleanup 0

cd "${WORK_DIR}"
RAT_VERSION="0.17"
RAT_ARCHIVE="apache-rat-${RAT_VERSION}-bin.tar.gz"
RAT_BASE_URL="https://downloads.apache.org/creadur/apache-rat-${RAT_VERSION}"
RAT_SHA512="32848673dc4fb639c33ad85172dfa9d7a4441a0144e407771c9f7eb6a9a0b7a9b557b9722af968500fae84a6e60775449d538e36e342f786f20945b1645294a0"

download-and-verify "${RAT_BASE_URL}/${RAT_ARCHIVE}" "${RAT_ARCHIVE}" sha512 "${RAT_SHA512}"

tar xf "${RAT_ARCHIVE}"
mv "apache-rat-${RAT_VERSION}/apache-rat-${RAT_VERSION}.jar" /bin/apache-rat.jar
