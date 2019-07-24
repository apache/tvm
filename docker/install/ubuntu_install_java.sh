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

set -o errexit -o nounset
set -o pipefail

apt-get update && apt-get install -y openjdk-8-jdk maven
arch=$(uname -m)
jre_arch="unknown"
case $arch in
    'x86_64')
	jre_arch="amd64"
	;;
    'aarch64')
	jre_arch="arm64"
	;;
    default)
	echo "Unknown architecture $arch" >&2
	exit 1
        ;;
esac

if [ ! -d "/usr/lib/jvm/java-8-openjdk-$jre_arch/jre" ]; then
  echo "error: missing openjdk for $jre_arch" >&2
  exit 1
fi
echo "export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-$jre_arch/jre" >> /etc/profile
