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

. /etc/profile

set -o errexit -o nounset
set -o pipefail

ANDROID_HOME=/opt/android-sdk-linux
ASDKTOOLS_HOME=/opt/android-sdk-tools
ASDKTOOLS_VERSION=3859397
ASDKTOOLS_SHA256=444e22ce8ca0f67353bda4b85175ed3731cae3ffa695ca18119cbacef1c1bea0
COMMANDLINETOOLS_VERSION=11076708
COMMANDLINETOOLS_SHA256=2d2d50857e4eb553af5a6dc3ad507a17adf43d115264b1afc116f95c92e5e258

ANDROID_NDK_VERSION=21.3.6528147
CMAKE_VERSION=3.6.4111459
BUILD_TOOLS_VERSION=27.0.3
ANDROID_PLATFORM=27
ANDROID_NDK_MAJOR=21

if [ $# -gt 0 ] ; then
  ANDROID_NDK_VERSION=$1
  ANDROID_NDK_MAJOR=`echo $1 | cut -d "." -f1`
fi
if [ $# -gt 1 ] ; then
  CMAKE_VERSION=$2
fi
if [ $# -gt 2 ] ; then
  BUILD_TOOLS_VERSION=$3
fi
if [ $# -gt 3 ] ; then
  ANDROID_PLATFORM=$4
fi

echo "NDK Version: ${ANDROID_NDK_VERSION}"
echo "NDK Major  : ${ANDROID_NDK_MAJOR}"
echo "Cmake Version: ${CMAKE_VERSION}"
echo "Build Tools: ${BUILD_TOOLS_VERSION}"
echo "Android Platform: ${ANDROID_PLATFORM}"

wget -q https://dl.google.com/android/repository/commandlinetools-linux-${COMMANDLINETOOLS_VERSION}_latest.zip  -O commandlinetools-linux.zip
echo "${COMMANDLINETOOLS_SHA256} commandlinetools-linux.zip" | sha256sum --check -
unzip commandlinetools-linux.zip
rm commandlinetools-linux.zip
mv cmdline-tools/ "${ASDKTOOLS_HOME}/"
# The following popular fix makes sdkmanager honour $http_proxy variables
mv ${ASDKTOOLS_HOME}/bin/sdkmanager ${ASDKTOOLS_HOME}/bin/sdkmanager-vanilla
cat >${ASDKTOOLS_HOME}/bin/sdkmanager <<"EOF"
#!/bin/sh
if test -n "$http_proxy"; then
  PROXY_HOST=`echo $http_proxy | sed 's@.*//\(.*\):.*@\1@'`
  PROXY_PORT=`echo $http_proxy | sed 's@.*//.*:\(.*\)@\1@'`
  PROXY="--proxy=http --proxy_host=$PROXY_HOST --proxy_port=$PROXY_PORT"
else
  PROXY=""
fi
exec "`dirname $0`/sdkmanager-vanilla" $PROXY "$@"
EOF
for f in ${ASDKTOOLS_HOME}/bin/* ; do
  chmod +x "$f"
  ln --symbolic "$f" "/usr/bin/`basename $f`"
done


cat >/install/package-list-minimal.txt <<EOF
build-tools;26.0.3
build-tools;${BUILD_TOOLS_VERSION}
cmake;${CMAKE_VERSION}
emulator
extras;android;m2repository
extras;google;auto
extras;google;google_play_services
extras;google;instantapps
extras;google;m2repository
extras;google;market_apk_expansion
extras;google;market_licensing
extras;google;simulators
extras;google;webdriver
platforms;android-26
platforms;android-${ANDROID_PLATFORM}
tools
ndk;${ANDROID_NDK_VERSION}
EOF

mkdir /root/.android 2>/dev/null || true
touch /root/.android/repositories.cfg
# NOTE: sdkmanager returns exit code 141
(yes || true) | sdkmanager --licenses --sdk_root="$ANDROID_HOME" || true
sdkmanager --verbose --package_file=/install/package-list-minimal.txt --sdk_root="$ANDROID_HOME"
test -d "${ANDROID_HOME}/build-tools/${BUILD_TOOLS_VERSION}"
test -d "${ANDROID_HOME}/ndk/${ANDROID_NDK_VERSION}"
for f in ${ANDROID_HOME}/ndk/${ANDROID_NDK_VERSION}/* ; do
  ln --symbolic "$f" "/usr/bin/`basename $f`"
done
echo "export ANDROID_HOME=${ANDROID_HOME}" >> /etc/profile
echo "export ANDROID_NDK_HOME=/opt/android-sdk-linux/ndk/${ANDROID_NDK_VERSION}/" >> /etc/profile
echo "export ANDROID_NDK_VERSION=${ANDROID_NDK_VERSION}" >> /etc/profile
echo "export ANDROID_NDK_MAJOR=${ANDROID_NDK_MAJOR}" >> /etc/profile
