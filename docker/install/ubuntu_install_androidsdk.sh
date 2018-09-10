. /etc/profile

set -o errexit -o nounset

ANDROID_HOME=/opt/android-sdk-linux
ASDKTOOLS_HOME=/opt/android-sdk-tools
ASDKTOOLS_VERSION=3859397
ASDKTOOLS_SHA256=444e22ce8ca0f67353bda4b85175ed3731cae3ffa695ca18119cbacef1c1bea0

wget http://dl.google.com/android/repository/sdk-tools-linux-${ASDKTOOLS_VERSION}.zip -O sdk-tools-linux.zip
echo "${ASDKTOOLS_SHA256} *sdk-tools-linux.zip" | sha256sum --check -
unzip sdk-tools-linux.zip
rm sdk-tools-linux.zip
mv tools "${ASDKTOOLS_HOME}/"
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
build-tools;27.0.3
cmake;3.6.4111459
emulator
extras;android;gapid;1
extras;android;gapid;3
extras;android;m2repository
extras;google;auto
extras;google;google_play_services
extras;google;instantapps
extras;google;m2repository
extras;google;market_apk_expansion
extras;google;market_licensing
extras;google;simulators
extras;google;webdriver
extras;m2repository;com;android;support;constraint;constraint-layout;1.0.2
extras;m2repository;com;android;support;constraint;constraint-layout-solver;1.0.2
lldb;2.3
platforms;android-26
platforms;android-27
tools
ndk-bundle
EOF

mkdir /root/.android 2>/dev/null || true
touch /root/.android/repositories.cfg
yes | sdkmanager --licenses --sdk_root="$ANDROID_HOME"
sdkmanager --verbose --package_file=/install/package-list-minimal.txt --sdk_root="$ANDROID_HOME"
test -d "${ANDROID_HOME}/build-tools/27.0.3"
test -d "${ANDROID_HOME}/ndk-bundle"
for f in ${ANDROID_HOME}/ndk-bundle/* ; do
  ln --symbolic "$f" "/usr/bin/`basename $f`"
done
echo "export ANDROID_HOME=${ANDROID_HOME}" >> /etc/profile

