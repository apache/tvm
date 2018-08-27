#!/bin/bash
CURR_DIR=$(cd `dirname $0`; pwd)
APK_DIR=$CURR_DIR/../app/build/outputs/apk/release
UNSIGNED_APK=$APK_DIR/app-release-unsigned.apk
SIGNED_APK=$APK_DIR/tvmrpc-release.apk
jarsigner -verbose -keystore $CURR_DIR/tvmrpc.keystore -signedjar $SIGNED_APK $UNSIGNED_APK 'tvmrpc'
echo $SIGNED_APK
