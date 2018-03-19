#!/bin/bash
CURR_DIR=$(cd `dirname $0`; pwd)
keytool -genkey -keystore $CURR_DIR/tvmdemo.keystore -alias tvmdemo -keyalg RSA -validity 10000
