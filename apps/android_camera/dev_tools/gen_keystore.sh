#!/bin/bash
CURR_DIR=$(cd `dirname $0`; pwd)
keytool -genkey -keystore $CURR_DIR/tv8mdemo.keystore -alias tv8mdemo -keyalg RSA -validity 10000
