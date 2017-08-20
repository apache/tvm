#!/bin/bash
CURR_DIR=$(cd `dirname $0`; pwd)
keytool -genkey -keystore $CURR_DIR/tvmrpc.keystore -alias tvmrpc -keyalg RSA -validity 10000
