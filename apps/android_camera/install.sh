#!/bin/bash
./dev_tools/sign_apk.sh && adb install -r ./app/build/outputs/apk/release/tv8mdemo-release.apk
