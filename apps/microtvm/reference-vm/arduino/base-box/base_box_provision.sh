#!/bin/bash -e
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
#
#   Using this script we can reuse docker/install scripts to configure the reference 
#   virtual machine similar to CI QEMU setup.
#

set -e
set -x

source ~/.profile

# Init Arduino
cd ~

sudo apt-get install -y ca-certificates

# Install Arduino-CLI (specific version)
export PATH="/home/vagrant/bin:$PATH"
wget -O - https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh -s 0.18.3

# Arduino (the CLI and GUI) require the dialout permission for uploading
sudo usermod -a -G dialout $USER

# ubuntu_init_arduino.sh only installs a few officially
# supported architectures, so we don't use it here

# 3rd party board URLs
ADAFRUIT_BOARDS_URL="https://adafruit.github.io/arduino-board-index/package_adafruit_index.json"
ESP32_BOARDS_URL="https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_dev_index.json"
SPARKFUN_BOARDS_URL="https://raw.githubusercontent.com/sparkfun/Arduino_Boards/master/IDE_Board_Manager/package_sparkfun_index.json"
SEEED_BOARDS_URL="https://files.seeedstudio.com/arduino/package_seeeduino_boards_index.json"
SPRESENSE_BOARDS_URL="https://github.com/sonydevworld/spresense-arduino-compatible/releases/download/generic/package_spresense_index.json"
arduino-cli core update-index --additional-urls $ADAFRUIT_BOARDS_URL,$ESP32_BOARDS_URL,$SPARKFUN_BOARDS_URL,$SEEED_BOARDS_URL,$SPRESENSE_BOARDS_URL

# Install supported cores from those URLS
arduino-cli version
arduino-cli core install arduino:mbed_nano
arduino-cli core install arduino:sam
arduino-cli core install adafruit:samd --additional-urls $ADAFRUIT_BOARDS_URL
arduino-cli core install esp32:esp32 --additional-urls $ESP32_BOARDS_URL
arduino-cli core install Seeeduino:samd --additional-urls $SEEED_BOARDS_URL
arduino-cli core install SPRESENSE:spresense --additional-urls $SPRESENSE_BOARDS_URL

# The Sony Spresense SDK has a major bug that breaks TVM. It's scheduled to be fixed in
# release 2.3.0, but until that's published we need to use the below hack. This ONLY
# fixes the bug in the main core release SDK - the subcore release SDK and both
# the main and subcore debug SDKs will continue to fail until an official fix is made.
# https://github.com/sonydevworld/spresense/issues/200
SPRESENSE_NUTTX_BUGFIX_PATH=~/.arduino15/packages/SPRESENSE/tools/spresense-sdk/2.2.1/spresense/release/nuttx/include/sys/types.h
sed -i 's/#ifndef CONFIG_WCHAR_BUILTIN/#if !defined(__cplusplus)/g' $SPRESENSE_NUTTX_BUGFIX_PATH

# There's also a bug in arduino-cli where {runtime.os} is not properly templated in
# platform.txt. This bug only seems to appear with the SPRESENSE SDK. A fix has been
# merged and will be part of arduino-cli 0.18.4, but that has yet to be published.
# This change is only needed to upload code (not compile) for the Spresense.
# https://github.com/arduino/arduino-cli/issues/1198
SPRESENSE_FLASH_WRITER_BUGFIX_PATH=~/.arduino15/packages/SPRESENSE/hardware/spresense/2.2.1/platform.txt
sed -i 's/tools.spresense-tools.cmd.path={path}\/flash_writer\/{runtime.os}\/flash_writer/tools.spresense-tools.cmd.path={path}\/flash_writer\/linux\/flash_writer/g' $SPRESENSE_FLASH_WRITER_BUGFIX_PATH
sed -i 's/tools.spresense-tools.cmd.path.linux={path}\/flash_writer\/{runtime.os}\/flash_writer/tools.spresense-tools.cmd.path.linux={path}\/flash_writer\/linux\/flash_writer/g' $SPRESENSE_FLASH_WRITER_BUGFIX_PATH

# Cleanup
rm -f *.sh
