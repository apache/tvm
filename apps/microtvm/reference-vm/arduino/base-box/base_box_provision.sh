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

# Install Arduino-CLI (latest version)
export PATH="/home/vagrant/bin:$PATH"
wget -O - https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh -s

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
arduino-cli core install arduino:mbed_nano # Arduino Nano BLE
arduino-cli core install arduino:sam # Arduino Due
arduino-cli core install SPRESENSE:spresense --additional-urls $SPRESENSE_BOARDS_URL # Sony Spresense
arduino-cli core install adafruit:samd --additional-urls $ADAFRUIT_BOARDS_URL # Adafruit PyBadge
arduino-cli core install esp32:esp32 --additional-urls $ESP32_BOARDS_URL # Adafruit FeatherS2

# Cleanup
rm -f *.sh
