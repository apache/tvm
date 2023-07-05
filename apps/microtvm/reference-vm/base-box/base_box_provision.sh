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
#   virtual machine similar to CI Cortex-M setup.
#

set -x

source ~/.profile

# Init Zephyr
cd ~
~/ubuntu_init_zephyr_project.sh ~/zephyr

# Install CMSIS
cd ~
~/ubuntu_install_cmsis.sh ~/cmsis

# Cleanup
rm -f ubuntu_init_zephyr_project.sh ubuntu_install_cmsis.sh

# Init Arduino
source ~/.profile
cd ~

sudo apt-get install -y ca-certificates

# Install Arduino-CLI (specific version)
# To keep in sync with the version
# defined in apps/microtvm/arduino/template_project/microtvm_api_server.py
ARDUINO_CLI_VERSION="0.21.1"

export PATH="/home/vagrant/bin:$PATH"
wget -O - https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh -s ${ARDUINO_CLI_VERSION}

# Arduino (the CLI and GUI) require the dialout permission for uploading
sudo usermod -a -G dialout $USER

# ubuntu_init_arduino.sh only installs a few officially
# supported architectures, so we don't use it here

# 3rd party board URLs
ADAFRUIT_BOARDS_URL="https://raw.githubusercontent.com/adafruit/arduino-board-index/7840c768/package_adafruit_index.json"
ESP32_BOARDS_URL="https://github.com/espressif/arduino-esp32/releases/download/2.0.3/package_esp32_dev_index.json"
RP2040_BOARDS_URL="https://github.com/earlephilhower/arduino-pico/releases/download/2.0.3/package_rp2040_index.json"
SPRESENSE_BOARDS_URL="https://github.com/sonydevworld/spresense-arduino-compatible/releases/download/v2.5.0/package_spresense_index.json"
arduino-cli core update-index --additional-urls $ADAFRUIT_BOARDS_URL,$ESP32_BOARDS_URL,$RP2040_BOARDS_URL,$SPRESENSE_BOARDS_URL

# Install supported cores from those URLS
arduino-cli version
arduino-cli core install arduino:mbed_nano@3.0.1
arduino-cli core install arduino:sam@1.6.12
arduino-cli core install arduino:mbed_portenta@3.1.1
arduino-cli core install adafruit:samd@1.7.10 --additional-urls $ADAFRUIT_BOARDS_URL
arduino-cli core install esp32:esp32@2.0.2 --additional-urls $ESP32_BOARDS_URL
arduino-cli core install rp2040:rp2040@2.0.3 --additional-urls $RP2040_BOARDS_URL
arduino-cli core install SPRESENSE:spresense@2.5.0 --additional-urls $SPRESENSE_BOARDS_URL

# The Arduino Code API has a major bug that breaks TVM. It has been worked around in
# most board SDKs (including arduino:sam), but it still exists for the Portenta H7.
# There is a PR to fix it (https://github.com/arduino/ArduinoCore-API/pull/163), but
# it may not be merged for a while (and a new release will have to be deployed too).
# The below sed command avoids the bug, and will be removed when no longer needed.
PORTENTA_H7_BUGFIX_PATH=~/.arduino15/packages/arduino/hardware/mbed_portenta/3.1.1/cores/arduino/api/Common.h
sed -i '3 i #include <stdbool.h>' $PORTENTA_H7_BUGFIX_PATH
