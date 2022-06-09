<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# microTVM Arduino Reference Virtual Machine

This directory contains setup files for Arduino virtual machine used for testing
microTVM platforms that are supported by [Arduino](https://www.arduino.cc/).

## VM Information for Developers
Arduino VM is published under [tlcpack](https://app.vagrantup.com/tlcpack).
Here is a list of different release versions and their tools.

(none currently)

## Supported Arduino Boards
This RVM has been tested and is known to work with these boards:
- Adafruit Metro M4
- Adafruit Pybadge
- Arduino Due
- Arduino Nano 33 BLE
- Arduino Portenta H7
- Feather S2
- Sony Spresense
- Wio Terminal

However, the RVM *should* work with any Arduino with sufficient memory, provided
its core is installed in `base-box/base_box_provision.sh`.

Note that this RVM does not work with the Teensy boards, even though they are
supported by microTVM. This is because arduino-cli does not support Teensy
boards (https://github.com/arduino/arduino-cli/issues/700)/).
