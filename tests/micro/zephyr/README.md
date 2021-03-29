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

This directory contains tests for MicroTVM's integration with Zephyr.

To run the test, you first need to be running in a Python environment with
all of the appropriate TVM dependencies installed. If you have [Poetry](https://python-poetry.org/)
installed, you can do the following to get an appropriately-configured Python
environment:

```
$ cd tvm/apps/microtvm/
$ poetry lock && poetry install && poetry shell
```

You can then run this test (either on real hardware or on a QEMU-emulated
device) using:

```
$ cd tvm/tests/micro/zephyr
$ pytest test_zephyr.py --microtvm-platforms=host       # For QEMU emulation
$ pytest test_zephyr.py --microtvm-platforms=nrf5340dk  # For nRF5340DK
```

To see the list of supported values for `--microtvm-platforms`, run:
```
$ pytest test_zephyr.py --help
```
