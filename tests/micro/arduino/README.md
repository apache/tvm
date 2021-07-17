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

This directory contains tests for MicroTVM's integration with Arduino.

To run the test, you first need to be running in a Python environment with
all of the appropriate TVM dependencies installed. You can run the test with:

```
$ cd tvm/tests/micro/arduino
$ pytest test_arduino_workflow.py --platform spresense
$ pytest test_arduino_workflow.py --platform nano33ble
```

By default, only project generation and compilation tests are run. If you
have compatible Arduino hardware connected, you can pass the flag
`--run-hardware-tests` to test board auto-detection and code execution:

```
pytest test_arduino_workflow.py --platform spresense --run-hardware-tests
```

To see the list of supported values for `--platform`, run:
```
$ pytest test_arduino_workflow.py --help
```
