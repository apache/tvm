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

# microTVM Reference Virtual Machines (RVM)

This directory contains Vagrant specifications that create Reference Virtual
Machines (RVM) for use with microTVM. These machines help microTVM users
collaborate by providing a stable reference environment to build and test
microTVM.

For more information on how to use them, see the
[microTVM Reference VM tutorial](../../../tutorials/micro/micro_reference_vm.py).


## microTVM Developer Information

Each RTOS or platform (like Zephyr, Ardunio, etc) that integrates with microTVM
can check-in installation scripts in the Reference VM in this directory to help
the community collaborate. You should use the tools provided here to ensure a
uniform release process across all platforms. Typically, releases need to be
created by TVM committers.

Generally speaking, it's expected that any integrated platform with a regression
test checked-in to the tvm repository should also define a reference VM. If you
want to integrate a new platform, please raise a discussion on
[the forum](https://discuss.tvm.ai).


## Reference VM Organization

The Reference VM is organized in this directory as follows:

```
.
+-- base-box-tool.py - Reference VM build, test, and release tool.
    +-- Vagrantfile  - Vagrantfile that end-users will invoke. Should be based
    |                  off a base box which contains dependencies other than the
    |                  TVM python dependencies.
    +-- base-box/    - Top-level directory which defines the base box.
        +-- Vagrantfile.packer-template - 'packer' template Vagrantfile which
        |                                 will be used to build the base box.
        +-- test-config.json            - JSON file explaining how to perform
                                          release tests to base-box-tool.py.
```


## Creating Releases

1. **Build** the base box for a given platform:
```bash
$ ./base-box-tool.py [--provider=PROVIDER] build
```

For example:
```bash
$ ./base-box-tool.py --provider virtualbox build
```

2. **Run** release tests for each platform:

   A. Connect any needed hardware to the VM host machine;

   B. Run tests:
   ```bash
   $ ./base-box-tool.py [--provider=PROVIDER] test --microtvm-board=MICROTVM_BOARD [--test-device-serial=SERIAL] PLATFORM
   ```
   where MICROTVM_BOARD is one of the options listed in the
   PLATFORM/base-box/test-config.json file.

   For example:
   ```base
   $ ./base-box-tool.py --provider virtualbox test --microtvm-board=stm32f746g_disco zephyr
   ```

   This command does the following for the specified provider:

   * Copies all files inside this dir except `.vagrant` and `base-box` to
   `release-test/`. This is done to avoid reusing any VM the developer may have
   started;

   * Executes `$ vagrant up [--provider=PROVIDER]`;

   * Finds an attached USB device matching the VID and PID specified in
   `test-config.json`, and if `--test-device-serial` was given, that serial
   number (as reported to USB). Creates a rule to autoconnect this device to the
   VM, and also attaches it to the VM;

   * SSHs to the VM, `cd` to the TVM root directory, and runs `test_cmd` from
   `test-config.json`. Nonzero status means failure.

3. If release tests _fail_, fix them and restart from step 1.

4. If release tests pass, **release** the box:
```bash
$ ./base-box-tool.py [--provider=PROVIDER] release --release-version=RELEASE_VER
```
   For that step be sure you've logged in to Vagrant Cloud using the `vagrant`
   tool.

## Versioning
We use semantic versioning as it is recommended by [Vagrant](https://www.vagrantup.com/docs/boxes/versioning). We use `X.Y.Z` version where we maintain the same major version `X` it has minor changes and newer version is still compatible with older versions and we increase minor version `Y`. However, We increase the major version `X` when new RVM is not compatible with older onces. Updates to the Zephyr SDK or Arduino board SDKs are considered major changes and require incrementing major version `X`. In this versioning, `Z` is barely used but we kept it since Vagrant requires this format.

**Note**: We will release all microTVM RVM boxes under [microtvm](https://app.vagrantup.com/tlcpack/boxes/microtvm) and use box versioning in Vagrant file. Previous versions like `microtvm-zephyr`, `microtvm-arduino`, `microtvm-zephyr-2.5`, etc. are deprecated and will be removed in the future.
