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

# microTVM Reference Virtual Machines

This directory contains Vagrant specifications that create reference Virtual Machines for use with
microTVM. These machines help microTVM users collaborate by providing a stable reference test
environment.

For more information on how to use them, see the microTVM Reference Virtual Machines tutorial.


## Reference VM Developer Information

Each RTOS or platform that integrates with microTVM can check-in a Reference VM in this directory to
help the community collaborate. You should use the tools provided here to ensure a uniform release
process across all platforms. Typically, releases need to be created by TVM committers.

Generally speaking, it's expected that any integrated platform with a regression test checked-in to
the tvm repository should also define a reference VM. If you want to integrate a new platform,
please raise a discussion on [the forum](https://discuss.tvm.ai).

### Organization

Reference VMs are organized as follows:

* `base-box-tool.py` - Reference VM build, test, and release tool
* `<platform>/`
** `Vagrantfile` Vagrantfile that end-users will inovke. Should be based off a base box
    which contains dependencies other than the TVM python dependencies.
** `base-box` - Top-level directory which defines the base box.
*** `Vagrantfile.packer-template` - Packer template Vagrantfile which will be used to build the
    base box.
*** `test-config.json` - JSON file explaining how to perform release tests to `base-box-tool.py`

## Creating Releases

1. Build the base box for the given platform: `$ ./base-box-tool.py [--provider=<provider>] build <platform>`
2. Run release tests for each platform:
    1. Connect any needed hardware to the VM host machine.
    2. Run tests: `$ ./base-box-tool.py [--provider=<provider>] test [--microtvm-platform=<platform>] <platform> [--test-device-serial=<serial>]`. This
       command does the following for each provider:
        1. Copies all files inside `./<platform>` except `.vagrant` and `base-box` to
           `./release-test`. This is done to avoid reusing any VM the developer may have started.
        2. Executes `$ vagrant up [--provider=<provider>]`.
        3. Finds an attached USB device matching the VID and PID specified in `test-config.json`,
           and if `--test-device-serial` was given, that serial number (as reported to USB). Creates
           a rule to autoconnect this device to the VM, and also attaches it to the VM>
        4. SSHs to the VM, `cd` to the TVM root directory, and runs `test_cmd` from
           `test-config.json`. Nonzero status means failure.
3. If release tests fail, fix them and restart from step 1.
4. If release tests pass: `$ ./base-box-tool.py [--provider=<provider>] release <--release-version=<version>> <--platform-version=<version>> <platform>`. Be sure you've logged
   in to Vagrant Cloud using the `vagrant` tool.
