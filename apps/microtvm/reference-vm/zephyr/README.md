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

# microTVM Zephyr Reference Virtual Machine

This directory contains setup files for Zephyr virtual machine used for testing microTVM platforms
that are supported by [Zephyr Project](https://zephyrproject.org/).

## VM Information for Developers
Zephyr VM is published under [tlcpack](https://app.vagrantup.com/tlcpack).
Here is a list of different release versions and their tools.

**Note**: We will release all microTVM RVM boxes under [microtvm-zephyr](https://app.vagrantup.com/tlcpack/boxes/microtvm-zephyr) and use box versioning in Vagrant file. Previous versions like `microtvm-zephyr-2.5`, `microtvm-zephyr-2.4` are not continued and will be removed in future.

## Versioning
We use semantic versioning as it is recommended by [Vagrant](https://www.vagrantup.com/docs/boxes/versioning). We use `X.Y.Z` version where we maintain the same major version `X` it has minor changes and newer version is still compatible with older versions and we increase minor version `Y`. However, We increase the major version `X` when new RVM is not compatible with older onces. Updating Zephyr SDK is considered a major change and it requires incrementing major version `X`.
