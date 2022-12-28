..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

..  Boilerplate script for installing Zephyr in the microTVM
    tutorials that use it. Does not show up as a separate file
    on the documentation website.

Install Zephyr
----------------------------

    .. code-block:: bash

        %%shell
        # Install west and ninja
        python3 -m pip install west
        apt-get install -y ninja-build

        # Install ZephyrProject
        ZEPHYR_PROJECT_PATH="/content/zephyrproject"
        export ZEPHYR_BASE=${ZEPHYR_PROJECT_PATH}/zephyr
        west init ${ZEPHYR_PROJECT_PATH}
        cd ${ZEPHYR_BASE}
        git checkout v2.7-branch
        cd ..
        west update
        west zephyr-export
        chmod -R o+w ${ZEPHYR_PROJECT_PATH}

        # Install Zephyr SDK
        ZEPHYR_SDK_VERSION=0.13.2
        ZEPHYR_SDK_FILE="/content/zephyr-sdk-linux-setup.run"
        wget --no-verbose -O $ZEPHYR_SDK_FILE \
            https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v${ZEPHYR_SDK_VERSION}/zephyr-sdk-${ZEPHYR_SDK_VERSION}-linux-x86_64-setup.run
        chmod +x $ZEPHYR_SDK_FILE
        "$ZEPHYR_SDK_FILE" -- -d /content/zephyr-sdk --quiet

        # Install python dependencies
        python3 -m pip install -r "${ZEPHYR_BASE}/scripts/requirements.txt"
