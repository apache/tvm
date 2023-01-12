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

..  Boilerplate script for installing CMSIS-NN in the microTVM
    tutorials that use it. Does not show up as a separate file
    on the documentation website.

Install CMSIS-NN
----------------------------

    .. code-block:: bash

        %%shell
        CMSIS_SHA="51263182d16c92649a48144ba56c0945f9fce60e"
        CMSIS_URL="http://github.com/ARM-software/CMSIS_5/archive/${CMSIS_SHA}.tar.gz"
        export CMSIS_PATH=/content/cmsis
        DOWNLOAD_PATH="/content/${CMSIS_SHA}.tar.gz"
        mkdir ${CMSIS_PATH}
        wget ${CMSIS_URL} -O "${DOWNLOAD_PATH}"
        tar -xf "${DOWNLOAD_PATH}" -C ${CMSIS_PATH} --strip-components=1
        rm ${DOWNLOAD_PATH}

        CMSIS_NN_TAG="v4.0.0"
        CMSIS_NN_URL="https://github.com/ARM-software/CMSIS-NN.git"
        git clone ${CMSIS_NN_URL} --branch ${CMSIS_NN_TAG} --single-branch ${CMSIS_PATH}/CMSIS-NN
