%%shell
# Install west and ninja
python3 -m pip install west
apt-get install -y ninja-build

# Install ZephyrProject
ZEPHYR_PROJECT_PATH="/root/zephyrproject"
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
ZEPHYR_SDK_FILE="/root/zephyr-sdk-linux-setup.run"
wget --no-verbose -O $ZEPHYR_SDK_FILE \
    https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v${ZEPHYR_SDK_VERSION}/zephyr-sdk-${ZEPHYR_SDK_VERSION}-linux-x86_64-setup.run
chmod +x $ZEPHYR_SDK_FILE
"$ZEPHYR_SDK_FILE" -- -d /root/zephyr-sdk

# Install TVM and other python dependencies
cd /content
wget https://github.com/guberti/tvm/raw/hackathon/tutorials/apps/microtvm/requirements.txt
python3 -m pip install -r requirements.txt
python3 -m pip install -r "${ZEPHYR_BASE}/scripts/requirements.txt"
