#!/bin/bash -e

# Zephyr
pip3 install --user -U west
echo 'export PATH=$HOME/.local/bin:"$PATH"' >> ~/.profile
source ~/.profile
echo PATH=$PATH
west init --mr v2.3.0 ~/zephyr
cd ~/zephyr
west update
west zephyr-export

cd ~
echo "Downloading zephyr SDK..."
wget --no-verbose https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.11.3/zephyr-sdk-0.11.3-setup.run
chmod +x zephyr-sdk-0.11.3-setup.run
./zephyr-sdk-0.11.3-setup.run -- -d ~/zephyr-sdk -y

# Poetry
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3
sed -i "/^# If not running interactively,/ i source \$HOME/.poetry/env" ~/.bashrc
sed -i "/^# If not running interactively,/ i export ZEPHYR_BASE=$HOME/zephyr/zephyr" ~/.bashrc
sed -i "/^# If not running interactively,/ i\\ " ~/.bashrc
