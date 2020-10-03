#!/bin/bash -e

cp ~vagrant/setup-workspace.sh /home/tvm/setup-workspace.sh
chown tvm:tvm /home/tvm/setup-workspace.sh
chmod u+x /home/tvm/setup-workspace.sh
sudo -u tvm -sH bash --login ~tvm/setup-workspace.sh "${TVM_HOME}"
