#!/bin/bash
set -e
set -u
set -o pipefail

apt-get install -y --no-install-recommends openssh-server

echo "
LogLevel DEBUG2
PermitRootLogin yes
PasswordAuthentication yes
Subsystem sftp /usr/lib/openssh/sftp-server
" > /etc/ssh/sshd_config

mkdir /run/sshd

echo 'root:123456' | chpasswd