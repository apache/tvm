# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# For lint test
# CI docker lint env
FROM ubuntu:16.04

RUN apt-get update && apt-get install -y sudo wget
COPY install/ubuntu_install_python.sh /install/ubuntu_install_python.sh
RUN bash /install/ubuntu_install_python.sh

RUN apt-get install -y doxygen graphviz git
RUN pip3 install cpplint pylint==1.9.4 mypy

# java deps for rat
COPY install/ubuntu_install_java.sh /install/ubuntu_install_java.sh
RUN bash /install/ubuntu_install_java.sh

COPY install/ubuntu_install_rat.sh /install/ubuntu_install_rat.sh
RUN bash /install/ubuntu_install_rat.sh
