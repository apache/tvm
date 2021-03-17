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
"""
Installing TVM
==============
**Authors**:
`Jocelyn Shiue <https://github.com/>`_,
`Chris Hoge <https://github.com/hogepodge>`_

Depending on your needs and your working environment, there are a few different
methods for installing TVM. These include:
    * Installing from source (recommended)
    * Installing from the TLCPack Conda and Pip Packages
"""

################################################################################
# Installing from Source
# ----------------------
# Installing from source is the recommended method for installing TVM. It will
# allow you to enable specific features such as GPU support, microcontroller
# support (uTVM), and a debugging runtime, and other features. You will also
# want to install from source if you want to actively contribute to the TVM
# project. The full instructions are on the `Install TVM From Source
# </install/from_source.html>`_ page.

################################################################################
# Installing from TLC Pack
# ------------------------
# TVM is packaged and distributed as part of the volunteer TLCPack community.
# TLCPack is not affiliated with the Apache Software Foundation, and software
# distributed by them may include non-free hardware drivers. You can lean more
# about TLCPack at their `website <https://tlcpack.ai>`_. TLCPack offers a
# support matrix with instructions for how to install TVM on different
# platforms, with different features. You can choose between the latest stable
# or nightly builds. If your environment isn't supported by TLCPack, you will
# need to install from source.
