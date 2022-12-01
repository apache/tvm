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

* Installing from source
* Installing from third-party binary package.
"""

################################################################################
# Installing From Source
# ----------------------
# Installing from source is the recommended method for installing TVM. It will
# allow you to enable specific features such as GPU support, microcontroller
# support (microTVM), and a debugging runtime, and other features. You will also
# want to install from source if you want to actively contribute to the TVM
# project. The full instructions are on the :ref:`Install TVM From Source
# <install-from-source>` page.

################################################################################
# Installing From Binary Packages
# --------------------------------
# You may install convenient third party binary package distributions to
# quickly try things out. TLCPack is a third party volunteer community that
# builds binary packages from TVM source. It offers a support matrix with
# instructions to install on different platforms, with different features.
# Check out  `TLCPack <https://tlcpack.ai>`_ to learn more. Note that the
# third party binary packages could contain additional licensing terms for
# the hardware drivers that are bundled with it.

# sphinx_gallery_start_ignore
from tvm import testing

testing.utils.install_request_hook(depth=3)
# sphinx_gallery_end_ignore
