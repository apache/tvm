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
.. _tutorial-micro-reference-vm:

===================================
microTVM Reference Virtual Machines
===================================
**Author**: `Andrew Reusch <areusch@octoml.ai>`_

This tutorial explains how to launch microTVM Reference Virtual Machines. You can use these to
develop on real physical hardware without needing to individually install the microTVM
dependencies. These are also particularly useful when trying to reproduce behavior with
microTVM, such as when filing bug reports.

microTVM is the effort to allow TVM to build and execute models on bare-metal microcontrollers.
microTVM aims to be compatible with a wide variety of SoCs and runtime environments (i.e. bare metal,
RTOS, etc). However, some stable software environment is needed to allow developers to share and
reproduce bugs and results. The microTVM Reference Virtual Machines are intended to provide that
environment.

How it works
============

No Virtual Machines are stored in the TVM repository--instead, the files stored in
``apps/microtvm/reference-vm`` describe how to build VMs to the Vagrant_ VM builder tool.

The Reference VMs are split into two parts:

1. A Vagrant Base Box, which contains all of the stable dependencies for that platform. Build
   scripts are stored in ``apps/microtvm/reference-vm/<platform>/base-box``. TVM committers run
   these when a platform's "stable" dependencies change, and the generated base boxes are stored in
   `Vagrant Cloud`_.
2. A per-workspace VM, which users normally build using the Base Box as a starting point. Build
   scripts are stored in ``apps/microtvm/reference-vm/<platform>`` (everything except ``base-box``).

.. _Vagrant: https://vagrantup.com
.. _Vagrant Cloud: https://app.vagrantup.com/tlcpack

Setting up the VM
=================

Installing prerequisites
------------------------

A minimal set of prerequisites are needed:

1. `Vagrant <https://vagrantup.com>`__
2. A supported Virtual Machine hypervisor (**VirtualBox**, **Parallels**, or **VMWare Fusion/Workstation**).
   `VirtualBox <https://www.virtualbox.org>`__ is a suggested free hypervisor, but please note
   that the `VirtualBox Extension Pack`_ is required for proper USB forwarding. If using VirtualBox,
   also consider installing the `vbguest <https://github.com/dotless-de/vagrant-vbguest>`_ plugin.

.. _VirtualBox Extension Pack: https://www.virtualbox.org/wiki/Downloads#VirtualBox6.1.16OracleVMVirtualBoxExtensionPack

3. If required for your hypervisor, the
   `Vagrant provider plugin <https://github.com/hashicorp/vagrant/wiki/Available-Vagrant-Plugins#providers>`__ (or see `here <https://www.vagrantup.com/vmware>`__ for VMWare).

First boot
----------

The first time you use a reference VM, you need to create the box locally and then provision it.

.. code-block:: bash

    # Replace zephyr with the name of a different platform, if you are not using Zephyr.
    ~/.../tvm $ cd apps/microtvm/reference-vm/zephyr
    # Replace <provider_name> with the name of the hypervisor you wish to use (i.e. virtualbox, parallels, vmware_desktop).
    ~/.../tvm/apps/microtvm/reference-vm/zephyr $ vagrant up --provider=<provider_name>


This command will take a couple of minutes to run and will require 4 to 5GB of storage on your
machine. It does the following:

1. Downloads the `microTVM base box`_ and clones it to form a new VM specific to this TVM directory.
2. Mounts your TVM directory (and, if using ``git-subtree``, the original ``.git`` repo) into the
   VM.
3. Builds TVM and installs a Python virtualenv with the dependencies corresponding with your TVM
   build.

.. _microTVM base box: https://app.vagrantup.com/tlcpack/boxes/microtvm

Connect Hardware to the VM
--------------------------

Next, you need to configure USB passthrough to attach your physical development board to the virtual
machine (rather than directly to your laptop's host OS).

It's suggested you setup a device filter, rather than doing a one-time forward, because often the
device may reboot during the programming process and you may, at that time, need to enable
forwarding again. It may not be obvious to the end user when this occurs. Instructions to do that:

 * `VirtualBox <https://www.virtualbox.org/manual/ch03.html#usb-support>`__
 * `Parallels <https://kb.parallels.com/122993>`__
 * `VMWare Workstation <https://docs.vmware.com/en/VMware-Workstation-Pro/15.0/com.vmware.ws.using.doc/GUID-E003456F-EB94-4B53-9082-293D9617CB5A.html>`__

Rebuilding TVM inside the Reference VM
--------------------------------------

After the first boot, you'll need to ensure you keep the build, in ``$TVM_HOME/build-microtvm-zephyr``,
up-to-date when you modify the C++ runtime or checkout a different revision. You can either
re-provision the machine (``vagrant provision`` in the same directory you ran ``vagrant up`` before)
or manually rebuild TVM yourself.

Remember: the TVM ``.so`` built inside the VM is different from the one you may use on your host
machine. This is why it's built inside the special directory ``build-microtvm-zephyr``.

Logging in to the VM
--------------------

The VM should be available to your host only with the hostname ``microtvm``. You can SSH to the VM
as follows:

.. code-block:: bash

    $ vagrant ssh

Then ``cd`` to the same path used on your host machine for TVM. For example, on Mac:

.. code-block:: bash

    $ cd /Users/yourusername/path/to/tvm

Running tests
=============

Once the VM has been provisioned, tests can be executed using ``poetry``:

.. code-block:: bash

    $ cd apps/microtvm/reference-vm/zephyr
    $ poetry run python3 ../../../../tests/micro/zephyr/test_zephyr.py --board=stm32f746g_disco

If you do not have physical hardware attached, but wish to run the tests using the
local QEMU emulator running within the VM, run the following commands instead:

.. code-block:: bash

    $ cd /Users/yourusername/path/to/tvm
    $ cd apps/microtvm/reference-vm/zephyr/
    $ poetry run pytest ../../../../tests/micro/zephyr/test_zephyr.py --board=qemu_x86



"""

# sphinx_gallery_start_ignore
from tvm import testing

testing.utils.install_request_hook(depth=3)
# sphinx_gallery_end_ignore
