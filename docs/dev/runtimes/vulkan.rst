..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

.. _tvm-runtime-vulkan:

Vulkan Runtime
==============

TVM supports using Vulkan compute shaders to execute queries.  Each
computational kernel is compiled into a SPIR-V shader, which can then
be called using the TVM interface.

.. _tvm-runtime-vulkan-features:

Vulkan Features, Limits
-----------------------

.. _Required Limits: https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#limits-minmax

Since different Vulkan implementations may enable different optional
features or have different physical limits, the code generation must
know which features are available to use.  These correspond to
specific Vulkan capabilities/limits as in
:ref:`Vulkan Capabilities Table <tvm-table-vulkan-capabilities>`.
If unspecified, TVM assumes that a capability is not available, or
that a limit is the minimum guaranteed by the Vulkan spec in the
`Required Limits`_ section.

These parameters can be either explicitly specific when defining a
:ref:`Target <tvm-target-specific-target>`, or can be queried from a
device.  To query from a device, the special parameter
``-from_device=N`` can be used to query all vulkan device parameters
from device id ``N``.  Any additional parameters explicitly specified
will override the parameters queried from the device.

.. _VkSubgroupFeatureFlagBits: https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkSubgroupFeatureFlagBits.html

.. list-table:: Vulkan Capabilities
   :name: tvm-runtime-table-vulkan-capabilities
   :header-rows: 1

   * - Target Parameter
     - Required Vulkan Version/Extension
     - Parameter Queried
     - Default Value

   * - ``supported_subgroup_operations``
     - Vulkan 1.1+
     - ``VkPhysicalDeviceSubgroupProperties::supportedOperations``
     - 0 (interpreted as `VkSubgroupFeatureFlagBits`_)

   * - ``max_push_constants_size``
     -
     - ``VkPhysicalDeviceLimits::maxPushConstantsSize``
     - 128 bytes

   * - ``max_uniform_buffer_range``
     -
     - ``VkPhysicalDeviceLimits::maxUniformBufferRange``
     - 16384 bytes


   * - ``max_storage_buffer_range``
     -
     - ``VkPhysicalDeviceLimits::maxStorageBufferRange``
     - 2\ :sup:`27`\ bytes


   * - ``max_per_stage_descriptor_storage_buffer``
     -
     - ``VkPhysicalDeviceLimits::maxPerStageDescriptorStorageBuffers``
     - 4


   * - ``supports_storage_buffer_storage_class``
     - VK_KHR_storage_buffer_storage_class
     -
     - false


   * - ``supports_storage_buffer_8bit_access``
     - VK_KHR_8bit_storage
     - ``VkPhysicalDevice8BitStorageFeaturesKHR::storageBuffer8BitAccess``
     - false


   * - ``supports_storage_buffer_16bit_access``
     - VK_KHR_16bit_storage
     - ``VkPhysicalDevice16BitStorageFeaturesKHR::storageBuffer16BitAccess``
     - false


   * - ``supports_float16``
     - VK_KHR_shader_float16_int8
     - ``VkPhysicalDeviceShaderFloat16Int8FeaturesKHR::shaderFloat16``
     - false


   * - ``supports_float64``
     -
     - ``VkPhysicalDeviceFeatures::shaderFloat64``
     - false


   * - ``supports_int8``
     - VK_KHR_shader_float16_int8
     - ``VkPhysicalDeviceShaderFloat16Int8FeaturesKHR::shaderInt8``
     - false


   * - ``supports_int16``
     -
     - ``VkPhysicalDeviceFeatures::shaderInt16``
     - false


   * - ``supports_int64``
     -
     - ``VkPhysicalDeviceFeatures::shaderInt64``
     - false



As of May 2021, not all Vulkan implementations are supported.  For
example, support for 64-bit integers is required.  If a Vulkan target
is not supported, an error message should be issued during SPIR-V code
generation.  Efforts are also underway to remove these requirements
and support additional Vulkan implementations.


.. _tvm-runtime-vulkan-spirv-capabilities:

SPIR-V Capabilities
-------------------

Some of the device-specific capabilities also correspond to SPIR-V
capabilities or extensions that must be declared in the shader, or a
minimum SPIR-V version required in order to use a feature.  The
TVM-generated shaders will declare the minimum set of
extensions/capabilities and the minimum allowed version of SPIR-V
that are needed to execute the compiled graph.

If the shader generation requires a capability or extension that is
not enabled in the ``Target``, an exception will be raised.


.. list-table:: Vulkan Capabilities
   :name: tvm-table-vulkan-capabilities
   :header-rows: 1

   * - Target Parameter
     - Required SPIR-V Version/Extension
     - Declared Capability

   * - ``supported_subgroup_operations``
     - SPIR-V 1.3+
     - Varies, see `VkSubgroupFeatureFlagBits`_

   * - ``supports_storage_buffer_storage_class``
     - SPV_KHR_storage_buffer_storage_class
     -

   * - ``supports_storage_buffer_8bit_access``
     - SPV_KHR_8bit_storage
     - StorageBuffer8BitAccess

   * - ``supports_storage_buffer_16bit_access``
     - SPV_KHR_16bit_storage
     - StorageBuffer16BitAccess

   * - ``supports_float16``
     -
     - Float16


   * - ``supports_float64``
     -
     - Float64


   * - ``supports_int8``
     -
     - Int8


   * - ``supports_int16``
     -
     - Int16


   * - ``supports_int64``
     -
     - Int64
