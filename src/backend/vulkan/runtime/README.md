<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->


## Components

### VulkanDeviceAPI

Implements the TVM DeviceAPI interface. Owns the core Vulkan datastructures. Is
responsible for initializing the Vulkan instance and devices, querying for
possible extensions.

### VulkanThreadEntry

Thread-local state for the Vulkan runtime. Maintains a staging buffer (for
copies), and a VulkanStream per device.

### VulkanWrappedFunc

Responsible for launching computation kernels. Responsible for obtaining a
VulkanPipeline instance (from the VulkanModuleNode), and launches the kernel
(via immediate or deferred mode) on the active VulkanStream instance.

## Stream execution in the Vulkan programming model.

The natural model for TVM DeviceAPI implementation and runtime follows the CUDA
API model. That is, we launch "kernels" onto a (implicit or explicit) "stream"
(which execute asynchronously with respect to the host, but ordered with respect
to the stream), and explicitly synchronize the stream with respect to the host.
We simulate this behaviour in the Vulkan model by maintaining a thread-local
`vkCommandBuffer` instance, and queueing up (or eagerly executing, depending on
the availability of the `VK_KHR_push_descriptor` extension). When we synchronize
the stream, we end the command buffer recording, submit it to the device queue,
and wait on the corresponding fence.
