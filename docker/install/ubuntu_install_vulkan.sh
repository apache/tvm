#/bin/bash

wget https://sdk.lunarg.com/sdk/download/1.0.65.0/linux/vulkansdk-linux-x86_64-1.0.65.0.run

bash vulkansdk-linux-x86_64-1.0.65.0.run
mv VulkanSDK /usr/local/VulkanSDK
cd /usr/local/VulkanSDK/1.0.65.0
./build_tools.sh
./build_samples.sh
