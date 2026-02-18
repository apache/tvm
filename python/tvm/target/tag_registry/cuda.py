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
"""NVIDIA CUDA target tags."""
from .registry import register_tag


def _register_cuda_tag(name, arch, shared_mem=49152, regs=65536, **extra):
    config = {
        "kind": "cuda",
        "keys": ["cuda", "gpu"],
        "arch": arch,
        "max_shared_memory_per_block": shared_mem,
        "max_threads_per_block": 1024,
        "thread_warp_size": 32,
        "registers_per_block": regs,
    }
    config.update(extra)
    register_tag(name, config)


def _register_jetson_tag(name, arch, mcpu, num_cores, regs=65536):
    register_tag(
        name,
        {
            "kind": "cuda",
            "arch": arch,
            "max_shared_memory_per_block": 49152,
            "max_threads_per_block": 1024,
            "thread_warp_size": 32,
            "registers_per_block": regs,
            "host": {
                "kind": "llvm",
                "mtriple": "aarch64-linux-gnu",
                "mcpu": mcpu,
                "num-cores": num_cores,
            },
        },
    )


# =====================================================================
# Data center / Tesla GPUs
# =====================================================================
_register_cuda_tag("nvidia/nvidia-a100", "sm_80", l2_cache_size_bytes=41943040)
_register_cuda_tag("nvidia/nvidia-h100", "sm_90a", l2_cache_size_bytes=52428800)
_register_cuda_tag("nvidia/nvidia-b100", "sm_100a", l2_cache_size_bytes=52428800)
_register_cuda_tag("nvidia/nvidia-a40", "sm_86")
_register_cuda_tag("nvidia/nvidia-a30", "sm_80")
_register_cuda_tag("nvidia/nvidia-a10", "sm_86")
_register_cuda_tag("nvidia/nvidia-a10g", "sm_86")
_register_cuda_tag("nvidia/nvidia-a16", "sm_86")
_register_cuda_tag("nvidia/nvidia-a2", "sm_86")
_register_cuda_tag("nvidia/nvidia-t4", "sm_75")
_register_cuda_tag("nvidia/nvidia-v100", "sm_70")
_register_cuda_tag("nvidia/tesla-p100", "sm_60")
_register_cuda_tag("nvidia/tesla-p40", "sm_61")
_register_cuda_tag("nvidia/tesla-p4", "sm_61")
_register_cuda_tag("nvidia/tesla-m60", "sm_52")
_register_cuda_tag("nvidia/tesla-m40", "sm_52")
_register_cuda_tag("nvidia/tesla-k80", "sm_37")
_register_cuda_tag("nvidia/tesla-k40", "sm_35")
_register_cuda_tag("nvidia/tesla-k20", "sm_35")
_register_cuda_tag("nvidia/tesla-k10", "sm_30")
_register_cuda_tag("nvidia/tesla-c2075", "sm_20", regs=32768)
_register_cuda_tag("nvidia/tesla-c2050", "sm_20", regs=32768)
_register_cuda_tag("nvidia/tesla-c2070", "sm_20", regs=32768)

# =====================================================================
# Quadro / RTX professional desktop GPUs
# =====================================================================
_register_cuda_tag("nvidia/rtx-a6000", "sm_86")
_register_cuda_tag("nvidia/quadro-rtx-8000", "sm_75")
_register_cuda_tag("nvidia/quadro-rtx-6000", "sm_75")
_register_cuda_tag("nvidia/quadro-rtx-5000", "sm_75")
_register_cuda_tag("nvidia/quadro-rtx-4000", "sm_75")
_register_cuda_tag("nvidia/quadro-gv100", "sm_70")
_register_cuda_tag("nvidia/quadro-gp100", "sm_60")
_register_cuda_tag("nvidia/quadro-p6000", "sm_61")
_register_cuda_tag("nvidia/quadro-p5000", "sm_61")
_register_cuda_tag("nvidia/quadro-p4000", "sm_61")
_register_cuda_tag("nvidia/quadro-p2200", "sm_61")
_register_cuda_tag("nvidia/quadro-p2000", "sm_61")
_register_cuda_tag("nvidia/quadro-p1000", "sm_61")
_register_cuda_tag("nvidia/quadro-p620", "sm_61")
_register_cuda_tag("nvidia/quadro-p600", "sm_61")
_register_cuda_tag("nvidia/quadro-p400", "sm_61")
_register_cuda_tag("nvidia/quadro-m6000-24gb", "sm_52")
_register_cuda_tag("nvidia/quadro-m6000", "sm_52")
_register_cuda_tag("nvidia/quadro-k6000", "sm_35")
_register_cuda_tag("nvidia/quadro-m5000", "sm_52")
_register_cuda_tag("nvidia/quadro-k5200", "sm_35")
_register_cuda_tag("nvidia/quadro-k5000", "sm_30")
_register_cuda_tag("nvidia/quadro-m4000", "sm_52")
_register_cuda_tag("nvidia/quadro-k4200", "sm_30")
_register_cuda_tag("nvidia/quadro-k4000", "sm_30")
_register_cuda_tag("nvidia/quadro-m2000", "sm_52")
_register_cuda_tag("nvidia/quadro-k2200", "sm_50")
_register_cuda_tag("nvidia/quadro-k2000", "sm_30")
_register_cuda_tag("nvidia/quadro-k2000d", "sm_30")
_register_cuda_tag("nvidia/quadro-k1200", "sm_50")
_register_cuda_tag("nvidia/quadro-k620", "sm_50")
_register_cuda_tag("nvidia/quadro-k600", "sm_30")
_register_cuda_tag("nvidia/quadro-k420", "sm_30")
_register_cuda_tag("nvidia/quadro-410", "sm_30")
_register_cuda_tag("nvidia/quadro-plex-7000", "sm_20", regs=32768)

# =====================================================================
# Quadro / RTX professional mobile GPUs (Turing)
# =====================================================================
_register_cuda_tag("nvidia/rtx-5000", "sm_75")
_register_cuda_tag("nvidia/rtx-4000", "sm_75")
_register_cuda_tag("nvidia/rtx-3000", "sm_75")
_register_cuda_tag("nvidia/t2000", "sm_75")
_register_cuda_tag("nvidia/t1000", "sm_75")
_register_cuda_tag("nvidia/p620", "sm_61")
_register_cuda_tag("nvidia/p520", "sm_61")

# =====================================================================
# Quadro professional mobile GPUs (Pascal / Maxwell)
# =====================================================================
_register_cuda_tag("nvidia/quadro-p5200", "sm_61")
_register_cuda_tag("nvidia/quadro-p4200", "sm_61")
_register_cuda_tag("nvidia/quadro-p3200", "sm_61")
_register_cuda_tag("nvidia/quadro-p3000", "sm_61")
_register_cuda_tag("nvidia/quadro-p500", "sm_61")
_register_cuda_tag("nvidia/quadro-m5500m", "sm_52")
_register_cuda_tag("nvidia/quadro-m2200", "sm_52")
_register_cuda_tag("nvidia/quadro-m1200", "sm_50")
_register_cuda_tag("nvidia/quadro-m620", "sm_52")
_register_cuda_tag("nvidia/quadro-m520", "sm_50")

# =====================================================================
# Quadro professional mobile GPUs (Kepler / Maxwell)
# =====================================================================
_register_cuda_tag("nvidia/quadro-k6000m", "sm_30")
_register_cuda_tag("nvidia/quadro-k5200m", "sm_30")
_register_cuda_tag("nvidia/quadro-k5100m", "sm_30")
_register_cuda_tag("nvidia/quadro-m5000m", "sm_50")
_register_cuda_tag("nvidia/quadro-k500m", "sm_30")
_register_cuda_tag("nvidia/quadro-k4200m", "sm_30")
_register_cuda_tag("nvidia/quadro-k4100m", "sm_30")
_register_cuda_tag("nvidia/quadro-m4000m", "sm_50")
_register_cuda_tag("nvidia/quadro-k3100m", "sm_30")
_register_cuda_tag("nvidia/quadro-m3000m", "sm_50")
_register_cuda_tag("nvidia/quadro-k2200m", "sm_30")
_register_cuda_tag("nvidia/quadro-k2100m", "sm_30")
_register_cuda_tag("nvidia/quadro-m2000m", "sm_50")
_register_cuda_tag("nvidia/quadro-k1100m", "sm_30")
_register_cuda_tag("nvidia/quadro-m1000m", "sm_50")
_register_cuda_tag("nvidia/quadro-k620m", "sm_50")
_register_cuda_tag("nvidia/quadro-k610m", "sm_35")
_register_cuda_tag("nvidia/quadro-m600m", "sm_50")
_register_cuda_tag("nvidia/quadro-k510m", "sm_35")
_register_cuda_tag("nvidia/quadro-m500m", "sm_50")

# =====================================================================
# NVS cards
# =====================================================================
_register_cuda_tag("nvidia/nvidia-nvs-810", "sm_50")
_register_cuda_tag("nvidia/nvidia-nvs-510", "sm_30")
_register_cuda_tag("nvidia/nvidia-nvs-315", "sm_21", regs=32768)
_register_cuda_tag("nvidia/nvidia-nvs-310", "sm_21", regs=32768)
_register_cuda_tag("nvidia/nvs-5400m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/nvs-5200m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/nvs-4200m", "sm_21", regs=32768)

# =====================================================================
# GeForce RTX 50-series desktop
# =====================================================================
_register_cuda_tag("nvidia/geforce-rtx-5060-ti", "sm_120", l2_cache_size_bytes=33554432)

# =====================================================================
# GeForce RTX 40-series desktop
# =====================================================================
_register_cuda_tag("nvidia/geforce-rtx-4090", "sm_89", l2_cache_size_bytes=75497472)

# =====================================================================
# GeForce RTX 30-series desktop
# =====================================================================
_register_cuda_tag("nvidia/geforce-rtx-3090-ti", "sm_86")
_register_cuda_tag("nvidia/geforce-rtx-3090", "sm_86")
_register_cuda_tag("nvidia/geforce-rtx-3080-ti", "sm_86")
_register_cuda_tag("nvidia/geforce-rtx-3080", "sm_86")
_register_cuda_tag("nvidia/geforce-rtx-3070-ti", "sm_86")
_register_cuda_tag("nvidia/geforce-rtx-3070", "sm_86")
_register_cuda_tag("nvidia/geforce-rtx-3060", "sm_86")

# =====================================================================
# GeForce RTX 20-series / TITAN (Turing)
# =====================================================================
_register_cuda_tag("nvidia/nvidia-titan-rtx", "sm_75")
_register_cuda_tag("nvidia/geforce-rtx-2080-ti", "sm_75")
_register_cuda_tag("nvidia/geforce-rtx-2080", "sm_75")
_register_cuda_tag("nvidia/geforce-rtx-2070", "sm_75")
_register_cuda_tag("nvidia/geforce-rtx-2060", "sm_75")

# =====================================================================
# GeForce TITAN / GTX 10-series (Pascal)
# =====================================================================
_register_cuda_tag("nvidia/nvidia-titan-v", "sm_70")
_register_cuda_tag("nvidia/nvidia-titan-xp", "sm_61")
_register_cuda_tag("nvidia/nvidia-titan-x", "sm_61")
_register_cuda_tag("nvidia/geforce-gtx-1080-ti", "sm_61")
_register_cuda_tag("nvidia/geforce-gtx-1080", "sm_61")
_register_cuda_tag("nvidia/geforce-gtx-1070-ti", "sm_61")
_register_cuda_tag("nvidia/geforce-gtx-1070", "sm_61")
_register_cuda_tag("nvidia/geforce-gtx-1060", "sm_61")
_register_cuda_tag("nvidia/geforce-gtx-1050", "sm_61")

# =====================================================================
# GeForce GTX 900/700 series desktop (Maxwell / Kepler)
# =====================================================================
_register_cuda_tag("nvidia/geforce-gtx-titan-x", "sm_52")
_register_cuda_tag("nvidia/geforce-gtx-titan-z", "sm_35")
_register_cuda_tag("nvidia/geforce-gtx-titan-black", "sm_35")
_register_cuda_tag("nvidia/geforce-gtx-titan", "sm_35")
_register_cuda_tag("nvidia/geforce-gtx-980-ti", "sm_52")
_register_cuda_tag("nvidia/geforce-gtx-980", "sm_52")
_register_cuda_tag("nvidia/geforce-gtx-970", "sm_52")
_register_cuda_tag("nvidia/geforce-gtx-960", "sm_52")
_register_cuda_tag("nvidia/geforce-gtx-950", "sm_52")
_register_cuda_tag("nvidia/geforce-gtx-780-ti", "sm_35")
_register_cuda_tag("nvidia/geforce-gtx-780", "sm_35")
_register_cuda_tag("nvidia/geforce-gtx-770", "sm_30")
_register_cuda_tag("nvidia/geforce-gtx-760", "sm_30")
_register_cuda_tag("nvidia/geforce-gtx-750-ti", "sm_50")
_register_cuda_tag("nvidia/geforce-gtx-750", "sm_50")
_register_cuda_tag("nvidia/geforce-gtx-690", "sm_30")
_register_cuda_tag("nvidia/geforce-gtx-680", "sm_30")
_register_cuda_tag("nvidia/geforce-gtx-670", "sm_30")
_register_cuda_tag("nvidia/geforce-gtx-660-ti", "sm_30")
_register_cuda_tag("nvidia/geforce-gtx-660", "sm_30")
_register_cuda_tag("nvidia/geforce-gtx-650-ti-boost", "sm_30")
_register_cuda_tag("nvidia/geforce-gtx-650-ti", "sm_30")
_register_cuda_tag("nvidia/geforce-gtx-650", "sm_30")

# =====================================================================
# GeForce GTX 500/400 series desktop (Fermi)
# =====================================================================
_register_cuda_tag("nvidia/geforce-gtx-560-ti", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gtx-550-ti", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gtx-460", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gts-450", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gtx-590", "sm_20", regs=32768)
_register_cuda_tag("nvidia/geforce-gtx-580", "sm_20", regs=32768)
_register_cuda_tag("nvidia/geforce-gtx-570", "sm_20", regs=32768)
_register_cuda_tag("nvidia/geforce-gtx-480", "sm_20", regs=32768)
_register_cuda_tag("nvidia/geforce-gtx-470", "sm_20", regs=32768)
_register_cuda_tag("nvidia/geforce-gtx-465", "sm_20", regs=32768)

# =====================================================================
# GeForce GT desktop (Kepler / Fermi)
# =====================================================================
_register_cuda_tag("nvidia/geforce-gt-740", "sm_30")
_register_cuda_tag("nvidia/geforce-gt-730", "sm_35")
_register_cuda_tag("nvidia/geforce-gt-730-ddr3,128bit", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gt-720", "sm_35")
_register_cuda_tag("nvidia/geforce-gt-705", "sm_35")
_register_cuda_tag("nvidia/geforce-gt-640-gddr5", "sm_35")
_register_cuda_tag("nvidia/geforce-gt-640-gddr3", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gt-630", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gt-620", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gt-610", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gt-520", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gt-440", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gt-430", "sm_21", regs=32768)

# =====================================================================
# GeForce notebook GPUs (Maxwell / Kepler)
# =====================================================================
_register_cuda_tag("nvidia/geforce-gtx-980m", "sm_52")
_register_cuda_tag("nvidia/geforce-gtx-970m", "sm_52")
_register_cuda_tag("nvidia/geforce-gtx-965m", "sm_52")
_register_cuda_tag("nvidia/geforce-gtx-960m", "sm_50")
_register_cuda_tag("nvidia/geforce-gtx-950m", "sm_50")
_register_cuda_tag("nvidia/geforce-940m", "sm_50")
_register_cuda_tag("nvidia/geforce-930m", "sm_50")
_register_cuda_tag("nvidia/geforce-920m", "sm_35")
_register_cuda_tag("nvidia/geforce-910m", "sm_52")
_register_cuda_tag("nvidia/geforce-gtx-880m", "sm_30")
_register_cuda_tag("nvidia/geforce-gtx-870m", "sm_30")
_register_cuda_tag("nvidia/geforce-gtx-860m-sm-30", "sm_30")
_register_cuda_tag("nvidia/geforce-gtx-860m-sm-50", "sm_50")
_register_cuda_tag("nvidia/geforce-gtx-850m", "sm_50")
_register_cuda_tag("nvidia/geforce-840m", "sm_50")
_register_cuda_tag("nvidia/geforce-830m", "sm_50")
_register_cuda_tag("nvidia/geforce-820m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-800m", "sm_21", regs=32768)

# =====================================================================
# GeForce notebook GPUs (Kepler / Fermi, older)
# =====================================================================
_register_cuda_tag("nvidia/geforce-gtx-780m", "sm_30")
_register_cuda_tag("nvidia/geforce-gtx-770m", "sm_30")
_register_cuda_tag("nvidia/geforce-gtx-765m", "sm_30")
_register_cuda_tag("nvidia/geforce-gtx-760m", "sm_30")
_register_cuda_tag("nvidia/geforce-gtx-680mx", "sm_30")
_register_cuda_tag("nvidia/geforce-gtx-680m", "sm_30")
_register_cuda_tag("nvidia/geforce-gtx-675mx", "sm_30")
_register_cuda_tag("nvidia/geforce-gtx-675m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gtx-670mx", "sm_30")
_register_cuda_tag("nvidia/geforce-gtx-670m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gtx-660m", "sm_30")
_register_cuda_tag("nvidia/geforce-gt-755m", "sm_30")
_register_cuda_tag("nvidia/geforce-gt-750m", "sm_30")
_register_cuda_tag("nvidia/geforce-gt-650m", "sm_30")
_register_cuda_tag("nvidia/geforce-gt-745m", "sm_30")
_register_cuda_tag("nvidia/geforce-gt-645m", "sm_30")
_register_cuda_tag("nvidia/geforce-gt-740m", "sm_30")
_register_cuda_tag("nvidia/geforce-gt-730m", "sm_30")
_register_cuda_tag("nvidia/geforce-gt-640m", "sm_30")
_register_cuda_tag("nvidia/geforce-gt-640m-le", "sm_30")
_register_cuda_tag("nvidia/geforce-gt-735m", "sm_30")
_register_cuda_tag("nvidia/geforce-gt-635m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gt-630m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gt-625m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gt-720m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gt-620m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-710m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-705m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-610m", "sm_21", regs=32768)

# =====================================================================
# GeForce notebook GPUs (Fermi, GTX 5xx/4xxM)
# =====================================================================
_register_cuda_tag("nvidia/geforce-gtx-580m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gtx-570m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gtx-560m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gt-555m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gt-550m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gt-540m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gt-525m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gt-520mx", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gt-520m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gtx-485m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gtx-470m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gtx-460m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gt-445m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gt-435m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gt-420m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gt-415m", "sm_21", regs=32768)
_register_cuda_tag("nvidia/geforce-gtx-480m", "sm_20", regs=32768)
_register_cuda_tag("nvidia/geforce-410m", "sm_21", regs=32768)

# =====================================================================
# Jetson boards (simple, no host)
# =====================================================================
_register_cuda_tag("nvidia/jetson-nano", "sm_53", regs=32768)
_register_cuda_tag("nvidia/jetson-tx2", "sm_62", regs=32768)
_register_cuda_tag("nvidia/jetson-tx1", "sm_53", regs=32768)
_register_cuda_tag("nvidia/tegra-x1", "sm_53", regs=32768)

# =====================================================================
# Jetson boards (with LLVM host)
# =====================================================================
_register_jetson_tag("nvidia/jetson-agx-xavier", "sm_72", "carmel", 8)
_register_jetson_tag("nvidia/jetson-orin-nano", "sm_87", "carmel", 6)
_register_jetson_tag("nvidia/jetson-agx-orin-32gb", "sm_87", "cortex-a78", 8)
_register_jetson_tag("nvidia/jetson-agx-orin-64gb", "sm_87", "cortex-a78", 12)
