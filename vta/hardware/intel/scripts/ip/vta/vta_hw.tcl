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

# request TCL package from ACDS 16.1
package require -exact qsys 16.1

# module vta
set_module_property DESCRIPTION ""
set_module_property NAME vta
set_module_property VERSION 1.0
set_module_property INTERNAL false
set_module_property OPAQUE_ADDRESS_MAP true
set_module_property AUTHOR ""
set_module_property DISPLAY_NAME "VTA Subsystem"
set_module_property INSTANTIATE_IN_SYSTEM_MODULE true
set_module_property EDITABLE true
set_module_property REPORT_TO_TALKBACK false
set_module_property ALLOW_GREYBOX_GENERATION false
set_module_property REPORT_HIERARCHY false

# file sets
add_fileset QUARTUS_SYNTH QUARTUS_SYNTH "" ""
set_fileset_property QUARTUS_SYNTH TOP_LEVEL IntelShell
set_fileset_property QUARTUS_SYNTH ENABLE_RELATIVE_INCLUDE_PATHS false
set_fileset_property QUARTUS_SYNTH ENABLE_FILE_OVERWRITE_MODE false
add_fileset_file VTAShell.v VERILOG PATH VTAShell.v TOP_LEVEL_FILE

# connection point clock
add_interface clock clock end
set_interface_property clock clockRate 0
set_interface_property clock ENABLED true
set_interface_property clock EXPORT_OF ""
set_interface_property clock PORT_NAME_MAP ""
set_interface_property clock CMSIS_SVD_VARIABLES ""
set_interface_property clock SVD_ADDRESS_GROUP ""

add_interface_port clock clock clk Input 1

# connection point reset
add_interface reset reset end
set_interface_property reset associatedClock clock
set_interface_property reset synchronousEdges DEASSERT
set_interface_property reset ENABLED true
set_interface_property reset EXPORT_OF ""
set_interface_property reset PORT_NAME_MAP ""
set_interface_property reset CMSIS_SVD_VARIABLES ""
set_interface_property reset SVD_ADDRESS_GROUP ""

add_interface_port reset reset reset Input 1

# connection point m_axi_gmem
add_interface m_axi_gmem axi start
set_interface_property m_axi_gmem associatedClock clock
set_interface_property m_axi_gmem associatedReset reset
set_interface_property m_axi_gmem readIssuingCapability 1
set_interface_property m_axi_gmem writeIssuingCapability 1
set_interface_property m_axi_gmem combinedIssuingCapability 1
set_interface_property m_axi_gmem ENABLED true
set_interface_property m_axi_gmem EXPORT_OF ""
set_interface_property m_axi_gmem PORT_NAME_MAP ""
set_interface_property m_axi_gmem CMSIS_SVD_VARIABLES ""
set_interface_property m_axi_gmem SVD_ADDRESS_GROUP ""

add_interface_port m_axi_gmem io_mem_ar_ready arready Input 1
add_interface_port m_axi_gmem io_mem_ar_valid arvalid Output 1
add_interface_port m_axi_gmem io_mem_ar_bits_addr araddr Output 32
add_interface_port m_axi_gmem io_mem_ar_bits_burst arburst Output 2
add_interface_port m_axi_gmem io_mem_ar_bits_cache arcache Output 4
add_interface_port m_axi_gmem io_mem_ar_bits_len arlen Output 4
add_interface_port m_axi_gmem io_mem_ar_bits_lock arlock Output 2
add_interface_port m_axi_gmem io_mem_ar_bits_prot arprot Output 3
add_interface_port m_axi_gmem io_mem_ar_bits_size arsize Output 3
add_interface_port m_axi_gmem io_mem_ar_bits_user aruser Output 5
add_interface_port m_axi_gmem io_mem_ar_bits_id arid Output 1
add_interface_port m_axi_gmem io_mem_r_ready rready Output 1
add_interface_port m_axi_gmem io_mem_r_valid rvalid Input 1
add_interface_port m_axi_gmem io_mem_r_bits_data rdata Input 64
add_interface_port m_axi_gmem io_mem_r_bits_id rid Input 1
add_interface_port m_axi_gmem io_mem_r_bits_last rlast Input 1
add_interface_port m_axi_gmem io_mem_r_bits_resp rresp Input 2
add_interface_port m_axi_gmem io_mem_aw_valid awvalid Output 1
add_interface_port m_axi_gmem io_mem_aw_ready awready Input 1
add_interface_port m_axi_gmem io_mem_aw_bits_addr awaddr Output 32
add_interface_port m_axi_gmem io_mem_aw_bits_prot awprot Output 3
add_interface_port m_axi_gmem io_mem_aw_bits_burst awburst Output 2
add_interface_port m_axi_gmem io_mem_aw_bits_cache awcache Output 4
add_interface_port m_axi_gmem io_mem_aw_bits_len awlen Output 4
add_interface_port m_axi_gmem io_mem_aw_bits_lock awlock Output 2
add_interface_port m_axi_gmem io_mem_aw_bits_size awsize Output 3
add_interface_port m_axi_gmem io_mem_aw_bits_user awuser Output 5
add_interface_port m_axi_gmem io_mem_aw_bits_id awid Output 1
add_interface_port m_axi_gmem io_mem_w_bits_data wdata Output 64
add_interface_port m_axi_gmem io_mem_w_ready wready Input 1
add_interface_port m_axi_gmem io_mem_w_valid wvalid Output 1
add_interface_port m_axi_gmem io_mem_w_bits_last wlast Output 1
add_interface_port m_axi_gmem io_mem_w_bits_strb wstrb Output 8
add_interface_port m_axi_gmem io_mem_w_bits_id wid Output 1
add_interface_port m_axi_gmem io_mem_b_ready bready Output 1
add_interface_port m_axi_gmem io_mem_b_valid bvalid Input 1
add_interface_port m_axi_gmem io_mem_b_bits_resp bresp Input 2
add_interface_port m_axi_gmem io_mem_b_bits_id bid Input 1

# connection point s_axi_control
add_interface s_axi_control axi end
set_interface_property s_axi_control associatedClock clock
set_interface_property s_axi_control associatedReset reset
set_interface_property s_axi_control readAcceptanceCapability 1
set_interface_property s_axi_control writeAcceptanceCapability 1
set_interface_property s_axi_control combinedAcceptanceCapability 1
set_interface_property s_axi_control readDataReorderingDepth 1
set_interface_property s_axi_control bridgesToMaster ""
set_interface_property s_axi_control ENABLED true
set_interface_property s_axi_control EXPORT_OF ""
set_interface_property s_axi_control PORT_NAME_MAP ""
set_interface_property s_axi_control CMSIS_SVD_VARIABLES ""
set_interface_property s_axi_control SVD_ADDRESS_GROUP ""

add_interface_port s_axi_control io_host_aw_ready awready Output 1
add_interface_port s_axi_control io_host_aw_valid awvalid Input 1
add_interface_port s_axi_control io_host_aw_bits_addr awaddr Input 16
add_interface_port s_axi_control io_host_aw_bits_prot awprot Input 3
add_interface_port s_axi_control io_host_w_valid wvalid Input 1
add_interface_port s_axi_control io_host_w_ready wready Output 1
add_interface_port s_axi_control io_host_w_bits_data wdata Input 32
add_interface_port s_axi_control io_host_b_ready bready Input 1
add_interface_port s_axi_control io_host_b_valid bvalid Output 1
add_interface_port s_axi_control io_host_b_bits_resp bresp Output 2
add_interface_port s_axi_control io_host_ar_ready arready Output 1
add_interface_port s_axi_control io_host_ar_valid arvalid Input 1
add_interface_port s_axi_control io_host_ar_bits_addr araddr Input 16
add_interface_port s_axi_control io_host_ar_bits_prot arprot Input 3
add_interface_port s_axi_control io_host_r_ready rready Input 1
add_interface_port s_axi_control io_host_r_valid rvalid Output 1
add_interface_port s_axi_control io_host_r_bits_resp rresp Output 2
add_interface_port s_axi_control io_host_r_bits_data rdata Output 32
add_interface_port s_axi_control io_host_aw_bits_id awid Input 13
add_interface_port s_axi_control io_host_ar_bits_id arid Input 13
add_interface_port s_axi_control io_host_aw_bits_len awlen Input 4
add_interface_port s_axi_control io_host_ar_bits_size arsize Input 3
add_interface_port s_axi_control io_host_r_bits_id rid Output 13
add_interface_port s_axi_control io_host_w_bits_id wid Input 13
add_interface_port s_axi_control io_host_b_bits_id bid Output 13
add_interface_port s_axi_control io_host_aw_bits_size awsize Input 3
add_interface_port s_axi_control io_host_aw_bits_burst awburst Input 2
add_interface_port s_axi_control io_host_aw_bits_lock awlock Input 2
add_interface_port s_axi_control io_host_aw_bits_cache awcache Input 4
add_interface_port s_axi_control io_host_ar_bits_burst arburst Input 2
add_interface_port s_axi_control io_host_ar_bits_cache arcache Input 4
add_interface_port s_axi_control io_host_ar_bits_len arlen Input 4
add_interface_port s_axi_control io_host_ar_bits_lock arlock Input 2
add_interface_port s_axi_control io_host_r_bits_last rlast Output 1
add_interface_port s_axi_control io_host_w_bits_last wlast Input 1
add_interface_port s_axi_control io_host_w_bits_strb wstrb Input 4
