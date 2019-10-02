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

set verilog_dir [lindex $argv 0]
set out_dir [lindex $argv 1]

create_project -force package_rtl $out_dir/package_rtl

add_files -norecurse [glob $verilog_dir/*.v]

update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
ipx::package_project -root_dir $out_dir/ip -vendor xilinx.com -library RTLKernel -taxonomy /KernelIP -import_files -set_current false
ipx::unload_core $out_dir/ip/component.xml
ipx::edit_ip_in_project -upgrade true -name tmp_edit_project -directory $out_dir/ip $out_dir/ip/component.xml
set_property core_revision 2 [ipx::current_core]
foreach up [ipx::get_user_parameters] {
  ipx::remove_user_parameter [get_property NAME $up] [ipx::current_core]
}
ipx::create_xgui_files [ipx::current_core]
ipx::associate_bus_interfaces -busif m_axi_gmem -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif s_axi_control -clock ap_clk [ipx::current_core]
set_property supported_families { } [ipx::current_core]
set_property auto_family_support_level level_2 [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::save_core [ipx::current_core]
close_project -delete
