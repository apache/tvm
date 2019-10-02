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
