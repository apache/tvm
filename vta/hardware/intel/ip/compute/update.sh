sed -r -i -e 's/reg\ \[31:0\]\ uop_mem\ \[0:1023\];/reg\ \[31:0\]\ uop_mem\ \[0:1023\]\ \/\*\ synthesis\ ramstyle\ =\ "M20K"\ \*\/;/g' ../../../chisel/ip/vta/compute/Compute.v
sed -r -i -e 's/reg\ \[511:0\]\ acc_mem\ \[0:255\];/reg\ \[511:0\]\ acc_mem\ \[0:255\]\ \/\*\ synthesis\ ramstyle\ =\ "M20K"\ \*\/;/g' ../../../chisel/ip/vta/compute/Compute.v
cp ../../../chisel/ip/vta/compute/Compute.v ../../ip/compute/compute.v
cp ../../../chisel/ip/vta/compute/Compute.v ../../scripts/soc_system/synthesis/submodules/compute.v
