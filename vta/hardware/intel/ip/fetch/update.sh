# sed -r -i -e 's/reg\ \[31:0\]\ uop_mem\ \[0:255\];/reg\ \[31:0\]\ uop_mem\ \[0:255\]\ \/\*\ synthesis\ ramstyle\ =\ "M20K"\ \*\/;/g' ../../../chisel/ip/vta/fetch/Fetch.v
# sed -r -i -e 's/reg\ \[511:0\]\ acc_mem\ \[0:255\];/reg\ \[511:0\]\ acc_mem\ \[0:255\]\ \/\*\ synthesis\ ramstyle\ =\ "M20K"\ \*\/;/g' ../../../chisel/ip/vta/fetch/Fetch.v
cp ../../../chisel/ip/vta/fetch/Fetch.v ../../ip/fetch/fetch.v
cp ../../../chisel/ip/vta/fetch/Fetch.v ../../scripts/soc_system/synthesis/submodules/fetch.v
