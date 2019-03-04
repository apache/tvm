sed -r -i -e 's/reg\ \[127:0\]\ _T_35\ \[0:7\];/reg\ \[127:0\]\ _T_35\ \[0:7\]\ \/\*\ synthesis\ ramstyle\ =\ "M20K"\ \*\/;/g' ../../../chisel/ip/vta/store/Store.v
cp ../../../chisel/ip/vta/store/Store.v ../../ip/store/store.v
cp ../../../chisel/ip/vta/store/Store.v ../../scripts/soc_system/synthesis/submodules/store.v
