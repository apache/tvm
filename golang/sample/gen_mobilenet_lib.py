import tvm
from tvm import relay
import tvm.relay.testing

######################################################################
# Load Neural Network in Relay
####################################################

mod, params = relay.testing.mobilenet.get_workload(batch_size=1)

# set show_meta_data=True if you want to show meta data
print(mod.astext(show_meta_data=False))

######################################################################
# Compilation
####################################################

target = 'llvm'

# Build with Relay
with relay.build_config(opt_level=0):
    graph, lib, params = relay.build_module.build(
        mod, target, params=params)

######################################################################
# Save and Load Compiled Module
# -----------------------------
# We can also save the graph, lib and parameters into files
####################################################

lib.export_library("./mobilenet.so")
print('lib export succeefully')

with open("./mobilenet.json", "w") as fo:
   fo.write(graph)

with open("./mobilenet.params", "wb") as fo:
   fo.write(relay.save_param_dict(params))
