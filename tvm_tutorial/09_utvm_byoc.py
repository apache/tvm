# Example inference on microtvm
# Using framework prequantized models in TVM
import pathlib
import tarfile
import subprocess
import shutil
import sys

from PIL import Image

import numpy as np

import torch

import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import wildcard, is_op, is_expr

class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(16, 16, 9, bias=False, padding=4, stride=1, dilation=1, groups=1),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
        )
        self.fc = torch.nn.Sequential(torch.nn.Linear(16, 10), torch.nn.Softmax(dim=0),)

    def forward(self, x):
        y = self.conv(x)
        y = x + y
#        y = y.mean(axis=-1)
#        y = self.fc(y)
        return y

pt_inp = torch.rand([1, 16, 32])
inp = pt_inp.numpy()
pt_model = TorchModel()
pt_model.eval()
script_module = torch.jit.trace(pt_model, pt_inp).eval()

with torch.no_grad():
    pt_result = script_module(pt_inp).numpy()

input_name = "input"
input_shapes = [(input_name, (1, 16, 32))]
mod, params = relay.frontend.from_pytorch(script_module, input_shapes)


#custom_target_name = "ccompiler"
custom_target_name = "custom_target"
@relay.op.contrib.register_pattern_table(custom_target_name)
def pattern_table():
    
    add_pattern = is_op("add")(wildcard(), wildcard())
 
    patterns = [
        ("custom_target.add", add_pattern, lambda exp: True)
    ]
    
    return patterns


@tvm.ir.register_op_attr("add", f"target.{custom_target_name}")
def add_attr(expr):
    shapes = set()
    for arg in expr.args:
        ty = arg.checked_type
        shape = tuple(ty.shape)
        print(shape)
        shapes.add(shape)

    return len(shapes) == 1

#mod = relay.transform.MergeComposite(pattern_table())(mod)
#print(mod)
mod = relay.transform.AnnotateTarget(custom_target_name)(mod)
print(mod)
mod = relay.transform.MergeCompilerRegions()(mod)
print(mod)
mod = relay.transform.PartitionGraph()(mod)
print(mod)

    

#target="llvm -mcpu=skylake"
target = "c"
target += " -system-lib=1 -runtime=c"



with tvm.transform.PassContext(
    opt_level=3, 
    config={"tir.disable_vectorize": True}, 
    disabled_pass=["AlterOpLayout"]
):
    module = relay.build(mod, target=target, params=params)


#for source_module in module.get_lib().imported_modules:
#    source_code = source_module.get_source()
#
#    print("Source Code:")
#    print(source_code)



# Model Library Format: http://tvm.apache.org/docs/arch/model_library_format.html

model_library_format_tar_path = pathlib.Path("build/09/module.tar")
model_library_format_tar_path.unlink(missing_ok=True)
model_library_format_tar_path.parent.mkdir(parents=True, exist_ok=True)

tvm.micro.export_model_library_format(module, model_library_format_tar_path)

print("Built MLF Library: ")
with tarfile.open(model_library_format_tar_path, "r:*") as tar_f:
    print("\n".join(f" - {m.name}" for m in tar_f.getmembers()))
    tar_f.extractall(model_library_format_tar_path.parent)

template_project_path = pathlib.Path(".").absolute() / "microtvm_template"
project_options = {'project_type': 'host_driven'}

generated_project_dir = model_library_format_tar_path.parent.absolute() / "generated-project"
if generated_project_dir.exists():
    shutil.rmtree(generated_project_dir, ignore_errors=True)

generated_project = tvm.micro.generate_project(
    template_project_path, module, generated_project_dir, project_options
)

# Build and flash the project
generated_project.build()
generated_project.flash()

with tvm.micro.Session(transport_context_manager=generated_project.transport()) as session:
    graph_mod = tvm.micro.create_local_graph_executor(
        module.get_graph_json(), session.get_system_lib(), session.device
    )

    graph_mod.set_input(**module.get_params())

    graph_mod.set_input(input_name, inp)
    graph_mod.run()

    tvm_result = graph_mod.get_output(0).numpy()

    print(tvm_result)
    print(pt_result)
