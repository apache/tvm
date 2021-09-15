import argparse
import os
import logging
import numpy as np

import tvm
import tvm.micro
import tvm.relay as relay

from tvm.micro.interface_api import generate_c_interface_header

from common import *

_LOG = logging.getLogger(__name__)


TEMPLATE_PROJECT_DIR = tvm_repo_root() + "/apps/microtvm/grovety/template_project"

PLATFORMS = {
    "stm32f746xx_nucleo": {"model": "stm32f746xx", "board": "nucleo_f746zg", "mcpu": "cortex-m7", "march": "armv7e-m"},
    "stm32f746xx_disco": {"model": "stm32f746xx", "board": "stm32f746g_disco", "mcpu": "cortex-m7", "march": "armv7e-m"},
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model on the MCU")
    parser.add_argument("--model", type=str, choices=["cifar10", "mnist8", "sine", "cifar10_2"], default="cifar10", help="Model type")
    parser.add_argument("--relay", action='store_true', help="print relay for the model and exit")
    parser.add_argument("--platform", type=str, choices=PLATFORMS.keys(), default=list(PLATFORMS.keys())[0], help="Platform")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO"], default="INFO", help="Log level")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--skip-flash", action='store_true', help="Do not flash the MCU after the build")
    parser.add_argument("--benchmark", action="store_true", help="Enable per-op benchmarking")

    args = parser.parse_args()
    print(args)

    logging.basicConfig(level=args.log_level)
    platform = PLATFORMS[args.platform]

    target_str = f"c -keys=arm_cpu -mcpu={platform['mcpu']}  -march={platform['march']} -model={platform['model']} -runtime=c -link-params=1 --executor=aot --unpacked-api=1 --interface-api=c"

    # open model
    if args.model == "mnist8":
        import mnist8
        relay_mod, params, input, output = mnist8.open_model()
        dataset = mnist8.get_data()
    elif args.model == "cifar10":
        import cifar10
        relay_mod, params, input, output = cifar10.open_model()
        dataset = cifar10.get_data()
    elif args.model == "cifar10_2":
        import cifar10_2
        relay_mod, params, input, output = cifar10_2.open_model(target_str)
        dataset = cifar10_2.get_data()
    elif args.model == "sine":
        import sine
        relay_mod, params, input, output = sine.open_model()
        dataset = sine.get_data()
    else:
        raise NotImplementedError

    if args.relay:
        print(relay_mod)
        exit(0)

    # CONV: layout conversion to met instrinsics requrements for conv2d
    desired_layouts = {'qnn.conv2d': ['NHWC', 'HWOI'], 'nn.conv2d': ['NHWC', 'HWOI']}
    seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(), relay.transform.ConvertLayout(desired_layouts)])
    with tvm.transform.PassContext(opt_level=3):
        relay_mod = seq(relay_mod)

    # build relay for the target
    target = tvm.target.target.Target(target_str)
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lowered = relay.build(relay_mod, target, params=params)

    # generate workspace
    workspace_dir = create_workspace_dir(platform["board"], args.model+"_aot", mkdir=False)
    project = tvm.micro.generate_project(
        str(TEMPLATE_PROJECT_DIR),
        lowered,
        workspace_dir,
        {
            "project_type": "aot_demo",
            "west_cmd": "west",
            "verbose": args.verbose,
            "zephyr_board": platform["board"],
            "benchmark": args.benchmark
        }
    )

    input_tensor, input_shape, input_dtype = input
    output = 'output', output[1], output[2] # TODO check TVM's code generation of default_lib0.c
    output_tensor, output_shape, output_dtype = output

    input_data =  np.zeros(shape=input_shape, dtype=input_dtype)
    output_data = np.zeros(shape=output_shape, dtype=output_dtype)

    generated_include_path = workspace_dir / "include"
    os.makedirs(generated_include_path, exist_ok=True)
    generate_c_interface_header(lowered.libmod_name, [input_tensor], ["output"], generated_include_path)
    create_header_file(generated_include_path, input, output)

    # build the project
    project.build()

    if args.skip_flash:
        exit(0)

    # flash to the MCU
    project.flash()

    # communicate with the board
    with project.transport() as transport:

        for label, data in dataset:
            data = np.reshape(data, -1)
            data_s = ','.join(str(e) for e in data)

            transport.write(bytes(f"#input:{data_s}\n", 'UTF-8'), timeout_sec=5)
            result_line = get_message(transport, "#result", timeout_sec=5)
            r = result_line.strip("\n").split(":")

            output_values = list(map(float, r[1].split(',')))
            max_index = np.argmax(output_values)

            op_timers = ['gemm', 'max_pool', 'avg_pool', 'relu', 'total']
            times = r[2:]

            if len(times) < len(op_timers):
                times = [0.0] * (len(op_timers) - len(times)) + times

            benchmark_str = ' '.join([f"{n}: {float(t)/1000.0}ms" for n, t in zip(op_timers,times)])

            logging.info(f"input={label}; max_index={max_index}; {benchmark_str}; output={output_values}")
