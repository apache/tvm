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
"""tvm.contrib.msc.framework.tensorrt.tools.quantize.quantizer"""

import os
import struct
from typing import List, Dict, Any, Tuple

import tvm
from tvm.contrib.msc.core.ir import MSCGraph
from tvm.contrib.msc.core.tools.tool import ToolType, ToolStrategy
from tvm.contrib.msc.core.tools.quantize import BaseQuantizer, QuantizeStage
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


class TensorRTQuantizerFactory(object):
    """Quantizer factory for tensorrt"""

    def create(self, base_cls: BaseQuantizer) -> BaseQuantizer:
        """Create adaptive quantizer

        Parameters
        ----------
        base_cls: BaseQuantizer
            The base quantizer class

        Returns
        -------
        quantizer_cls: BaseQuantizer
            The quantizer class.
        """

        class Quantizer(base_cls):
            """Adaptive quantizer for tensorrt"""

            def setup(self) -> dict:
                """Setup the tool

                Returns
                -------
                info: dict
                    The setup info.
                """

                if self._plan:
                    self._use_range = all(
                        info.get("use_range", False) for info in self._plan.values()
                    )
                else:
                    self._use_range = True
                return super().setup()

            def _reset(
                self, graphs: List[MSCGraph], weights: List[Dict[str, tvm.nd.array]]
            ) -> Tuple[List[MSCGraph], List[Dict[str, tvm.nd.array]]]:
                """Reset the tool

                Parameters
                ----------
                graphs: list<MSCgraph>
                    The msc graphs.
                weights: list<dict<str, tvm.nd.array>>
                    The weights

                Returns
                -------
                graphs: list<MSCgraph>
                    The msc graphs.
                weights: list<dict<str, tvm.nd.array>>
                    The weights
                """

                config_folder = msc_utils.get_config_dir()
                self._range_files = [config_folder.relpath(g.name + ".range") for g in graphs]
                calibrate_root = msc_utils.get_dataset_dir().create_dir("Calibrate")
                self._calibrate_folders = [calibrate_root.relpath(g.name) for g in graphs]
                if self._calibrated:
                    if self._use_range:
                        for r_file, graph in zip(self._range_files, graphs):
                            if not os.path.isfile(r_file):
                                self._plan_to_range(graph, r_file)
                            self._logger.debug(
                                "G[%s](%s) use range file: %s",
                                graph.name,
                                self._stage,
                                r_file,
                            )
                    else:
                        self._quantized_tensors = set()
                elif self._stage == QuantizeStage.GATHER:
                    self._calibrate_savers = []
                    for folder, graph in zip(self._calibrate_folders, graphs):
                        saver_options = {"input_names": [i.name for i in graph.get_inputs()]}
                        saver = msc_utils.IODataSaver(folder, saver_options)
                        self._calibrate_savers.append(saver)
                        self._logger.debug(
                            "G[%s](%s) create calibrate saver: %s",
                            graph.name,
                            self._stage,
                            saver,
                        )
                else:
                    assert all(
                        msc_utils.is_io_dataset(f) for f in self._calibrate_folders
                    ), "Some IODataset missing: " + str(self._calibrate_folders)
                return super()._reset(graphs, weights)

            def _execute_after_build(self, codegen_context: dict) -> dict:
                """Execute after model build

                Parameters
                ----------
                codegen_context: dict
                    The context.

                Returns
                ----------
                codegen_context: dict
                    The processed context.
                """

                if self._stage == QuantizeStage.GATHER and self._forward_cnt == 0:
                    return codegen_context
                if not self._use_range:
                    return codegen_context
                processed = ["// Set int8 calibrator"]
                range_file = self.get_graph().name + ".range"
                version = [int(v) for v in codegen_context["version"].split(".")]
                if msc_utils.compare_version(version, [6, 0, 0]) >= 0:
                    configer = codegen_context["config"]
                else:
                    configer = codegen_context["builder"]
                # check the range file if calibrated
                if self._calibrated:
                    processed.extend(
                        [
                            'if (!FileUtils::FileExist("{}")) {{'.format(range_file),
                            '  logger.log(ILogger::Severity::kERROR, "{} not exist!");'.format(
                                range_file
                            ),
                            "  return -1;",
                            "}",
                        ]
                    )
                processed.extend(
                    [
                        'MSCInt8EntropyCalibrator2 calibrator("{}", "{}");'.format(
                            range_file, self._calibrate_folders[self._graph_id]
                        ),
                        "{}->setInt8Calibrator(&calibrator);".format(configer),
                    ]
                )
                codegen_context["processed"].extend(processed)
                return codegen_context

            def _execute_before_forward(self, step_context: dict) -> dict:
                """Execute before model forward

                Parameters
                ----------
                step_context: dict
                    The context.

                Returns
                ----------
                step_context: dict
                    The processed context.
                """

                if self._stage == QuantizeStage.GATHER:
                    saver = self._calibrate_savers[self._graph_id]
                    saver.save_batch(
                        {name: data.asnumpy() for name, data in step_context["datas"].items()}
                    )
                    for name, data in step_context["datas"].items():
                        self.debug_tensor(data, name, "any", "ctx_gathered")
                super()._execute_before_forward(step_context)

            def _quantize_tensor(
                self,
                tensor_ctx: Dict[str, str],
                name: str,
                consumer: str,
                strategys: List[ToolStrategy],
            ) -> Dict[str, str]:
                """Quantize tensor

                Parameters
                -------
                tensor_ctx: dict<str, str>
                    Tensor describe items.
                name: str
                    The name of the tensor.
                consumer: str
                    The name of the consumer.
                strategys: list<ToolStrategy>
                    The strategys for the tensor.

                Returns
                -------
                tensor_ctx: dict<str, str>
                    Tensor items with processed.
                """

                if not self._use_range and name not in self._quantized_tensors:
                    self._quantized_tensors.add(name)
                    return super()._quantize_tensor(tensor_ctx, name, consumer, strategys)
                return tensor_ctx

            def calibrate(self) -> dict:
                """Calibrate the datas

                Returns
                -------
                plan: dict
                    The calibrated plan.
                """

                for r_file, graph in zip(self._range_files, self._graphs):
                    self._range_to_plan(graph, r_file)
                self._calibrated, self._forward_cnt = True, 0
                self.change_stage("quantize")
                return self._plan

            def config_generate(self, generate_config: Dict[str, Any]) -> Dict[str, Any]:
                """Update the generate configs

                Parameters
                ----------
                generate_config: dict<str, Any>
                    The generate_config.

                Returns
                -------
                generate_config: dict<str, Any>
                    The updated generate_config.
                """

                if self._calibrated:
                    if self._use_range:
                        for config, r_file in zip(generate_config["codegen"], self._range_files):
                            if os.path.isfile(r_file):
                                config.update({"range_file": r_file, "precision": "int8"})
                elif self._stage == QuantizeStage.GATHER and self._forward_cnt > 0:
                    for config, saver, r_file in zip(
                        generate_config["codegen"], self._calibrate_savers, self._range_files
                    ):
                        saver.finalize()
                        self._logger.debug(
                            "%ssave %d datas to %s",
                            self.msg_mark(in_forward=False),
                            self._forward_cnt,
                            saver.folder,
                        )
                        config.update(
                            {"dataset": saver.folder, "range_file": r_file, "precision": "int8"}
                        )
                    self.change_stage(QuantizeStage.CALIBRATE)
                return generate_config

            def _plan_to_range(self, graph: MSCGraph, range_file: str, title="MSCCalibrate"):
                """Extract plan config to range_file

                Parameters
                ----------
                plan: dict
                    The plan.
                graph: MSCGraph
                    The graph.
                range_file: str
                    The output range_file path.
                title: str
                    The title of the range file.
                """

                def _scale_to_hex(scale):
                    return hex(struct.unpack("<I", struct.pack("<f", scale / 127))[0])[2:]

                recorded = set()
                with open(range_file, "w") as f:
                    f.write(title + "\n")
                    for name, info in self._plan.items():
                        t_name, _ = self.from_tensor_id(name)
                        if not graph.find_tensor(t_name):
                            continue
                        if t_name not in recorded:
                            f.write("{}: {}\n").format(t_name, _scale_to_hex(info["scale"]))
                            recorded.add(t_name)
                self._logger.debug(
                    "Graph[%s](%s) extract %d plan to range %s",
                    graph.name,
                    self._stage,
                    len(recorded),
                    range_file,
                )

            def _range_to_plan(self, graph: MSCGraph, range_file: str):
                """Extract scale in range_file to plan

                Parameters
                ----------
                graph: MSCGraph
                    The graph.
                range_file: str
                    The input range_file path.
                """

                range_num = 0
                with open(range_file, "r") as f:
                    f.readline()
                    line = f.readline()
                    while line:
                        name, scale = line.split(": ")
                        scale = scale.strip()
                        if scale == "0":
                            value = 0.0
                        else:
                            value = struct.unpack("!f", bytes.fromhex(scale))[0] * 127
                        range_num += 1
                        consumers = graph.find_consumers(name)
                        if consumers:
                            for c in consumers:
                                self._plan[self.to_tensor_id(name, c.name)] = {
                                    "scale": value,
                                    "use_range": True,
                                }
                        else:
                            self._plan[self.to_tensor_id(name, "exit")] = {
                                "scale": value,
                                "use_range": True,
                            }
                        line = f.readline()
                self._logger.debug(
                    "Graph[%s](%s) extract %d range to plan from %s",
                    graph.name,
                    self._stage,
                    range_num,
                    range_file,
                )

            @classmethod
            def framework(cls):
                return MSCFramework.TENSORRT

        return Quantizer


factory = TensorRTQuantizerFactory()
tools = msc_utils.get_registered_tool_cls(MSCFramework.MSC, ToolType.QUANTIZER, tool_style="all")
for tool in tools.values():
    msc_utils.register_tool_cls(factory.create(tool))
