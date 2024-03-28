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
# pylint: disable=import-outside-toplevel, unused-argument
"""tvm.contrib.msc.pipeline.worker"""

import os
import time
import logging
from typing import Any, List, Tuple

import tvm
from tvm.contrib.msc.core.runtime import BaseRunner
from tvm.contrib.msc.core.tools import ToolType
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core.utils.message import MSCStage
from tvm.contrib.msc.core import utils as msc_utils
from .utils import support_tool, get_tool_stage, map_tools


class BasePipeWorker(object):
    """Base Worker of MSC pipeline

    Parameters
    ----------
    model: Any
        The raw model in framwork.
    config: dict
        The config for pipeline.
    workspace: MSCDirectory
        The workspace.
    plugins: dict
        The plugins for pipeline.
    run_optimize: bool
        Whether to run optimize.
    run_compile: bool
        Whether to run compile.
    logger: logging.Logger
        The logger.
    name: str
        The name of the worker.
    """

    def __init__(
        self,
        model: Any,
        config: dict,
        workspace: msc_utils.MSCDirectory,
        plugins: dict = None,
        logger: logging.Logger = None,
        name: str = "main",
    ):
        # check/set default stage
        for key in ["inputs", "outputs", "dataset"]:
            assert key in config, "Missing {} in config".format(key)

        self._config = msc_utils.copy_dict(config)
        self._workspace = workspace
        self._plugins = plugins
        self._model_type = config["model_type"]
        self._optimize_type = config.get(MSCStage.OPTIMIZE, {}).get("run_type", self._model_type)
        self._compile_type = config.get(MSCStage.COMPILE, {}).get("run_type", self._model_type)
        runner_cls = self._get_runner_cls(self._model_type)
        self._model, self._device, self._training = runner_cls.load_native(model, config)
        self._verbose = config.get("verbose", "info")
        self._logger = logger or msc_utils.get_global_logger()
        self._name = name
        self._optimized, self._compiled = False, False
        self.setup()

    def setup(self) -> dict:
        """Setup the manager

        Returns
        -------
        config: dict
            The updated config.
        """

        self._debug_levels = self.update_config()
        self._tools_config = map_tools(self._config.get("tools", []))
        self._relax_mod, self._sample_inputs = None, None
        self._runner = None

    def update_config(self) -> dict:
        """Update config

        Returns
        -------
        debug_levels: dict
            The debug_levels.
        """

        debug_levels = {}
        self._config = self._get_runner_cls(self._model_type).update_config(
            MSCStage.PARSE, self._config, self._model
        )

        # update runner config
        for stage in [MSCStage.BASELINE, MSCStage.OPTIMIZE, MSCStage.COMPILE]:
            if stage not in self._config:
                continue
            if "run_type" not in self._config[stage]:
                self._config[stage]["run_type"] = self._model_type
            runner_cls = self._get_runner_cls(self._config[stage]["run_type"])
            self._config = runner_cls.update_config(stage, self._config, self._model)

        # update tool config
        if self._config.get("tools"):
            self._config["tools"] = self._update_tools_config(self._config["tools"])

        # update export config
        self._config[MSCStage.EXPORT].update(
            {"inputs": self._config["inputs"], "outputs": self._config["outputs"]}
        )

        def _set_debug_level(stage: str, sub_config: dict, default: int = None) -> dict:
            if "debug_level" in sub_config:
                debug_levels[stage] = sub_config["debug_level"]
            elif default is not None:
                debug_levels[stage] = default
                sub_config["debug_level"] = default
            return debug_levels

        if self._verbose.startswith("debug:"):
            debug_level = int(self._verbose.split(":")[1])
        else:
            debug_level = 0
        for stage in [MSCStage.BASELINE, MSCStage.OPTIMIZE, MSCStage.COMPILE]:
            if stage not in self._config:
                continue
            debug_levels = _set_debug_level(stage, self._config[stage]["run_config"], debug_level)
            for t_config in self._config.get("tools", []):
                if not support_tool(t_config, stage, self._config[stage]["run_type"]):
                    continue
                t_stage = stage + "." + get_tool_stage(t_config["tool_type"])
                debug_levels = _set_debug_level(t_stage, t_config["tool_config"], debug_level)
        ordered_keys = [
            "model_type",
            "inputs",
            "outputs",
            "dataset",
            "tools",
            MSCStage.PREPARE,
            MSCStage.PARSE,
            MSCStage.BASELINE,
            MSCStage.OPTIMIZE,
            MSCStage.COMPILE,
            MSCStage.EXPORT,
        ]
        self._config = {k: self._config[k] for k in ordered_keys if k in self._config}
        return debug_levels

    def _update_tools_config(self, tools: List[dict]) -> List[dict]:
        """Update tool in stage config.

        Parameters
        ----------
        tools: list<dict>
            The config of tools.

        Returns
        -------
        tools: list<dict>
            The updated config of tools.
        """

        for tool in tools:
            tool_config = tool["tool_config"]
            if "plan_file" not in tool_config:
                tool_config["plan_file"] = "msc_{}.json".format(tool["tool_type"])
            tool_config["plan_file"] = msc_utils.to_abs_path(
                tool_config["plan_file"], msc_utils.get_config_dir()
            )
        return tools

    def prepare(self, data_loader: Any = None) -> Tuple[dict, dict]:
        """Prepare datas for the pipeline.

        Parameters
        ----------
        data_loader:
            The data loader.

        Returns
        -------
        info: dict
            The info of prepare.
        report: dict
            The report of prepare.
        """

        stage_config = self._config[MSCStage.PREPARE]
        use_cache = self._config.get("use_cache", True)
        runner_cls = self._get_runner_cls(self._model_type)
        run_func = runner_cls.run_native if hasattr(runner_cls, "run_native") else None
        input_names = [i[0] for i in self._config["inputs"]]

        # create golden
        if "golden" in self._config["dataset"]:
            golden_folder = self._config["dataset"]["golden"]["loader"]
        else:
            golden_folder = msc_utils.get_dataset_dir().relpath("Golden", use_cache)
        if msc_utils.is_io_dataset(golden_folder):
            loader, source_type = msc_utils.IODataLoader(golden_folder), "cache"
            self._sample_inputs = loader[0][0]
            datas_info = loader.info
            msg = "Load {} golden from {}".format(len(loader), golden_folder)
            self._logger.debug(self.worker_mark(msg))
        elif run_func:
            source_type = "native"
            saver_options = {"input_names": input_names, "output_names": self._config["outputs"]}
            cnt, max_golden = 0, self._config["dataset"][MSCStage.PREPARE].get("max_golden", 5)
            with msc_utils.IODataSaver(golden_folder, saver_options) as saver:
                for inputs in data_loader():
                    if cnt >= max_golden > 0:
                        break
                    if not self._sample_inputs:
                        self._sample_inputs = {
                            k: msc_utils.cast_array(v) for k, v in inputs.items()
                        }
                    try:
                        outputs, _ = run_func(
                            self._model, inputs, input_names, self._config["outputs"]
                        )
                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        if cnt == 0:
                            msg = "Failed to test native: {}".format(exc)
                            self._logger.warning(self.worker_mark(msg))
                        outputs = None
                    cnt = saver.save_batch(inputs, outputs)
                datas_info = saver.info
            msg = "Save {} golden to {}".format(cnt, golden_folder)
            self._logger.debug(self.worker_mark(msg))
        else:
            raise Exception("golden_folder or runner should given to save golden")
        self._config["dataset"]["golden"] = {"loader": golden_folder, "max_batch": -1}

        def _to_abstract(info: dict) -> dict:
            def _to_tensor_str(info):
                return "{},{}".format(";".join([str(s) for s in info["shape"]]), info["dtype"])

            return {
                "num_datas": info["num_datas"],
                "inputs": {n: _to_tensor_str(i) for n, i in info["inputs"].items()},
                "outputs": {n: _to_tensor_str(o) for n, o in info["outputs"].items()},
            }

        info = {
            "golden_folder({})".format(source_type): golden_folder,
            "datas_info": _to_abstract(datas_info),
            "smaple_inputs": self._sample_inputs,
        }

        # profile
        report = {}
        if "profile" in stage_config and run_func:
            benchmark = stage_config["profile"].get("benchmark", {})
            benchmark["repeat"] = self._get_repeat(benchmark)
            try:
                _, avg_time = run_func(
                    self._model,
                    self._sample_inputs,
                    input_names,
                    self._config["outputs"],
                    **benchmark,
                )
                latency = "{:.2f} ms @ {}".format(avg_time, self._device)
                info["latency"] = latency + " (X{})".format(benchmark["repeat"])
                report["profile"] = latency
            except Exception as exc:  # pylint: disable=broad-exception-caught
                msg = "Failed to profile native: {}".format(exc)
                self._logger.warning(self.worker_mark(msg))
                report["profile"] = "failed run native"
        return info, report

    def parse(self) -> Tuple[dict, dict]:
        """Parse the model to IRModule.

        Returns
        -------
        info: dict
            The info of parse.
        report: dict
            The report of parse.
        """

        stage_config = self._config[MSCStage.PARSE]
        if self._config.get("use_cache", True):
            cache_path = (
                msc_utils.get_cache_dir().create_dir(MSCStage.PARSE).relpath("parsed_relax.json")
            )
        else:
            cache_path = None
        info = {}
        if cache_path and os.path.isfile(cache_path):
            with open(cache_path, "r") as f:
                self._relax_mod = tvm.ir.load_json(f.read())
            info["cache"] = cache_path
        else:
            info = {"parser": stage_config["parser"], "config": stage_config.get("parse_config")}
            parse_config = msc_utils.copy_dict(stage_config.get("parse_config", {}))
            parse_config["as_msc"] = False
            if self._model_type in self._plugins:
                plugin = self._plugins[self._model_type]
                parse_config["custom_convert_map"] = plugin.get_convert_map()
            self._relax_mod, _ = stage_config["parser"](self._model, **parse_config)
            transformed = set()
            for stage in [MSCStage.OPTIMIZE, MSCStage.COMPILE]:
                if stage not in self._config:
                    continue
                run_type = self._config[stage]["run_type"]
                if run_type in transformed:
                    continue
                transformed.add(run_type)
                runner_cls = self._get_runner_cls(run_type)
                if hasattr(runner_cls, "target_transform"):
                    msg = "Transform for {}({})".format(run_type, stage)
                    self._logger.info(self.worker_mark(msg))
                    self._relax_mod = runner_cls.target_transform(self._relax_mod)
            if cache_path:
                with open(cache_path, "w") as f:
                    f.write(tvm.ir.save_json(self._relax_mod))
                msg = "Save parsed mod to " + cache_path
                self._logger.debug(self.worker_mark(msg))
        return info, {}

    def get_tool_config(self, tool_type: str, key: str = "tool_config", default: Any = None) -> Any:
        """Get the tool config

        Parameters
        ----------
        tool_type: str
            The tool type.
        key: str
            The config key

        Returns
        -------
        config:
            The tool config or info.
        """

        assert tool_type in self._tools_config, "Can not find tool_type " + str(tool_type)
        return self._tools_config[tool_type].get(key, default)

    def tool_applied(self, tool_type: str) -> bool:
        """Check if the tool is applied

        Parameters
        ----------
        tool_type: str
            The tool type.

        Returns
        -------
        applied: bool
            Whether the tool is applied.
        """

        config = self.get_tool_config(tool_type)
        return os.path.isfile(config["plan_file"])

    def apply_tool(
        self, tool_type: str, knowledge: dict = None, data_loader: Any = None
    ) -> Tuple[dict, dict]:
        """Apply tool with runner

        Parameters
        ----------
        tool_type: str
            The tool type to apply.
        knowledge: dict
            The pre knowledge.
        data_loader:
            The data loader.

        Returns
        -------
        info: dict
            The info of apply tool.
        report: dict
            The report of apply tool.
        """

        plan_file = self.get_tool_config(tool_type)["plan_file"]
        if knowledge:
            self._logger.info("Plan by %d knowledge for %s", len(knowledge), tool_type)
            msc_utils.save_dict(knowledge, plan_file)
        else:
            self._runner.make_plan(tool_type, data_loader)
        if self.get_tool_config(tool_type, "visualize", False):
            self._runner.visualize(
                msc_utils.get_visual_dir().create_dir(self._runner.stage.split(".")[0])
            )
        report = {}
        if os.path.isfile(plan_file):
            report["plan_num"] = len(msc_utils.load_dict(plan_file))
        return {}, report

    def create_runner(
        self,
        stage: str,
        tools: List[str] = None,
        run_type: str = None,
        run_config: dict = None,
        visualize: bool = True,
        profile: bool = True,
        use_cache: bool = True,
    ) -> Tuple[dict, dict]:
        """Create runner.

        Parameters
        ----------
        stage: str
            The stage name
        tools: list<str>
            The tools to apply.
        run_type: str
            The type of runner.
        run_config: dict
            The config of runner.
        visualize: bool
            Whether to visualize the runner
        profile: bool
            Whether to profile the runner.
        use_cache: bool
            Whether to use cache.

        Returns
        -------
        info: dict
            The info of create runner.
        report: dict
            The report of create runner.
        """

        if self._runner:
            self._runner.destory()
        tools = tools or []
        assert all(t in self._tools_config for t in tools), "Missing some tools " + str(tools)
        main_stage = stage.split(".")[0]
        if not run_type:
            run_type = self._config[main_stage]["run_type"]
        if not run_config:
            run_config = self._config[main_stage].get("run_config", {})
        runner_cls = self._get_runner_cls(run_type)
        if "generate_config" not in run_config:
            run_config["generate_config"] = {}
        cleanup = self._debug_levels.get(stage, 0) == 0
        run_config["generate_config"]["build_folder"] = msc_utils.get_build_dir().create_dir(
            stage, cleanup=cleanup
        )
        if "device" not in run_config:
            run_config["device"] = self._device
        if "training" not in run_config:
            run_config["training"] = self._training
        # Build runner
        runner = runner_cls(
            self._relax_mod,
            tools_config=[self._tools_config[t] for t in tools],
            plugin=self._plugins.get(run_type),
            stage=stage,
            name=self._name,
            logger=self._logger,
            **run_config,
        )
        cache_dir = msc_utils.get_cache_dir().create_dir(stage) if use_cache else None
        runner.build(cache_dir=cache_dir)
        if visualize:
            runner.visualize(msc_utils.get_visual_dir().create_dir(main_stage))
        if use_cache:
            runner.save_cache(cache_dir)
        info, report = {}, {"runtime": "{} @ {}".format(runner.framework, runner.device)}
        if profile and "profile" in self._config[main_stage]:
            profile_config = self._config[main_stage]["profile"]
            info["profile"], report["profile"] = self._profile_runner(runner, profile_config)
        self._runner = runner
        return info, report

    def _profile_runner(self, runner: BaseRunner, profile_config: dict) -> Tuple[dict, str]:
        """Profile the runner.

        Parameters
        ----------
        runner: BaseRunner
            The runner to be profiled
        profile_config: dict
            The config of profile.

        Returns
        -------
        info: dict
            The info of profile.
        report: str
            The report of profile.
        """

        stage = runner.stage
        info, report = {}, ""

        # check accuracy
        check_config = profile_config.get("check", {})
        if check_config:
            loader = msc_utils.IODataLoader(self._config["dataset"]["golden"]["loader"])
            acc_info = {"passed": ""}
            total, passed = 0, 0
            for idx, (inputs, outputs) in enumerate(loader):
                results = runner.run(inputs)
                if outputs:
                    iter_info = msc_utils.compare_arrays(
                        outputs,
                        results,
                        atol=check_config.get("atol", 1e-2),
                        rtol=check_config.get("rtol", 1e-2),
                        report_detail=runner.debug_level >= 2,
                    )
                else:
                    iter_info = {
                        "total": len(results),
                        "passed": len(results),
                        "info": {k: msc_utils.MSCArray(v).abstract() for k, v in results.items()},
                    }
                total += iter_info["total"]
                passed += iter_info["passed"]
                acc_info["iter_" + str(idx)] = iter_info["info"]
            pass_rate = float(passed) / total
            accuracy = "{}/{}({:.2f}%)".format(passed, total, pass_rate * 100)
            acc_info["passed"] = "{} {}".format(accuracy, check_config)
            info["accuracy"] = acc_info if runner.debug_level >= 1 else accuracy
            report = "pass " + accuracy
            if runner.get_tool(ToolType.PRUNER) or runner.get_tool(ToolType.QUANTIZER):
                disable_msg = "Disable accuracy check({}) by tools".format(stage)
                self._logger.debug(self.worker_mark(disable_msg))
            else:
                required_err, err_rate = check_config.get("err_rate", 0), (1 - pass_rate)
                if err_rate > required_err >= 0:
                    self._logger.error(msc_utils.msg_block(self.worker_mark("ACCURACY"), acc_info))
                    raise Exception(
                        "Failed to profile the runner({}), err_rate {} > required {}".format(
                            stage, err_rate, required_err
                        )
                    )

        # benchmark model
        benchmark_config = profile_config.get("benchmark", {})
        if benchmark_config:
            for _ in range(benchmark_config.get("warm_up", 10)):
                runner.run(self._sample_inputs)
            start = time.time()
            repeat = self._get_repeat(benchmark_config, runner.device)
            for _ in range(repeat):
                runner.run(self._sample_inputs)
            avg_time = (time.time() - start) * 1000 / repeat
            latency = "{:.2f} ms @ {}".format(avg_time, runner.device)
            info["latency"] = latency + " (X{})".format(repeat)
            report += (", " if report else "") + latency
        return info, report

    def export_model(self, stage: str, folder: msc_utils.MSCDirectory, dump: bool = True) -> Any:
        """Export the model

        Parameters
        ----------
        stage: str
            The pipeline stage.
        folder: MSCDirectory
            The export folder.
        dump: bool
            Whether to dump info.

        Returns
        -------
        exported:
            The exported model.
        """

        if stage == MSCStage.COMPILE:
            if not dump:
                return self._runner.runnable
            return self._runner.export_runnable(folder)

        if stage == MSCStage.OPTIMIZE:
            module = self._runner.export_module(folder)
            if not dump:
                return module
            path = folder.relpath("model.json")
            with open(path, "w") as f:
                f.write(tvm.ir.save_json(module))
            return path

        if not dump:
            return self._model
        dump_func = self._get_runner_cls(self._model_type).dump_nativate
        return dump_func(self._model, folder, self._config[MSCStage.EXPORT])

    def export_tool(self, tool_type: str, folder: msc_utils.MSCDirectory) -> dict:
        """Export the tool

        Parameters
        ----------
        tool_type: str
            The tool type.
        folder: MSCDirectory
            The export folder.

        Returns
        -------
        config: dict
            The exported tool config.
        """

        run_tool = self._runner.get_tool(tool_type)
        assert tool_type in self._tools_config, "Can not find tool_type " + str(tool_type)
        return run_tool.export_config(self._tools_config[tool_type]["tool_config"], folder)

    def export_info(self, stage: str, folder: msc_utils.MSCDirectory) -> dict:
        """Export the info of worker

        Parameters
        ----------
        stage: str
            The pipeline stage.
        folder: MSCDirectory
            The export folder.

        Returns
        -------
        info: dict
            The info.
        """

        return {
            "visualize": msc_utils.get_visual_dir().copy_to(folder.relpath("visualize")),
            "graphs": self._runner.export_graphs(folder.create_dir("graphs")),
        }

    def get_runnable(self, ret_type: str = "runner") -> Any:
        """Return object by type.

        Parameters
        ----------
        ret_type: str
            The return type runner| runnable| model.

        Returns
        -------
        runnable:
            The runner or model.
        """

        assert self._runner, "Failed to create runner, call run_pipe first"
        if ret_type == "runner":
            return self._runner
        if ret_type == "runnable":
            return self._runner.runnable
        if ret_type == "model":
            return self._runner.model
        raise TypeError("Unexpect return type " + str(ret_type))

    def _get_repeat(self, benchmark: dict, device: str = None) -> int:
        """Get the repeat number for benchmark

        Parameters
        ----------
        benchmark: dict
            The benchmark config.
        device: str
            The device name

        Returns
        -------
        repeat: int
            The repeat number.
        """

        device = device or self._device
        repeat = benchmark.get("repeat", -1)
        if repeat == -1:
            repeat = 500 if device.startswith("cuda") else 10
        return repeat

    def _get_runner_cls(self, run_type: str) -> BaseRunner:
        """Get the runner cls by type

        Parameters
        ----------
        run_type: str
            The run type.

        Returns
        -------
        runner_cls: class
            The runner class.
        """

        raise NotImplementedError("_get_runner_cls is not implemented in " + str(self.__class__))

    def destory(self):
        """Destroy the worker"""

        if self._runner:
            self._runner.destory()

    def worker_mark(self, msg: Any) -> str:
        """Mark the message with worker info

        Parameters
        -------
        msg: str
            The message

        Returns
        -------
        msg: str
            The message with mark.
        """

        return "WORKER[{}] {}".format(self._name, msg)

    @property
    def runner(self):
        return self._runner

    @property
    def model_type(self):
        return self._model_type

    @property
    def optimize_type(self):
        return self._optimize_type

    @property
    def compile_type(self):
        return self._compile_type


class MSCPipeWorker(BasePipeWorker):
    """Normal manager in MSC"""

    def _get_runner_cls(self, run_type: str) -> BaseRunner:
        """Get the runner cls by type

        Parameters
        ----------
        run_type: str
            The run type.

        Returns
        -------
        runner_cls: class
            The runner class.
        """

        if run_type == MSCFramework.TVM:
            from tvm.contrib.msc.framework.tvm.runtime import TVMRunner

            runner_cls = TVMRunner
        elif run_type == MSCFramework.TORCH:
            from tvm.contrib.msc.framework.torch.runtime import TorchRunner

            runner_cls = TorchRunner
        elif run_type == MSCFramework.TENSORFLOW:
            from tvm.contrib.msc.framework.tensorflow.runtime import TensorflowRunner

            runner_cls = TensorflowRunner
        elif run_type == MSCFramework.TENSORRT:
            from tvm.contrib.msc.framework.tensorrt.runtime import TensorRTRunner

            runner_cls = TensorRTRunner
        else:
            raise Exception("Unexpect run_type " + str(run_type))
        return runner_cls
