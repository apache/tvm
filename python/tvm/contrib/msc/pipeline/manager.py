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
# pylint: disable=import-outside-toplevel
"""tvm.contrib.msc.pipeline.manager"""

import os
import time
import json
from typing import Dict, Any
import traceback
import numpy as np

import tvm
from tvm.contrib.msc.core.runtime import BaseRunner
from tvm.contrib.msc.core.tools import ToolType
from tvm.contrib.msc.core.utils.namespace import MSCFramework, MSCMap
from tvm.contrib.msc.core.utils.message import MSCStage
from tvm.contrib.msc.core import utils as msc_utils


class BaseManager(object):
    """Base Manager of MSC

    Parameters
    ----------
    model: Any
        The raw model in framwork.
    config: dict
        The config for pipeline.
    """

    def __init__(self, model, config):
        # check config
        for stage in ["inputs", "outputs", "dataset", "prepare", "compile"]:
            assert stage in config, "{} should be given to run the pipeline".format(stage)
        MSCMap.reset()
        self._model = model
        self._workspace = msc_utils.set_workspace(config.get("workspace"))
        log_path = config.get("log_path") or self._workspace.relpath("MSC_LOG", keep_history=False)
        if config.get("debug_level", 0) > 0 and "verbose" not in config:
            verbose = "debug"
        else:
            verbose = config.get("verbose", "info")
        self._logger = msc_utils.set_global_logger(verbose, log_path)
        msc_utils.time_stamp(MSCStage.SETUP)
        self._logger.info(msc_utils.msg_block("SETUP", self.setup(config)))

    def setup(self, config: dict) -> dict:
        """Setup the manager

        Parameters
        ----------
        config: dict
            The config for manager.

        Returns
        -------
        info: dict
            The setup info.
        """

        self._config, self._debug_levels = self.update_config(config)
        self._tools_config = {}
        self._relax_mod, self._runner = None, None
        self._data_loader, self._sample_inputs = None, None
        self._report = {
            "success": False,
            "info": {
                "workspace": self._workspace.path,
                "model_type": self._config["model_type"],
            },
            "duration": {},
            "profile": {},
        }
        return {"workspace": self._workspace.path, "config": config}

    def update_config(self, config: dict) -> dict:
        """Update config

        Parameters
        ----------
        config: dict
            The config for manager.

        Returns
        -------
        config: dict
            The updated config.
        """

        # update prepare and parse
        assert "inputs" in config, "inputs should be given to run manager"
        assert "outputs" in config, "outputs should be given to run manager"
        config, debug_levels = msc_utils.copy_dict(config), {}
        for stage in ["prepare", "parse"]:
            if stage not in config:
                config[stage] = {}
        config = self._update_prepare_config(config)
        config = self._update_parse_config(config)
        for stage in ["baseline", "optimize", "compile"]:
            config = self._update_runner_config(config, stage)
        config = self._update_tool_config(config)

        def _set_debug_level(stage: str, stage_config: dict, default: int = None) -> dict:
            if "debug_level" in stage_config:
                debug_levels[stage] = stage_config["debug_level"]
            elif default is not None:
                debug_levels[stage] = default
                stage_config["debug_level"] = default
            return debug_levels

        debug_level = config.get("debug_level")
        for stage in ["baseline", "optimize", "compile"]:
            if stage not in config:
                continue
            debug_levels = _set_debug_level(stage, config[stage]["run_config"], debug_level)
        if "optimize" in config:
            for t_type in ToolType.all_types():
                if t_type not in config["optimize"]:
                    continue
                debug_levels = _set_debug_level(
                    self._get_tool_stage(t_type), config["optimize"][t_type], debug_level
                )
        ordered_keys = [
            "model_type",
            "inputs",
            "outputs",
            "dataset",
            "prepare",
            "parse",
            "baseline",
            "optimize",
            "compile",
        ]
        return {k: config[k] for k in ordered_keys if k in config}, debug_levels

    def run_pipe(self) -> dict:
        """Run the pipeline and return object.

        Returns
        -------
        report:
            The pipeline report.
        """

        err_msg = None
        use_cache = self._config.get("use_cache", True)
        try:
            self._data_loader, self._sample_inputs = self.prepare(
                self._config["prepare"], use_cache
            )
            self._relax_mod = self.parse(self._config["parse"], use_cache)
            if "baseline" in self._config:
                self._runner = self.baseline(self._config["baseline"], use_cache)
            if "optimize" in self._config:
                self._runner = self.optimize(self._config["optimize"], use_cache)
            self._runner = self.compile(self._config["compile"], use_cache)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            err_msg = "Pipeline failed:{}\nTrace: {}".format(exc, traceback.format_exc())
        report = self.summary(err_msg)
        self._logger.info(msc_utils.msg_block("SUMMARY", report, 0))
        return report

    def prepare(self, stage_config: dict, use_cache: bool = False) -> Dict[str, np.ndarray]:
        """Prepare datas for the pipeline.

        Parameters
        ----------
        stage_config: dict
            The config of this stage.
        use_cache: bool
            Whether to use cache.

        Returns
        -------
        sample_inputs: dict<str,np.ndarray>
            The sample inputs.
        """

        msc_utils.time_stamp(MSCStage.PREPARE)

        # create data loader
        source_loader = self._config["dataset"].get("loader")
        max_batch = self._config["dataset"].get("max_batch", 5)
        assert source_loader, "Dataset loader should be given for msc pipeline"
        if source_loader.startswith("from_random"):

            def get_random():
                for _ in range(max_batch):
                    yield {i[0]: np.random.rand(*i[1]).astype(i[2]) for i in self._config["inputs"]}

            data_loader, source_type = get_random, "Random"
        elif msc_utils.is_io_dataset(source_loader):

            def load_datas():
                for inputs, _ in msc_utils.IODataLoader(data_loader, end=max_batch):
                    yield inputs

            data_loader, source_type = load_datas, "IOData"
        elif callable(source_loader):

            def get_source():
                for idx, inputs in enumerate(source_loader()):
                    if idx >= max_batch:
                        break
                    yield inputs

            data_loader, source_type = get_source, "Custom"
        else:
            raise TypeError(
                "Unexpected source loader {}({})".format(source_loader, type(source_loader))
            )
        self._logger.info("Create data loader(%s) %s", source_type, data_loader)

        # create golden
        golden_folder = msc_utils.get_dataset_dir().relpath("Golden", use_cache)
        input_names, sample_inputs = [i[0] for i in self._config["inputs"]], None
        report = {"golden_folder": golden_folder}
        runner_cls = self._get_runner_cls(self._config["model_type"])
        run_func = runner_cls.run_native if hasattr(runner_cls, "run_native") else None
        if use_cache and msc_utils.is_io_dataset(golden_folder):
            golden_loader, source_type = msc_utils.IODataLoader(golden_folder), "Cache"
            report["datas_info"] = golden_loader.info
            sample_inputs = golden_loader[0][0]
            self._logger.debug("Load %d cached golden from %s", len(golden_loader), golden_folder)
        else:
            # save golden
            golden_cnt, max_golden = 0, self._config["dataset"].get("max_golden", 5)
            saver_options = {"input_names": input_names, "output_names": self._config["outputs"]}
            if run_func:
                with msc_utils.IODataSaver(golden_folder, saver_options) as saver:
                    for inputs in data_loader():
                        if golden_cnt >= max_golden:
                            break
                        if not sample_inputs:
                            sample_inputs = inputs
                        outputs, _ = run_func(
                            self._model, inputs, input_names, self._config["outputs"]
                        )
                        golden_cnt = saver.save_batch(inputs, outputs)
                    report["datas_info"] = saver.info
            elif isinstance(data_loader, msc_utils.IODataLoader):
                with msc_utils.IODataSaver(golden_folder, saver_options) as saver:
                    for inputs, outputs in data_loader():
                        if golden_cnt >= max_golden:
                            break
                        if not sample_inputs:
                            sample_inputs = inputs
                        golden_cnt = saver.save_batch(inputs, outputs)
                    report["datas_info"] = saver.info
            else:
                raise Exception("golden or runner should given in prepare to save golden")
            self._logger.debug("Saved %d golden to %s", golden_cnt, golden_folder)

        def _to_abstract(info: dict) -> dict:
            def _to_tensor_str(info):
                return "{},{}".format(";".join([str(s) for s in info["shape"]]), info["dtype"])

            return {
                "num_datas": info["num_datas"],
                "inputs": {n: _to_tensor_str(i) for n, i in info["inputs"].items()},
                "outputs": {n: _to_tensor_str(o) for n, o in info["outputs"].items()},
            }

        report["datas_info"] = _to_abstract(report["datas_info"])
        report["sample_inputs"] = sample_inputs
        self._logger.info(msc_utils.msg_block("GOLDEN({})".format(source_type), report))

        # profile
        if "profile" in stage_config and run_func:
            benchmark = stage_config["profile"].get("benchmark", {})
            repeat = benchmark.get("repeat", 100)
            self._logger.debug("Prepare profile with %s(%s)", run_func, benchmark)
            _, avg_time = run_func(
                self._model, sample_inputs, input_names, self._config["outputs"], **benchmark
            )
            self._logger.info("Profile(prepare) {} times -> {:.2f} ms".format(repeat, avg_time))
            self._report["profile"]["prepare"] = {"latency": "{:.2f} ms".format(avg_time)}
        return data_loader, sample_inputs

    def parse(self, stage_config: dict, use_cache: bool = False) -> tvm.IRModule:
        """Parse the model to IRModule.

        Parameters
        ----------
        stage_config: dict
            The config of this stage.
        use_cache: bool
            Whether to use cache.

        Returns
        -------
        relax_mod: tvm.IRModule
            The parsed module.
        """

        msc_utils.time_stamp(MSCStage.PARSE)
        cache_path = msc_utils.get_cache_dir().relpath("parsed_relax.json") if use_cache else None
        if cache_path and os.path.isfile(cache_path):
            with open(cache_path, "r") as f:
                relax_mod = tvm.ir.load_json(f.read())
            self._logger.info("Load parsed mod from %s", cache_path)
        else:
            parse_config = stage_config.get("parse_config", {})
            runner_cls = self._get_runner_cls(self._config["compile"]["run_type"])
            trans_func = (
                runner_cls.target_transform if hasattr(runner_cls, "target_transform") else None
            )
            parse_info = {
                "parser": stage_config["parser"],
                "config": parse_config,
                "trans_func": trans_func,
            }
            self._logger.info(msc_utils.msg_block("PARSE", parse_info))
            relax_mod, _ = stage_config["parser"](self._model, as_msc=False, **parse_config)
            if trans_func:
                relax_mod = trans_func(relax_mod)
            if cache_path:
                with open(cache_path, "w") as f:
                    f.write(tvm.ir.save_json(relax_mod))
                self._logger.debug("Save parsed mod to %s", cache_path)
        return relax_mod

    def baseline(self, stage_config: dict, use_cache: bool = False) -> BaseRunner:
        """Run the baseline.

        Parameters
        ----------
        stage_config: dict
            The config of this stage.
        use_cache: bool
            Whether to use cache.

        Returns
        -------
        runner: BaseRunner
            The runner.
        """

        msc_utils.time_stamp(MSCStage.BASELINE)
        return self._create_runner(MSCStage.BASELINE, stage_config, use_cache=use_cache)

    def optimize(self, stage_config: dict, use_cache: bool = False) -> BaseRunner:
        """Run the optimize and return object.

        Parameters
        ----------
        stage_config: dict
            The config of this stage.
        use_cache: bool
            Whether to use cache.

        Returns
        -------
        runner: BaseRunner
            The runner.
        """

        runner_cls = self._get_runner_cls(stage_config["run_type"])

        def _tool_enabled(tool_type: str) -> bool:
            return tool_type in stage_config and runner_cls.support_tool(tool_type)

        # run prune
        if _tool_enabled(ToolType.PRUNER):
            self._apply_tool(ToolType.PRUNER, stage_config)

        # optimize and get the runner
        msc_utils.time_stamp(MSCStage.OPTIMIZE)
        return self._create_runner(
            MSCStage.OPTIMIZE, stage_config, tools_config=self._tools_config, use_cache=use_cache
        )

    def compile(self, stage_config: dict, use_cache: bool = False) -> BaseRunner:
        """Run the compile and return object.

        Parameters
        ----------
        stage_config: dict
            The config of this stage.
        use_cache: bool
            Whether to use cache.
        ret_type: str
            The return type runner| model.

        Returns
        -------
        runner: BaseRunner
            The runner.
        """

        msc_utils.time_stamp(MSCStage.COMPILE)
        return self._create_runner(
            MSCStage.COMPILE, stage_config, tools_config=self._tools_config, use_cache=use_cache
        )

    def summary(self, err_msg=None):
        """Summary the pipeline.

        Parameters
        ----------
        err_msg: str
            The error message.

        Returns
        -------
        report: dict
            The report of the pipeline.
        """

        msc_utils.time_stamp(MSCStage.SUMMARY, False)
        if err_msg:
            self._report.update({"success": False, "err_msg": err_msg})
        else:
            self._report["success"] = True
        self._report["duration"] = msc_utils.get_duration()
        return self._report

    def destory(self, keep_workspace: bool = False):
        """Destroy the manager

        Parameters
        ----------
        keep_workspace: bool
            Whether to keep workspace.
        """

        if self._runner:
            self._runner.destory()
        if not keep_workspace:
            self._workspace.destory()

    def _create_runner(
        self,
        stage: str,
        stage_config: dict,
        tools_config: dict = None,
        visualize: bool = True,
        profile: bool = True,
        use_cache: bool = True,
    ) -> BaseRunner:
        """Create runner.

        Parameters
        ----------
        stage: str
            The stage name
        stage_config: dict
            The config of this stage.
        tools_config: dict
            The config of the tools
        visualize: bool
            Whether to visualize the runner
        profile: bool
            Whether to profile the runner.
        use_cache: bool
            Whether to use cache.

        Returns
        -------
        runner: BaseRunner
            The runner.
        """

        if self._runner:
            self._runner.destory()
        cache_dir = msc_utils.get_cache_dir().create_dir(stage) if use_cache else None
        tools_config = tools_config or {}
        msc_utils.time_stamp(stage + ".build", False)
        runner_cls = self._get_runner_cls(stage_config["run_type"])
        run_config = msc_utils.copy_dict(stage_config.get("run_config"))
        if "generate_config" not in run_config:
            run_config["generate_config"] = {}
        cleanup = self._debug_levels.get(stage, 0) == 0
        run_config["generate_config"]["build_folder"] = msc_utils.get_build_dir().create_dir(
            stage, cleanup=cleanup
        )
        opt_config = self._config.get("optimize", {})
        if ToolType.TRACKER in opt_config and runner_cls.support_tool(ToolType.TRACKER):
            tools_config = {**tools_config, ToolType.TRACKER: opt_config[ToolType.TRACKER]}
        # Build runner
        runner = runner_cls(
            self._relax_mod,
            tools_config=tools_config,
            stage=stage,
            logger=self._logger,
            **run_config,
        )
        runner.build(cache_dir=cache_dir)
        self._report["info"][stage + "_by"] = "{}({})".format(runner.framework, runner.device)
        if visualize:
            runner.visualize(msc_utils.get_visual_dir().create_dir(stage))
        if profile and "profile" in stage_config:
            self._report["profile"][stage] = self._profile_runner(runner, stage_config)
        if use_cache:
            runner.save_cache(cache_dir)
        if runner.get_tool(ToolType.TRACKER):
            runner.apply_tool(ToolType.TRACKER)
        return runner

    def _apply_tool(self, tool_type: str, stage_config: dict, add_tool: bool = True) -> str:
        """Apply tool with runner

        Parameters
        ----------
        tool_type: str
            The tool type.
        stage_config: dict
            The config of this stage.
        add_tool: bool
            Whether to add tool in self._tools.

        Returns
        -------
        plan_file: str
            The plan_file path.
        """

        assert tool_type in stage_config, "Can not find config for tool " + str(tool_type)
        tool_stage, tool_config = self._get_tool_stage(tool_type), stage_config[tool_type]
        plan_file = tool_config["plan_file"]
        if "gym_configs" in tool_config:
            gym_configs = tool_config.pop("gym_configs")
        else:
            gym_configs = None
        if add_tool:
            self._tools_config[tool_type] = tool_config
            tools_config = self._tools_config
        else:
            tools_config = {**self._tools_config, tool_type: tool_config}
        if os.path.isfile(plan_file):
            self._logger.info("Skip %s with plan %s", tool_type, plan_file)
            return plan_file
        msc_utils.time_stamp(tool_stage)
        t_stage_config = {
            "run_type": stage_config["run_type"],
            "run_config": stage_config["run_config"],
        }
        runner = self._create_runner(
            tool_stage, t_stage_config, tools_config=tools_config, profile=False, use_cache=False
        )
        if gym_configs:
            raise NotImplementedError("Gym is not implemented")
        return runner.apply_tool(tool_type, self._data_loader)

    def _profile_runner(self, runner: BaseRunner, stage_config: str) -> dict:
        """Profile the runner.

        Parameters
        ----------
        runner: BaseRunner
            The runner to be profiled
        stage_config: dict
            The config of this stage.

        Returns
        -------
        report: dict
            The profile report.
        """

        stage = runner.stage
        msc_utils.time_stamp(stage + ".profile", False)
        profile_config = stage_config["profile"]
        msg, report = "Profile({})".format(stage), {}

        # check accuracy
        check_config = profile_config.get("check", {})
        if check_config:
            loader = msc_utils.IODataLoader(msc_utils.get_dataset_dir().relpath("Golden"))
            total, passed = 0, 0
            acc_report = {"config": check_config}
            for idx, (inputs, outputs) in enumerate(loader):
                results = runner.run(inputs)
                iter_report = msc_utils.compare_arrays(
                    outputs,
                    results,
                    atol=check_config.get("atol", 1e-2),
                    rtol=check_config.get("rtol", 1e-2),
                )
                total += iter_report["total"]
                passed += iter_report["passed"]
                acc_report["iter_" + str(idx)] = iter_report["info"]
            pass_rate = float(passed) / total
            report["accuracy"] = "{}/{}({:.2f}%)".format(passed, total, pass_rate * 100)
            title = "Check({}) pass {}".format(stage, report["accuracy"])
            self._logger.debug(msc_utils.msg_block(title, acc_report, width=0))
            msg += " acc {} iters -> {}".format(len(loader), report["accuracy"])
            if runner.get_tool(ToolType.PRUNER) or runner.get_tool(ToolType.QUANTIZER):
                self._logger.debug("Disable accuracy check(%s) by tools", stage)
            else:
                required_err, err_rate = check_config.get("err_rate", 0), (1 - pass_rate)
                if err_rate > required_err >= 0:
                    raise Exception(
                        "Failed to profile the runner({}), err_rate {} > required {}".format(
                            stage, err_rate, required_err
                        )
                    )

        # benchmark model
        if runner.get_tool(ToolType.TRACKER):
            benchmark_config = None
            self._logger.debug("Disable benchmark(%s) by tools", stage)
        else:
            benchmark_config = profile_config.get("benchmark", {})
        if benchmark_config:
            for _ in range(benchmark_config.get("warm_up", 10)):
                runner.run(self._sample_inputs)
            start = time.time()
            repeat = benchmark_config.get("repeat", 100)
            for _ in range(repeat):
                runner.run(self._sample_inputs)
            avg_time = (time.time() - start) * 1000 / repeat
            report["latency"] = "{:.2f} ms @ {}".format(avg_time, runner.device)
            msg += " latency {} times -> {}".format(repeat, report["latency"])
        self._logger.info(msg)
        return report

    def _update_prepare_config(self, config: dict) -> dict:
        """Update prepare in stage config.

        Parameters
        ----------
        config: dict
            The config of a pipeline.

        Returns
        -------
        config: dict
            The updated config.
        """

        if config["model_type"] == MSCFramework.TORCH:
            import torch

            assert isinstance(
                self._model, torch.nn.Module
            ), "Model for torch should be nn.Module, get {}({})".format(
                self._model, type(self._model)
            )
        elif config["model_type"] == MSCFramework.TENSORFLOW:
            from tvm.contrib.msc.framework.tensorflow import tf_v1

            assert isinstance(
                self._model, tf_v1.GraphDef
            ), "Model for tenosrflow should be tf.GraphDef, get {}({})".format(
                self._model, type(self._model)
            )
        else:
            raise Exception("Unexpect model_type " + str(config["model_type"]))
        return config

    def _update_parse_config(self, config: dict) -> dict:
        """Update parse in stage config.

        Parameters
        ----------
        config: dict
            The config of a pipeline.

        Returns
        -------
        config: dict
            The updated config.
        """

        if config["model_type"] == MSCFramework.TORCH:
            from tvm.contrib.msc.framework.torch.frontend import from_torch

            config["parse"]["parser"] = from_torch
            parse_config = config["parse"].get("parse_config", {})
            parse_config.update(
                {
                    "input_info": [[i[1], i[2]] for i in config["inputs"]],
                    "input_names": [i[0] for i in config["inputs"]],
                }
            )
            config["parse"]["parse_config"] = parse_config
        elif config["model_type"] == MSCFramework.TENSORFLOW:
            from tvm.contrib.msc.framework.tensorflow.frontend import from_tensorflow

            config["parse"]["parser"] = from_tensorflow
            parse_config = config["parse"].get("parse_config", {})
            parse_config.update(
                {
                    "shape_dict": {i[0]: i[1] for i in config["inputs"]},
                    "outputs": config["outputs"],
                }
            )
            config["parse"]["parse_config"] = parse_config
        else:
            raise Exception("Unexpect model_type " + str(config["model_type"]))
        return config

    def _update_runner_config(self, config: dict, stage: str) -> dict:
        """Update runtime stage in stage config.

        Parameters
        ----------
        config: dict
            The config of a pipeline.
        stage: str
            The stage to be updated
        """

        if stage not in config:
            return config
        model_type = config["model_type"]
        if "run_type" not in config[stage]:
            config[stage]["run_type"] = model_type
        # update run config
        run_config = config[stage].get("run_config", {})
        if "translate_config" not in run_config:
            run_config["translate_config"] = {}
        if "build" not in run_config["translate_config"]:
            run_config["translate_config"]["build"] = {}
        if "generate_config" not in run_config:
            run_config["generate_config"] = {}
        run_config["translate_config"]["build"]["input_aliases"] = [i[0] for i in config["inputs"]]
        run_config["translate_config"]["build"]["output_aliases"] = config["outputs"]
        if model_type == MSCFramework.TORCH:
            parameters = list(self._model.parameters())
            if parameters:
                ref_device = parameters[0].device
                if ref_device.type == "cpu":
                    device = "cpu"
                else:
                    device = "{}:{}".format(ref_device.type, ref_device.index)
            else:
                device = "cpu"
            run_config.update({"device": device, "is_training": self._model.training})
        if config[stage]["run_type"] == MSCFramework.TENSORRT:
            if "extra_option" not in run_config["generate_config"]:
                run_config["generate_config"]["extra_option"] = {}
            run_config["generate_config"]["extra_option"]["stage"] = stage
        config[stage]["run_config"] = run_config
        return config

    def _update_tool_config(self, config: dict) -> dict:
        """Update tool in stage config.

        Parameters
        ----------
        config: dict
            The config of a pipeline.

        Returns
        -------
        config: dict
            The updated config.
        """

        if "optimize" not in config:
            return config
        for tool_type in ToolType.all_types():
            if tool_type not in config["optimize"]:
                continue
            tool_config = config["optimize"][tool_type]
            if "plan_file" not in tool_config:
                tool_config["plan_file"] = "msc_{}.json".format(tool_type)
            tool_config["plan_file"] = msc_utils.to_abs_path(
                tool_config["plan_file"], msc_utils.get_config_dir()
            )
        return config

    def _get_tool_stage(self, tool_type: str) -> str:
        """Map the stage according to tool_type

        Parameters
        ----------
        tool_type: str
            The tool type.

        Returns
        -------
        stage: str
            The stage.
        """

        if tool_type == ToolType.PRUNER:
            return MSCStage.PRUNE
        if tool_type == ToolType.QUANTIZER:
            return MSCStage.QUANTIZE
        if tool_type == ToolType.DISTILLER:
            return MSCStage.DISTILL
        return tool_type

    def get_runnable(self, ret_type: str = "runner") -> Any:
        """Return object by type.

        Parameters
        ----------
        ret_type: str
            The return type runner| model.

        Returns
        -------
        runnable:
            The runner or model.
        """

        if ret_type == "runner":
            return self._runner
        elif ret_type == "runnable":
            return self._runner.runnable
        elif ret_type == "model":
            return self._runner.model
        raise Exception("Unexpect return type " + str(ret_type))

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

        raise NotImplementedError("_get_runner_cls is not implemented for BaseManager")

    @property
    def runner(self):
        return self._runner


class MSCManager(BaseManager):
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
