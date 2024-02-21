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
from tvm.contrib.msc.core import transform as msc_transform
from tvm.contrib.msc.core.runtime import BaseRunner
from tvm.contrib.msc.core.tools import ToolType
from tvm.contrib.msc.core.utils.namespace import MSCFramework, MSCMap, MSCKey
from tvm.contrib.msc.core.utils.message import MSCStage
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.core.gym.control import create_controller
from tvm.contrib.msc.core import _ffi_api
from tvm.contrib.msc.plugin.utils import load_plugins


class BaseManager(object):
    """Base Manager of MSC

    Parameters
    ----------
    model: Any
        The raw model in framwork.
    config: dict
        The config for pipeline.
    plugins: dict
        The plugins for pipeline.
    root: str
        The root path for files.
    """

    def __init__(self, model: Any, config: dict, plugins: dict = None, root: str = None):
        # change path to root path
        if root:

            def _from_root_mark(val):
                if root and isinstance(val, str) and MSCKey.ROOT_MARK in val:
                    return val.replace(MSCKey.ROOT_MARK, root)
                return val

            model = _from_root_mark(model)
            config = msc_utils.map_dict(config, _from_root_mark)
            plugins = msc_utils.map_dict(plugins, _from_root_mark)

        # check stage
        for stage in ["inputs", "outputs", "dataset", MSCStage.PREPARE, MSCStage.COMPILE]:
            assert stage in config, "{} should be given to run the pipeline".format(stage)

        MSCMap.reset()
        self._model_type = config["model_type"]
        self._model, self._device, self._training = self._get_runner_cls(
            self._model_type
        ).load_native(model)
        if plugins:
            self._plugins = load_plugins(plugins)
        else:
            self._plugins = {}
        use_cache = config.get("use_cache", True)
        self._workspace = msc_utils.set_workspace(config.get("workspace"), use_cache)
        self._verbose = config.get("verbose", "info")
        if "logger" in config:
            self._logger = config["logger"]
            MSCMap.set(MSCKey.GLOBALE_LOGGER, self._logger)
        else:
            log_path = config.get("log_path") or self._workspace.relpath(
                "MSC_LOG", keep_history=False
            )
            self._logger = msc_utils.set_global_logger(self._verbose, log_path)
        self._optimized, self._compiled = False, False
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

        self._meta_config = config
        self._optimize_type = config.get(MSCStage.OPTIMIZE, {}).get("run_type", self._model_type)
        self._compile_type = config.get(MSCStage.COMPILE, {}).get("run_type", self._model_type)
        # register plugins
        if self._plugins:
            for t in [self._model_type, self._optimize_type, self._compile_type]:
                assert t in self._plugins, "Missing plugin for {}".format(t)
            for name, plugin in self._plugins[self._model_type].get_ops_info().items():
                _ffi_api.RegisterPlugin(name, msc_utils.dump_dict(plugin))
        self._config, self._debug_levels = self.update_config(config)
        self._tools_config = {}
        self._relax_mod, self._runner = None, None
        self._sample_inputs = None
        self._report = {
            "success": False,
            "info": {
                "workspace": self._workspace.path,
                "model_type": "{}({})".format(self._model_type, self._device),
            },
            "duration": {},
            "profile": {},
        }
        return {"workspace": self._workspace.path, "plugins": self._plugins, "config": config}

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
        for stage in [MSCStage.PREPARE, MSCStage.PARSE]:
            if stage not in config:
                config[stage] = {}
        config = self._get_runner_cls(self._model_type).update_config(
            MSCStage.PARSE, config, self._model
        )
        for stage in [MSCStage.BASELINE, MSCStage.OPTIMIZE, MSCStage.COMPILE]:
            if stage not in config:
                continue
            if "run_type" not in config[stage]:
                config[stage]["run_type"] = self._model_type
            config = self._get_runner_cls(config[stage]["run_type"]).update_config(
                stage, config, self._model
            )
        if MSCStage.OPTIMIZE in config:
            config[MSCStage.OPTIMIZE] = self._update_tool_config(config[MSCStage.OPTIMIZE])

        def _set_debug_level(stage: str, stage_config: dict, default: int = None) -> dict:
            if "debug_level" in stage_config:
                debug_levels[stage] = stage_config["debug_level"]
            elif default is not None:
                debug_levels[stage] = default
                stage_config["debug_level"] = default
            return debug_levels

        if self._verbose.startswith("debug:"):
            debug_level = int(self._verbose.split(":")[1])
        else:
            debug_level = 0
        for stage in [MSCStage.BASELINE, MSCStage.OPTIMIZE, MSCStage.COMPILE]:
            if stage not in config:
                continue
            debug_levels = _set_debug_level(stage, config[stage]["run_config"], debug_level)
        if MSCStage.OPTIMIZE in config:
            for t_type in ToolType.all_types():
                if t_type not in config[MSCStage.OPTIMIZE]:
                    continue
                debug_levels = _set_debug_level(
                    self._get_tool_stage(t_type), config[MSCStage.OPTIMIZE][t_type], debug_level
                )
        ordered_keys = [
            "model_type",
            "inputs",
            "outputs",
            "dataset",
            MSCStage.PREPARE,
            MSCStage.PARSE,
            MSCStage.BASELINE,
            MSCStage.OPTIMIZE,
            MSCStage.COMPILE,
        ]
        return {k: config[k] for k in ordered_keys if k in config}, debug_levels

    def run_pipe(self, run_optimize: bool = True, run_compile: bool = True) -> dict:
        """Run the pipeline and return object.

        Parameters
        ----------
        run_optimize: bool
            Whether to run the optimize.
        run_compile: bool
            Whether to run the compile.

        Returns
        -------
        report:
            The pipeline report.
        """

        err_msg = None
        try:
            self.prepare()
            self.parse()
            if MSCStage.BASELINE in self._config:
                self.baseline()
            if run_optimize and MSCStage.OPTIMIZE in self._config:
                self.optimize()
            if run_compile:
                self.compile()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            err_msg = "Pipeline failed:{}\nTrace: {}".format(exc, traceback.format_exc())
        self.summary(err_msg)
        self._logger.info(msc_utils.msg_block("SUMMARY", self._report, 0))
        return self._report

    def prepare(self) -> Dict[str, np.ndarray]:
        """Prepare datas for the pipeline.

        Returns
        -------
        dataloader:
            The dataloader
        sample_inputs: dict<str,np.ndarray>
            The sample inputs.
        """

        msc_utils.time_stamp(MSCStage.PREPARE)
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
        report = {"golden_folder": golden_folder}
        if msc_utils.is_io_dataset(golden_folder):
            loader, source_type = msc_utils.IODataLoader(golden_folder), "Cache"
            self._sample_inputs = loader[0][0]
            report["datas_info"] = loader.info
            self._logger.debug("Load %d golden from %s", len(loader), golden_folder)
        elif run_func:
            loader, source_type = self._get_loader(MSCStage.PREPARE), "Native"
            saver_options = {"input_names": input_names, "output_names": self._config["outputs"]}
            cnt, max_golden = 0, self._config["dataset"][MSCStage.PREPARE].get("max_golden", 5)
            with msc_utils.IODataSaver(golden_folder, saver_options) as saver:
                for inputs in loader():
                    if cnt >= max_golden > 0:
                        break
                    if not self._sample_inputs:
                        self._sample_inputs = inputs
                    outputs, _ = run_func(self._model, inputs, input_names, self._config["outputs"])
                    cnt = saver.save_batch(inputs, outputs)
                report["datas_info"] = saver.info
            self._logger.debug("Saved %d golden to %s", cnt, golden_folder)
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

        report["datas_info"] = _to_abstract(report["datas_info"])
        report["sample_inputs"] = self._sample_inputs
        self._logger.info(msc_utils.msg_block("GOLDEN({})".format(source_type), report))

        # profile
        if "profile" in stage_config and run_func:
            benchmark = stage_config["profile"].get("benchmark", {})
            benchmark["repeat"] = self._get_repeat(benchmark)
            self._logger.debug("Prepare profile with %s(%s)", run_func, benchmark)
            _, avg_time = run_func(
                self._model, self._sample_inputs, input_names, self._config["outputs"], **benchmark
            )
            msg = "{:.2f} ms @ {}".format(avg_time, self._device)
            self._report["profile"][MSCStage.PREPARE] = {"latency": msg}
            self._logger.info("Profile(prepare) %d times -> %s", benchmark["repeat"], msg)

        return self._sample_inputs

    def parse(self) -> tvm.IRModule:
        """Parse the model to IRModule.

        Returns
        -------
        relax_mod: tvm.IRModule
            The parsed module.
        """

        msc_utils.time_stamp(MSCStage.PARSE)
        stage_config = self._config[MSCStage.PARSE]
        use_cache = self._config.get("use_cache", True)

        cache_path = msc_utils.get_cache_dir().relpath("parsed_relax.json") if use_cache else None
        if cache_path and os.path.isfile(cache_path):
            with open(cache_path, "r") as f:
                self._relax_mod = tvm.ir.load_json(f.read())
            self._logger.info("Load parsed mod from %s", cache_path)
        else:
            parse_config = msc_utils.copy_dict(stage_config.get("parse_config", {}))
            parse_info = {"parser": stage_config["parser"], "config": parse_config}
            self._logger.info(msc_utils.msg_block("PARSE", parse_info))
            parse_config["as_msc"] = False
            if self._model_type in self._plugins:
                plugin = self._plugins[self._model_type]
                parse_config["custom_convert_map"] = plugin.get_convert_map()
            self._relax_mod, _ = stage_config["parser"](self._model, **parse_config)
            for stage in [MSCStage.OPTIMIZE, MSCStage.COMPILE]:
                if stage not in self._config:
                    continue
                runner_cls = self._get_runner_cls(self._config[stage]["run_type"])
                if hasattr(runner_cls, "target_transform"):
                    self._logger.info(
                        "Transform for stage %s: %s", stage, runner_cls.target_transform
                    )
                    self._relax_mod = runner_cls.target_transform(self._relax_mod)
            self._relax_mod = msc_transform.SetExprName()(self._relax_mod)
            if cache_path:
                with open(cache_path, "w") as f:
                    f.write(tvm.ir.save_json(self._relax_mod))
                self._logger.debug("Save parsed mod to %s", cache_path)
        return self._relax_mod

    def baseline(self) -> BaseRunner:
        """Run the baseline.

        Returns
        -------
        runner: BaseRunner
            The runner.
        """

        msc_utils.time_stamp(MSCStage.BASELINE)
        self._runner = self._create_runner(
            MSCStage.BASELINE,
            self._config[MSCStage.BASELINE],
            use_cache=self._config.get("use_cache", True),
        )
        return self._runner

    def optimize(self) -> BaseRunner:
        """Run the optimize and return object.

        Returns
        -------
        runner: BaseRunner
            The runner.
        """

        stage_config = self._config[MSCStage.OPTIMIZE]
        self.apply_tools(stage_config)
        msc_utils.time_stamp(MSCStage.OPTIMIZE)
        self._runner = self._create_runner(
            MSCStage.OPTIMIZE,
            stage_config,
            tools_config=self._tools_config,
            use_cache=self._config.get("use_cache", True),
        )
        self._optimized = True
        return self._runner

    def compile(self) -> BaseRunner:
        """Run the compile and return object.

        Returns
        -------
        runner: BaseRunner
            The runner.
        """

        stage_config = self._config[MSCStage.COMPILE]
        self.apply_tools(stage_config)
        msc_utils.time_stamp(MSCStage.COMPILE)
        self._runner = self._create_runner(
            MSCStage.COMPILE,
            stage_config,
            tools_config=self._tools_config,
            use_cache=self._config.get("use_cache", True),
        )
        self._compiled = True
        return self._runner

    def apply_tools(self, stage_config: dict):
        """Apply tools for a stage.

        Parameters
        ----------
        stage_config: dict
            The config of this stage.
        """

        runner_cls = self._get_runner_cls(stage_config["run_type"])

        def _tool_enabled(tool_type: str) -> bool:
            return tool_type in stage_config and runner_cls.support_tool(tool_type)

        # run prune
        if _tool_enabled(ToolType.PRUNER):
            self._apply_tool(ToolType.PRUNER, stage_config)

        # run quantize
        if _tool_enabled(ToolType.QUANTIZER):
            self._apply_tool(ToolType.QUANTIZER, stage_config)

        # run distill
        if _tool_enabled(ToolType.DISTILLER):
            self._apply_tool(ToolType.DISTILLER, stage_config)

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
        if "device" not in run_config:
            run_config["device"] = self._device
        if "training" not in run_config:
            run_config["training"] = self._training
        opt_config = self._config.get(MSCStage.OPTIMIZE, {})
        if ToolType.TRACKER in opt_config and runner_cls.support_tool(ToolType.TRACKER):
            tools_config = {**tools_config, ToolType.TRACKER: opt_config[ToolType.TRACKER]}
        # Build runner
        runner = runner_cls(
            self._relax_mod,
            tools_config=tools_config,
            plugin=self._plugins.get(stage_config["run_type"]),
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
        if "run_type" in tool_config:
            run_type = tool_config.pop("run_type")
        else:
            run_type = stage_config["run_type"]
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
        t_stage_config = {"run_type": run_type, "run_config": stage_config["run_config"]}
        runner = self._create_runner(
            tool_stage, t_stage_config, tools_config=tools_config, profile=False, use_cache=False
        )
        if gym_configs:
            knowledge = None
            for idx, config in enumerate(gym_configs):
                self._logger.info("GYM[%d/%d].CREATE(%s)", idx, len(gym_configs), tool_stage)
                extra_config = {
                    "env": {
                        "runner": runner,
                        "data_loader": self._get_loader(tool_stage),
                        "knowledge": knowledge,
                    },
                    "verbose": self._verbose,
                }
                controller = create_controller(runner.stage, config, extra_config)
                knowledge = controller.run()
            with open(plan_file, "w") as f:
                f.write(json.dumps(knowledge, indent=2))
            self._logger.info(
                "Gym save %d knowledge(%s) -> %s", len(knowledge), tool_type, plan_file
            )
            return plan_file
        return runner.apply_tool(tool_type, self._get_loader(tool_stage))

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
            loader = msc_utils.IODataLoader(self._config["dataset"]["golden"]["loader"])
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
            repeat = self._get_repeat(benchmark_config, runner.device)
            for _ in range(repeat):
                runner.run(self._sample_inputs)
            avg_time = (time.time() - start) * 1000 / repeat
            report["latency"] = "{:.2f} ms @ {}".format(avg_time, runner.device)
            msg += " latency {} times -> {}".format(repeat, report["latency"])
        self._logger.info(msg)
        return report

    def _update_tool_config(self, opt_config: dict) -> dict:
        """Update tool in stage config.

        Parameters
        ----------
        opt_config: dict
            The config of optimize.

        Returns
        -------
        config: dict
            The updated config of optimize.
        """

        for tool_type in ToolType.all_types():
            if tool_type not in opt_config:
                continue
            tool_config = opt_config[tool_type]
            if "plan_file" not in tool_config:
                tool_config["plan_file"] = "msc_{}.json".format(tool_type)
            tool_config["plan_file"] = msc_utils.to_abs_path(
                tool_config["plan_file"], msc_utils.get_config_dir()
            )
        return opt_config

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

    def _get_loader(self, name: str = MSCStage.PREPARE) -> Any:
        """Get the data loader"""

        config = self._config["dataset"].get(name, self._config["dataset"][MSCStage.PREPARE])
        source_loader = config.get("loader")
        max_batch = config.get("max_batch", 5)
        assert source_loader, "Dataset loader should be given for msc pipeline"
        if source_loader == "from_random":
            max_batch = max(max_batch, 5)

            def get_random():
                for _ in range(max_batch):
                    yield {i[0]: np.random.rand(*i[1]).astype(i[2]) for i in self._config["inputs"]}

            loader, source_type = get_random, "Random"
        elif msc_utils.is_io_dataset(source_loader):

            def load_datas():
                for inputs, _ in msc_utils.IODataLoader(source_loader, end=max_batch):
                    yield inputs

            loader, source_type = load_datas, "IOData"
        elif callable(source_loader):

            def get_source():
                for idx, inputs in enumerate(source_loader()):
                    if idx >= max_batch > 0:
                        break
                    yield inputs

            loader, source_type = get_source, "Custom"
        else:
            raise TypeError(
                "Unexpected source loader {}({})".format(source_loader, type(source_loader))
            )
        self._logger.debug("Create data loader(%s) %s(%s)", name, loader, source_type)
        return loader

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

    @property
    def runner(self):
        return self._runner

    @property
    def report(self):
        return self._report

    @property
    def model_type(self):
        return self._model_type

    @property
    def optimize_type(self):
        return self._optimize_type

    @property
    def compile_type(self):
        return self._compile_type


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
