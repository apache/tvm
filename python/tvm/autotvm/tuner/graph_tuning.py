"""Utilities for tuning a whole graph (a set of tasks)"""

import os
from . import callback, XGBTuner, GATuner, RandomTuner, GridSearchTuner
from .. import measure, record, task

def tune_tasks(tasks,
               rpc_device_key,

               tuner='ga',
               n_trial=500,
               early_stopping=200,
               log_filename='tuning.log',

               mea_number=5,
               mea_parallel_num=1,
               mea_timeout=20,
               mea_use_ndk=False,

               use_transfer_learning=True):
    """
    Tune a set of tasks

    Parameters
    ----------
    tasks: Array of Task
        A list of tasks to tune
    rpc_device_key: str
        The key of devices in rpc tracker
    tuner: str
        The type of tuner.
        If is 'xgb', use :any:`XGBTuner`.
        If is 'ga', use :any:`GATuner`.
        If is 'random', use :any:`RandomTuner`.
        If is 'gridsearch', use :any:`GridSearchTuner`.
    n_trial: int
        The maximum number of trials for a workload
    early_stopping: int
        The early stopping metric. The tuner will stop when it cannot find better
        config after `early_stopping` trials
    log_filename: str
        The filename of output log file to store best configs
    mea_number: int
        The number of runs for taking average for one measurement.
    mea_parallel_num: int
        The parallel number in measurement. Set this to the number of devices you have.
    mea_timeout: int
        The timeout of a measurement.
    mea_use_ndk: bool
        Whether use Android NDK. The this to true if your target is android system
    use_transfer_learning: bool
        Whether reuse history tuning log to accelerate tuning
    """

    for i in range(len(tasks)):  # pylint:disable=consider-using-enumerate
        try:  # try winograd template
            tsk = task.create(tasks[i].name, tasks[i].args,
                              tasks[i].target, tasks[i].target_host,
                              'winograd')
            tasks.append(tsk)
        except Exception:  # pylint:disable=broad-except
            pass

    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(tasks):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))

        measure_option = measure.measure_option(mode='rpc',
                                                repeat=3,
                                                number=mea_number,
                                                rpc_device_key=rpc_device_key,
                                                parallel_num=mea_parallel_num,
                                                timeout=mea_timeout,
                                                use_ndk=mea_use_ndk)

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank', verbose=0)
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(record.load_from_file(tmp_log_file))

        # do tuning
        tuner_obj.tune(n_trial=min(n_trial, len(tsk.config_space)),
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       verbose=0,
                       callbacks=[
                           callback.progress_bar(n_trial, prefix=prefix),
                           callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)
