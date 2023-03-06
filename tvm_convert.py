import onnx
import os
import pickle
import tvm
import tvm.relay as relay
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner
import tvm.auto_scheduler as auto_scheduler


def convert_onnx_to_relay(model_path: str):
    shape_dict = {"input_ids": (1, 128)}
    onnx_model = onnx.load(model_path)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    return mod, params


def save_mod_params(mod, params, save_path: str):
    mod_path = os.path.join(save_path, "mod.pkl")
    params_path = os.path.join(save_path, "params.pkl")
    with open(mod_path, "wb") as f:
        pickle.dump(mod, f, pickle.HIGHEST_PROTOCOL)
    with open(params_path, "wb") as f:
        pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)


def load_mod_params(save_path: str):
    mod_path = os.path.join(save_path, "mod.pkl")
    params_path = os.path.join(save_path, "params.pkl")
    with open(mod_path, "rb") as f:
        mod = pickle.load(f)
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    return mod, params


def build_tvm_lib(mod, params, target: str):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target, params=params)
    return lib


def save_tvm_lib(lib, lib_path: str):
    lib.export_library(lib_path)


def load_tvm_lib(lib_path: str):
    loaded_lib = tvm.runtime.load_module(lib_path)
    return loaded_lib


def build_tuned_lib(mod, params, target):
    number = 10  # 一次实验中测速的次数
    repeat = 1  # 重复实验次数
    min_repeat_ms = 0  # 调优 CPU 时设置为 0
    timeout = 100  # 编译时间超时（单位：秒）

    # 创建 TVM 运行器
    runner = autotvm.LocalRunner(
        number=number,
        repeat=repeat,
        timeout=timeout,
        min_repeat_ms=min_repeat_ms,
        enable_cpu_cache_flush=True,
    )
    # 如果要在投产的项目中应用，则需要将试验次数设置为大于此处的 20。
    # 对于 CPU 推荐 1500，对于 GPU 推荐 3000-4000。
    tuning_option = {
        "tuner": "xgb",
        "trials": 1500,
        "early_stopping": 100,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func="default"), runner=runner
        ),
        "tuning_records": "bert-autotuning.json",
    }

    # 首先从 onnx 模型中提取任务
    tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

    # 按顺序调优提取的任务
    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        tuner_obj = XGBTuner(task, loss_type="rank")
        tuner_obj.tune(
            n_trial=min(tuning_option["trials"], len(task.config_space)),
            early_stopping=tuning_option["early_stopping"],
            measure_option=tuning_option["measure_option"],
            callbacks=[
                autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
                autotvm.callback.log_to_file(tuning_option["tuning_records"]),
            ],
        )

    with autotvm.apply_history_best(tuning_option["tuning_records"]):
        with tvm.transform.PassContext(opt_level=3, config={}):
            lib = relay.build(mod, target=target, params=params)

    return lib


def build_auto_scheduler_tuned_lib(mod, params, target):
    log_file = "bert-autoscheduler-tuning.json"
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=800 * len(tasks),
        runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )
    tuner.tune(tune_option)
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=target, params=params)
    return lib


max_len = 128
target = "llvm -libs=dnnl"
save_path = "save_dir"
onnx_save_path = os.path.join(save_path, "onnx/model.onnx")
tvm_lib_path = os.path.join(save_path, "tvm")
mod, params = convert_onnx_to_relay(onnx_save_path)

lib = build_tvm_lib(mod, params, target)
save_tvm_lib(lib, os.path.join(tvm_lib_path, "bert_tvm.so"))

lib = build_tuned_lib(mod, params, target)
save_tvm_lib(lib, os.path.join(tvm_lib_path, "bert_tune_tvm.so"))

lib = build_auto_scheduler_tuned_lib(mod, params, target)
save_tvm_lib(lib, os.path.join(tvm_lib_path, "bert_autoschduler_tune_tvm.so"))
