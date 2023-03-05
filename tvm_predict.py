import contextlib
import os
import time
import tvm
import numpy as np
from tvm.contrib import graph_executor


@contextlib.contextmanager
def timer(message: str = ""):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(
            "{} - Elapsed time: {:.4f} ms".format(
                message, (end_time - start_time) * 1000
            )
        )


def load_tvm_lib(lib_path: str):
    loaded_lib = tvm.runtime.load_module(lib_path)
    return loaded_lib


def build_tvm_module(target: str, lib):
    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))
    return module


def tvm_infer_once(module, input_name, input_ids):
    module.set_input(input_name, input_ids)
    module.run()
    tvm_output = module.get_output(0).numpy()
    return tvm_output


def tvm_infer(module, input_name, input_ids, times):
    outputs = []
    with timer("tvm infer first time"):
        module.set_input(input_name, input_ids)
        module.run()
        output = module.get_output(0).numpy()
    with timer("tvm infer {} times".format(times)):
        for _ in range(times):
            module.set_input(input_name, input_ids)
            module.run()
            output = module.get_output(0).numpy()
            outputs.append(output)
    return outputs


times = 1000
max_len = 128
target = "llvm -libs=dnnl"
save_path = "save_dir"
input_name = "onnx::Gather_0"
tvm_lib_path = os.path.join(save_path, "tvm")
input_sample = [i for i in range(3000, 3000 + max_len - 2)]
input_sample = [101, *input_sample, 102]
input_sample = np.array([[input_sample]])

# untune lib
lib = load_tvm_lib(os.path.join(tvm_lib_path, "bert_tvm.tar"))
module = build_tvm_module(target, lib)
outputs = tvm_infer(module, input_name, input_sample, times)
print(outputs[0])

# autotvm tuned
lib = load_tvm_lib(os.path.join(tvm_lib_path, "bert_tvm_tuned.tar"))
module = build_tvm_module(target, lib)
outputs = tvm_infer(module, input_name, input_sample, times)
print(outputs[0])

# autoscheduler tuned
lib = load_tvm_lib(os.path.join(tvm_lib_path, "bert_tvm_autoscheduler_tuned.tar"))
module = build_tvm_module(target, lib)
outputs = tvm_infer(module, input_name, input_sample, times)
print(outputs[0])
